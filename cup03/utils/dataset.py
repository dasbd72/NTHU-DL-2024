import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


@dataclass
class DatasetToolConfig:
    id2word_path: str = "./dataset/dictionary/id2Word.npy"
    vocab_path: str = "./dataset/dictionary/vocab.npy"
    word2id_path: str = "./dataset/dictionary/word2Id.npy"
    image_path: str = "./dataset"
    text2img_path: str = "./dataset/dataset/text2ImgData.pkl"
    test_data_path: str = "./dataset/dataset/testData.pkl"


class Text2ImgDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        split=Literal["train", "val", "test"],
    ):
        self.transform = transform
        self.split = split

        # One string of caption per image
        self.captions: list[str] = []
        if split in ["train", "val"]:
            self.image_paths: list[str] = []
        if split in ["test"]:
            self.indices: list[int] = []
        for _, row in df.iterrows():
            if split in ["train", "val"]:
                captions = row["Captions"]
                image_path = row["ImagePath"]
                self.captions.extend(captions)
                self.image_paths.extend([image_path] * len(captions))
            elif split in ["test"]:
                index = row["ID"]
                caption = row["Captions"]
                self.captions.append(caption)
                self.indices.append(index)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str]:
        if self.split in ["train", "val"]:
            caption = self.captions[idx]
            image_path = self.image_paths[idx]

            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

            return image, caption
        elif self.split in ["test"]:
            indices = self.indices[idx]
            caption = self.captions[idx]
            return indices, caption


class DatasetTool:
    def __init__(self, cfg: DatasetToolConfig):
        assert os.path.exists(
            cfg.id2word_path
        ), f"File not found: {cfg.id2word_path}"
        assert os.path.exists(
            cfg.vocab_path
        ), f"File not found: {cfg.vocab_path}"
        assert os.path.exists(
            cfg.word2id_path
        ), f"File not found: {cfg.word2id_path}"
        assert os.path.isdir(
            cfg.image_path
        ), f"Directory not found: {cfg.image_path}"
        assert os.path.exists(
            cfg.text2img_path
        ), f"File not found: {cfg.text2img_path}"
        assert os.path.exists(
            cfg.test_data_path
        ), f"File not found: {cfg.test_data_path}"

        self.cfg = cfg

        self.id2word, self.vocab, self.word2id, self.df_train, self.df_test = (
            self.load_and_process_data()
        )

    def load_and_process_data(self):
        cfg = self.cfg

        id2word = np.load(cfg.id2word_path)
        id2word_dict = dict(id2word)
        vocab = np.load(cfg.vocab_path)
        word2id = np.load(cfg.word2id_path)
        word2id_dict = dict(word2id)
        df_train = pd.read_pickle(cfg.text2img_path)
        df_test = pd.read_pickle(cfg.test_data_path)

        id2word_dict[word2id_dict["<PAD>"]] = ""
        # train data has "ID", "Captions", and "ImagePath" column
        # the "Captions" column is a list of list of integers
        # where each integer is a word ID
        # the "ImagePath" column is a string
        # Convert the "Captions" column back to a list of list of strings
        df_train["Captions"] = df_train["Captions"].apply(
            lambda x: [
                " ".join([id2word_dict[i] for i in caption]).strip()
                for caption in x
            ]
        )
        df_train["ImagePath"] = df_train["ImagePath"].apply(
            lambda x: os.path.join(cfg.image_path, x)
        )
        df_test["Captions"] = df_test["Captions"].apply(
            lambda x: " ".join([id2word_dict[i] for i in x]).strip()
        )
        return id2word, vocab, word2id, df_train, df_test

    def get_train_transform(self, image_size: int):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(2),
                transforms.ToTensor(),
            ]
        )

    def get_val_transform(self, image_size: int):
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def get_train_val_loader(
        self,
        batch_size: int,
        image_size: int,
        train_val_ratio: float = 0.8,
        repeats: int = 1,
        shuffle: bool = False,
        pin_memory: bool = True,
        num_workers: int = 4,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        distributed: bool = False,
    ):
        generator = np.random.default_rng(seed)
        idx = np.arange(len(self.df_train))
        generator.shuffle(idx)
        train_idx, val_idx = np.split(idx, [int(train_val_ratio * len(idx))])
        df_train, df_val = (
            self.df_train.iloc[train_idx],
            self.df_train.iloc[val_idx],
        )

        train_transform = self.get_train_transform(image_size)
        val_transform = self.get_val_transform(image_size)

        train_dataset = Text2ImgDataset(
            df_train, transform=train_transform, split="train"
        )
        train_dataset = ConcatDataset(
            [train_dataset] * repeats
        )  # repeat the dataset
        val_dataset = Text2ImgDataset(
            df_val, transform=val_transform, split="val"
        )

        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
            )

            shuffle = None

        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=val_sampler,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader

    def get_train_loader(
        self,
        batch_size: int,
        image_size: int,
        repeats: int = 1,
        shuffle: bool = False,
        drop_last: bool = True,
        pin_memory: bool = True,
        num_workers: int = 4,
        rank: int = 0,
        world_size: int = 1,
        distributed: bool = False,
    ):
        transform = self.get_train_transform(image_size)

        dataset = Text2ImgDataset(
            self.df_train, transform=transform, split="train"
        )
        dataset = ConcatDataset([dataset] * repeats)

        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
            )
            shuffle = None
        else:
            sampler = None

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory,
        )

        return dataloader

    def get_test_loader(
        self,
        batch_size: int,
        image_size: int,
        shuffle: bool = False,
        pin_memory: bool = True,
        num_workers: int = 4,
        rank: int = 0,
        world_size: int = 1,
        distributed: bool = False,
    ):
        transform = self.get_val_transform(image_size)

        dataset = Text2ImgDataset(
            self.df_test, transform=transform, split="test"
        )

        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
            )
            shuffle = None
        else:
            sampler = None

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory,
        )

        return dataloader


if __name__ == "__main__":
    cfg = DatasetToolConfig()
    dataset_tool = DatasetTool(cfg)
    train_loader, val_loader = dataset_tool.get_train_val_loader(
        batch_size=8, image_size=128
    )
    test_loader = dataset_tool.get_test_loader(batch_size=8, image_size=128)

    for i, (images, captions) in enumerate(train_loader):
        print(images.shape, captions)
        if i == 0:
            break

    for i, captions in enumerate(test_loader):
        print(captions)
        if i == 0:
            break

    print("Passed!")
