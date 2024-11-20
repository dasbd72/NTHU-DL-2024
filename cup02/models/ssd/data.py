import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .ssd import SSD
from .ssd_utils import *

class TrainDatasetGenerator(Dataset):
    """
    Load PascalVOC 2007 dataset and create an input pipeline.
    - Reshapes images into 448 x 448
    - Converts [0, 1] to [-1, 1]
    - Supports shuffling and batching with DataLoader
    """

    def __init__(
        self, data_path, image_dir, image_size
    ):
        self.image_names = []
        self.record_list = []
        # self.object_num_list = []
        # self.max_objects_per_image = max_objects_per_image
        self.image_size = image_size
        self.image_dir = image_dir

        model = SSD()
        self.anchors = model.anchors
        # Filling the record_list
        with open(data_path, "r") as input_file:
            for line in input_file:
                line = line.strip()
                ss = line.split(" ")
                if len(ss) < 6:
                    continue
                self.image_names.append(ss[0])
                self.record_list.append([float(num) for num in ss[1:]])
                # self.object_num_list.append(
                #     min(len(self.record_list[-1]) // 5, max_objects_per_image)
                # )

                # Padding or cropping the list as needed
                # if len(self.record_list[-1]) < max_objects_per_image * 5:
                #     self.record_list[-1] += [0.0, 0.0, 0.0, 0.0, 0.0] * (
                #         max_objects_per_image - len(self.record_list[-1]) // 5
                #     )
                # elif len(self.record_list[-1]) > max_objects_per_image * 5:
                #     self.record_list[-1] = self.record_list[-1][
                #         : max_objects_per_image * 5
                #     ]

        # Define image transformations
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size), antialias=False),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Rescale [0,1] to [-1,1]
            ]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        raw_labels = np.array(self.record_list[idx]).reshape(-1, 5)
        # object_num = self.object_num_list[idx]

        # Load and process image
        image = Image.open(self.image_dir + image_name)
        w, h = image.size
        image = self.transform(image)

        # Process labels
        raw_labels = torch.tensor(raw_labels, dtype=torch.float32)

        xmin = raw_labels[:, 0]
        ymin = raw_labels[:, 1]
        xmax = raw_labels[:, 2]
        ymax = raw_labels[:, 3]
        class_num = raw_labels[:, 4]

        # Ratios for scaling to model input size
        height_ratio = self.image_size / float(h)
        width_ratio = self.image_size / float(w)

        # Scaled (normalized) labels for model input
        xmin_scaled = xmin * width_ratio
        ymin_scaled = ymin * height_ratio
        xmax_scaled = xmax * width_ratio
        ymax_scaled = ymax * height_ratio


        scaled_labels = torch.stack(
            [
                xmin_scaled,
                ymin_scaled,
                xmax_scaled,
                ymax_scaled,
                class_num,
            ],
            dim=1,
        )
        # print(f"height: {float(h)}, imagesize: {self.image_size}, height_ratio: {height_ratio}")
        # print(f"width: {float(w)}, imagesize: {self.image_size}, width_ratio: {height_ratio}")
        # print(raw_labels[0])
        # print(scaled_labels[0].tolist())
        scaled_labels_np = scaled_labels.numpy()
        # print(scaled_labels_np)
        anchors_np = self.anchors.numpy()
        reg_raw_label = scaled_labels_np[:, :4] / self.image_size
        cls_raw_label = scaled_labels_np[:, 4]
        reg_label_np, cls_label_np = match_np( 
                                            reg_raw_label, 
                                            anchors_np, 
                                            cls_raw_label)
        # print(reg_raw_label)
        # print(f"anchor {anchors_np}")
        # Convert reg_label and cls_label back to PyTorch tensors
        reg_label = torch.tensor(reg_label_np, dtype=torch.float32)
        cls_label = torch.tensor(cls_label_np, dtype=torch.int64)
        return image, reg_label, cls_label

class AugmentedTrainDatasetGeneratorV2(Dataset):
    def __init__(
        self,
        data_path,
        image_dir,
        image_size,
        target_class_count=5000,
        max_image_count=50000,
    ):
        model = SSD()
        self.anchors = model.anchors
        self.image_size = image_size
        self.image_dir = image_dir

        # Load image names and corresponding labels from the dataset file
        image_labels: dict[str, list[tuple[int, int, int, int, int]]] = {}
        with open(data_path, "r") as input_file:
            for line in input_file:
                parts = line.strip().split()
                if len(parts) < 6:
                    # Skip images without labels or skip empty lines
                    continue
                image_name = parts[0]
                bboxes = []
                for i in range(1, len(parts), 5):
                    x1, y1, x2, y2, cls = map(int, parts[i : i + 5])
                    bboxes.append((x1, y1, x2, y2, cls))
                image_labels[image_name] = bboxes
        # Create a dictionary of class to image mapping
        class_images: dict[int, list[str]] = {}
        for image_name, bboxes in image_labels.items():
            for x1, y1, x2, y2, cls in bboxes:
                if cls not in class_images:
                    class_images[cls] = []
                class_images[cls].append(image_name)
        # Create a dictionary of class to image mapping
        image_classes: dict[str, set[int]] = {}
        for image_name, bboxes in image_labels.items():
            image_classes[image_name] = set(cls for _, _, _, _, cls in bboxes)

        self.image_names = []
        self.labels = []
        class_count: dict[int, int] = {}
        for cls, _ in class_images.items():
            class_count[cls] = 0
        # Add all images at least once
        for image_name, bboxes in image_labels.items():
            self.image_names.append(image_name)
            self.labels.append(bboxes)
            for _, _, _, _, cls in bboxes:
                class_count[cls] = class_count.get(cls, 0) + 1
        # Add more images to balance the dataset
        while len(self.image_names) < max_image_count:
            if all(cls >= target_class_count for cls in class_count.values()):
                break
            # Select the class with the least number of images
            top_class_count = sorted(
                class_count.items(), key=lambda x: x[1], reverse=True
            )
            top_classes = [cls for cls, _ in top_class_count]
            candidate_images = class_images[top_classes[-1]]
            image_name = random.choice(candidate_images)
            bboxes = image_labels[image_name]
            # Skip the image if it contains the top classes
            if any(
                cls in image_classes[image_name] for cls in top_classes[:3]
            ):
                continue
            # Add the image to the dataset
            self.image_names.append(image_name)
            self.labels.append(bboxes)
            for _, _, _, _, cls in bboxes:
                class_count[cls] = class_count.get(cls, 0) + 1
        appearance_transforms = [
            A.Sharpen(alpha=(0.1, 0.45), lightness=(0.1, 0.45), p=0.5),
            A.SomeOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=(-15, 15),
                        sat_shift_limit=(-20, 20),
                        val_shift_limit=(-15, 15),
                        p=1,
                    ),
                    A.RGBShift(
                        r_shift_limit=(-13, 13),
                        g_shift_limit=(-13, 13),
                        b_shift_limit=(-13, 13),
                        p=1,
                    ),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ],
                n=2,
                p=1,
            ),
        ]
        shape_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(width=300, height=300, p=0.1),
            A.Affine(
                shear=(-10, 10),
                p=0.1,
                mode=cv2.BORDER_CONSTANT,
            ),
            A.Affine(
                rotate=(-30, 30),
                p=1,
                mode=cv2.BORDER_CONSTANT,
            ),
        ]
        dropout_transform = A.CoarseDropout(
            max_holes=12,
            min_height=8,
            max_height=64,
            min_width=8,
            max_width=64,
            fill_value=0,
            p=0.2,
        )
        self.transform_no_pos = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                *appearance_transforms,
                dropout_transform,
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.1),
        )
        self.transform = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                *appearance_transforms,
                *shape_transforms,
                A.Resize(self.image_size, self.image_size),
                dropout_transform,
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.1),
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        bboxes = self.labels[idx]
        # print(bboxes)
        # object_num = self.object_num_list[idx]

        # Load and process image
        image = cv2.imread(self.image_dir + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]  # Extract height and width

        # image = self.transform(image)

        # Process labels
        # raw_labels = torch.tensor(raw_labels, dtype=torch.float32)
        transformed = self.transform(image=image, bboxes=bboxes)
        transformed_image, transformed_bboxes = (
            transformed["image"],
            transformed["bboxes"],
        )
        transformed_bboxes = np.array(transformed_bboxes)
        print(transformed_bboxes.shape, flush=True)
        xmin = transformed_bboxes[:, 0]
        ymin = transformed_bboxes[:, 1]
        xmax = transformed_bboxes[:, 2]
        ymax = transformed_bboxes[:, 3]
        class_num = transformed_bboxes[:, 4]

        # Ratios for scaling to model input size
        height_ratio = self.image_size / float(h)
        width_ratio = self.image_size / float(w)

        # Scaled (normalized) labels for model input
        xmin_scaled = xmin * width_ratio
        ymin_scaled = ymin * height_ratio
        xmax_scaled = xmax * width_ratio
        ymax_scaled = ymax * height_ratio


        scaled_labels = torch.stack(
            [
                torch.tensor(xmin_scaled) if not isinstance(xmin_scaled, torch.Tensor) else xmin_scaled,
                torch.tensor(ymin_scaled) if not isinstance(ymin_scaled, torch.Tensor) else ymin_scaled,
                torch.tensor(xmax_scaled) if not isinstance(xmax_scaled, torch.Tensor) else xmax_scaled,
                torch.tensor(ymax_scaled) if not isinstance(ymax_scaled, torch.Tensor) else ymax_scaled,
                torch.tensor(class_num) if not isinstance(class_num, torch.Tensor) else class_num,
            ],
            dim=1,
        )
        # print(f"height: {float(h)}, imagesize: {self.image_size}, height_ratio: {height_ratio}")
        # print(f"width: {float(w)}, imagesize: {self.image_size}, width_ratio: {height_ratio}")
        # print(raw_labels[0])
        # print(scaled_labels[0].tolist())
        scaled_labels_np = scaled_labels.numpy()
        # print(scaled_labels_np)
        anchors_np = self.anchors.numpy()
        reg_raw_label = scaled_labels_np[:, :4] / self.image_size
        cls_raw_label = scaled_labels_np[:, 4]
        reg_label_np, cls_label_np = match_np( 
                                            reg_raw_label, 
                                            anchors_np, 
                                            cls_raw_label)
        # print(reg_raw_label)
        # print(f"anchor {anchors_np}")
        # Convert reg_label and cls_label back to PyTorch tensors
        reg_label = torch.tensor(reg_label_np, dtype=torch.float32)
        cls_label = torch.tensor(cls_label_np, dtype=torch.int64)
        return transformed_image, reg_label, cls_label
class TestDatasetGenerator(Dataset):
    def __init__(self, data_path, image_dir, image_size):
        self.image_names = []
        self.image_size = image_size
        self.image_dir = image_dir

        with open(data_path, "r") as input_file:
            for line in input_file:
                line = line.strip()
                self.image_names.append(line)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(self.image_dir + image_name).convert("RGB")
        image_height = image.height
        image_width = image.width
        image = self.transform(image)
        return image_name, image, image_height, image_width
