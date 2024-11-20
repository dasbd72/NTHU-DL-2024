import random

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TrainDatasetGenerator(Dataset):
    """
    Load PascalVOC 2007 dataset and create an input pipeline.
    - Reshapes images into specified `image_size`
    - Converts [0, 1] to [-1, 1]
    - Supports shuffling and batching with DataLoader
    """

    def __init__(self, data_path, image_dir, image_size):
        self.image_names = []  # List of strings
        self.labels = (
            []
        )  # List of lists, each containing [x1, y1, x2, y2, class] for each object
        self.image_size = image_size
        self.image_dir = image_dir

        # Load image names and corresponding labels from the dataset file
        with open(data_path, "r") as input_file:
            for line in input_file:
                line = line.strip()
                parts = line.split(" ")
                if len(parts) < 6:
                    print(f"Invalid label for image {parts[0]}")
                    continue
                image_name, labels = parts[0], [
                    float(num) for num in parts[1:]
                ]
                self.image_names.append(image_name)
                # Convert labels to (x1, y1, x2, y2, class) format
                if len(labels) % 5 != 0:
                    # prevent incomplete labels
                    print(f"Invalid label for image {image_name}")
                    labels = labels[: len(labels) - (len(labels) % 5)]
                self.labels.append(
                    [labels[i : i + 5] for i in range(0, len(labels), 5)]
                )

        # Define image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Rescale [0,1] to [-1,1]
            ]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load and process image
        image_name = self.image_names[idx]
        raw_labels = self.labels[idx]
        image = Image.open(self.image_dir + image_name)
        w, h = image.size
        image = self.transform(image)

        # Ratios for scaling to model input size
        height_ratio = self.image_size / float(h)
        width_ratio = self.image_size / float(w)

        # Process labels for the image
        labels = torch.tensor(raw_labels, dtype=torch.float32)  # (n_boxes, 5)

        # Also resize the bounding boxes
        x_min = labels[:, 0]
        y_min = labels[:, 1]
        x_max = labels[:, 2]
        y_max = labels[:, 3]
        class_num = labels[:, 4]

        # Ratios for scaling to model input size
        height_ratio = self.image_size / float(h)
        width_ratio = self.image_size / float(w)

        # Also resize the bounding boxes
        x_center = (x_min + x_max) / 2.0 * width_ratio
        y_center = (y_min + y_max) / 2.0 * height_ratio
        box_width = (x_max - x_min) * width_ratio
        box_height = (y_max - y_min) * height_ratio
        x_min_scaled = x_center - box_width / 2.0
        y_min_scaled = y_center - box_height / 2.0
        x_max_scaled = x_center + box_width / 2.0
        y_max_scaled = y_center + box_height / 2.0

        scaled_labels = torch.stack(
            [
                x_min_scaled,
                y_min_scaled,
                x_max_scaled,
                y_max_scaled,
                class_num,
            ],
            dim=1,
        )  # (n_boxes, 5)

        # Convert (x1, y1, x2, y2) from absolute to relative coordinates based on image size
        scaled_labels[
            :, :4
        ] /= self.image_size  # Scale bounding box coordinates to [0, 1]
        return image, scaled_labels


class AugmentedTrainDatasetGenerator(Dataset):
    """
    Load PascalVOC 2007 dataset and create an input pipeline.
    - Reshapes images into specified `image_size`
    - Converts [0, 1] to [-1, 1]
    - Supports shuffling and batching with DataLoader
    """

    def __init__(
        self,
        data_path,
        image_dir,
        image_size,
        target_class_count=5000,
        max_image_count=50000,
        n_cutmix=2,
        p_cutmix=0.0,
    ):
        self.image_size = image_size
        self.image_dir = image_dir
        self.n_cutmix = n_cutmix
        self.p_cutmix = p_cutmix

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

        print(f"Loaded {len(self.image_names)} images")
        print(f"Class distribution: {class_count}")

        self.class_bboxes = (
            {}
        )  # The bounding boxes to apply cutmix, stores (image_name, x1, y1, x2, y2, cls)
        for image_name, bboxes in image_labels.items():
            for x1, y1, x2, y2, cls in bboxes:
                if cls not in self.class_bboxes:
                    self.class_bboxes[cls] = []
                self.class_bboxes[cls].append(
                    (image_name, x1, y1, x2, y2, cls)
                )

        # Define image transformations
        self.init_transforms()

    def init_transforms(self):
        # Define image transformations
        appearance_transforms = [
            A.Sharpen(alpha=(0.1, 0.4), lightness=(0.1, 0.4), p=0.2),
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
                n=1,
                p=1,
            ),
        ]
        shape_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(width=420, height=420, p=0.1),
            A.Affine(
                shear=(-10, 10),
                p=0.1,
                mode=cv2.BORDER_CONSTANT,
            ),
            A.Affine(
                rotate=(-30, 30),
                p=0.5,
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
            p=0.1,
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
        # Load and process image
        image_name = self.image_names[idx]
        bboxes = self.labels[idx]
        image = cv2.imread(self.image_dir + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply cutmix
        if random.random() < self.p_cutmix:
            image, bboxes = self.random_cutmix(
                image, bboxes, n_cutmix=self.n_cutmix
            )

        transformed = self.transform(image=image, bboxes=bboxes)
        transformed_image, transformed_bboxes = (
            transformed["image"],
            transformed["bboxes"],
        )
        if len(transformed_bboxes) == 0:
            # Empty bboxes after shape transformation, do again with no shape transformation
            transformed = self.transform_no_pos(image=image, bboxes=bboxes)
            transformed_image, transformed_bboxes = (
                transformed["image"],
                transformed["bboxes"],
            )
            assert (
                len(transformed_bboxes) != 0
            ), "Image {} got no bboxes after transformation".format(image_name)
        image, bboxes = transformed_image, transformed_bboxes

        bboxes = [
            np.clip(
                bbox,
                0,
                [
                    self.image_size - 1,
                    self.image_size - 1,
                    self.image_size,
                    self.image_size,
                    19,
                ],
            )
            for bbox in bboxes
        ]
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes[:, :4] /= self.image_size
        return image, bboxes

    def random_cutmix(
        self,
        image: np.ndarray,
        bboxes: list[tuple[int, int, int, int, int]],
        n_cutmix=2,
        max_tries=10,
    ):
        """
        Randomly select sub-images from another image and paste it onto the current image.
        """
        sub_bboxes = []
        for _ in range(n_cutmix):
            # Randomly select a class to paste
            cls = random.choice(list(self.class_bboxes.keys()))
            sub_bboxes.extend(random.sample(self.class_bboxes[cls], 1))

        h, w, _ = image.shape
        for sub_image_name, x1_sub, y1_sub, x2_sub, y2_sub, cls in sub_bboxes:
            sub_image = cv2.imread(self.image_dir + sub_image_name)
            for _ in range(max_tries):
                w_min_sub = max(int((x2_sub - x1_sub) * 0.2), 1)
                h_min_sub = max(int((y2_sub - y1_sub) * 0.2), 1)

                # Randomly select a location to paste the sub-image
                x1 = random.randint(0, w - 1 - w_min_sub)
                y1 = random.randint(0, h - 1 - h_min_sub)
                x2 = min(x1 + x2_sub - x1_sub, w)
                y2 = min(y1 + y2_sub - y1_sub, h)
                x2_sub = x1_sub + x2 - x1
                y2_sub = y1_sub + y2 - y1

                if (
                    self.max_overlapped_ratio(bboxes, (x1, y1, x2, y2, 0))
                    > 0.8
                ):
                    continue

                # Apply cutmix
                image[y1:y2, x1:x2] = sub_image[y1_sub:y2_sub, x1_sub:x2_sub]
                bboxes.append((x1, y1, x2, y2, cls))
                break
        return image, bboxes

    def max_overlapped_ratio(self, bboxes, bbox_target):
        """
        Calculate the maximum overlapped ratio of a bounding box with a list of bounding boxes.
        """
        max_ratio = 0
        x1_target, y1_target, x2_target, y2_target, _ = bbox_target
        for bbox in bboxes:
            x1, y1, x2, y2, _ = bbox
            # Step 1: Calculate intersection
            # Calculate intersection boundaries
            x_left = max(x1, x1_target)
            y_top = max(y1, y1_target)
            x_right = min(x2, x2_target)
            y_bottom = min(y2, y2_target)
            # Check if there is an intersection
            if x_right <= x_left or y_bottom <= y_top:
                continue
            intersection = (x_right - x_left) * (y_bottom - y_top)
            # Step 2: Calculate ratio
            box_area = (x2 - x1) * (y2 - y1)
            ratio = intersection / (box_area)
            max_ratio = max(max_ratio, ratio)
        return max_ratio


class AugmentedTrainDatasetGeneratorV2(AugmentedTrainDatasetGenerator):
    def init_transforms(self):
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
            A.RandomCrop(width=420, height=420, p=0.1),
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


def train_collate_fn(batch):
    """
    Custom collate function to combine labels into a padded tensor of shape (batch_n_boxes, 6),
    where each row is (batch_idx, x1, y1, x2, y2, class)
    """
    images, labels = zip(*batch)
    n_boxes = torch.tensor([label.size(0) for label in labels])

    # Combine images into a single tensor
    images = torch.stack(images)

    # Add batch_idx to labels and combine into a single tensor
    labels = torch.cat(labels, dim=0)  # (batch_n_boxes, 5)
    batch_idx = torch.arange(len(n_boxes)).repeat_interleave(
        n_boxes
    )  # (batch_n_boxes,)
    batch_idx = batch_idx.unsqueeze(1).float()  # (batch_n_boxes, 1)
    labels = torch.cat([batch_idx, labels], dim=1)  # (batch_n_boxes, 6)

    return images, labels


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
