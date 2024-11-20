import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TrainDatasetGenerator(Dataset):
    """
    Load PascalVOC 2007 dataset and create an input pipeline.
    - Reshapes images into 448 x 448
    - Converts [0, 1] to [-1, 1]
    - Supports shuffling and batching with DataLoader
    """

    def __init__(
        self, data_path, image_dir, max_objects_per_image, image_size
    ):
        self.image_names = []
        self.record_list = []
        self.object_num_list = []
        self.max_objects_per_image = max_objects_per_image
        self.image_size = image_size
        self.image_dir = image_dir

        # Filling the record_list
        with open(data_path, "r") as input_file:
            for line in input_file:
                line = line.strip()
                ss = line.split(" ")
                if len(ss) < 6:
                    continue
                self.image_names.append(ss[0])
                self.record_list.append([float(num) for num in ss[1:]])
                self.object_num_list.append(
                    min(len(self.record_list[-1]) // 5, max_objects_per_image)
                )

                # Padding or cropping the list as needed
                if len(self.record_list[-1]) < max_objects_per_image * 5:
                    self.record_list[-1] += [0.0, 0.0, 0.0, 0.0, 0.0] * (
                        max_objects_per_image - len(self.record_list[-1]) // 5
                    )
                elif len(self.record_list[-1]) > max_objects_per_image * 5:
                    self.record_list[-1] = self.record_list[-1][
                        : max_objects_per_image * 5
                    ]

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
        object_num = self.object_num_list[idx]

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
        xcenter_scaled = (xmin + xmax) / 2.0 * width_ratio
        ycenter_scaled = (ymin + ymax) / 2.0 * height_ratio
        box_w_scaled = (xmax - xmin) * width_ratio
        box_h_scaled = (ymax - ymin) * height_ratio
        scaled_labels = torch.stack(
            [
                xcenter_scaled,
                ycenter_scaled,
                box_w_scaled,
                box_h_scaled,
                class_num,
            ],
            dim=1,
        )

        return image, scaled_labels, object_num


def flip_image_and_boxes(image, boxes):
    flipped_image = cv2.flip(image, 1)
    img_width = image.shape[1]

    flipped_boxes = []
    for x_min, y_min, x_max, y_max, label in boxes:
        new_x_min = img_width - x_max
        new_x_max = img_width - x_min
        flipped_boxes.append((new_x_min, y_min, new_x_max, y_max, label))

    return flipped_image, flipped_boxes


def rotate_image_and_boxes(image, boxes, angle=90):
    """Rotates the image and bounding boxes, scaling the image to ensure all content is visible."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # Calculate the rotation matrix with the scaling factor to fit the entire image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])

    # New width and height bounds to fit the rotated image
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # Adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Rotate the image with the new bounding dimensions
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (new_width, new_height)
    )

    rotated_boxes = []
    for x_min, y_min, x_max, y_max, label in boxes:
        # Define the points of the bounding box
        points = np.array(
            [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]
        )
        # Rotate the points
        rotated_points = cv2.transform(np.array([points]), rotation_matrix)[0]

        # Calculate the new bounding box
        new_x_min, new_y_min = np.min(rotated_points, axis=0)
        new_x_max, new_y_max = np.max(rotated_points, axis=0)

        rotated_boxes.append(
            (
                int(new_x_min),
                int(new_y_min),
                int(new_x_max),
                int(new_y_max),
                label,
            )
        )

    return rotated_image, rotated_boxes


def cutmix_single_box(image1, boxes1, image2, boxes2, scale_factor=0.4):
    height, width = image1.shape[:2]

    chosen_box = random.choice(boxes2)
    x_min2, y_min2, x_max2, y_max2, label = chosen_box

    paste_region = image2[y_min2:y_max2, x_min2:x_max2]
    box_w, box_h = x_max2 - x_min2, y_max2 - y_min2

    if paste_region.size == 0:
        return image1, boxes1

    new_w = int(box_w * scale_factor)
    new_h = int(box_h * scale_factor)
    paste_region_resized = cv2.resize(paste_region, (new_w, new_h))

    x1_start = np.random.randint(0, width - new_w)
    y1_start = np.random.randint(0, height - new_h)
    x1_end = x1_start + new_w
    y1_end = y1_start + new_h

    image1[y1_start:y1_end, x1_start:x1_end] = paste_region_resized

    new_boxes = boxes1.copy()
    new_boxes.append((x1_start, y1_start, x1_end, y1_end, label))

    return image1, new_boxes


class AugmentedTrainDatasetGenerator(Dataset):
    """
    Load PascalVOC 2007 dataset and create an input pipeline.
    - Reshapes images into 448 x 448
    - Converts [0, 1] to [-1, 1]
    - Supports shuffling and batching with DataLoader
    """

    def __init__(
        self,
        data_path,
        image_dir,
        max_objects_per_image,
        image_size,
        repeats=3,
        apply_flip=True,
        apply_rotation=True,
        apply_cutmix=False,
    ):
        self.image_names = []
        self.record_list = []
        self.object_num_list = []
        self.max_objects_per_image = max_objects_per_image
        self.image_size = image_size
        self.image_dir = image_dir
        self.apply_flip = apply_flip
        self.apply_rotation = apply_rotation
        self.apply_cutmix = apply_cutmix

        # Filling the record_list
        with open(data_path, "r") as input_file:
            for line in input_file:
                line = line.strip()
                ss = line.split(" ")
                if len(ss) < 6:
                    continue
                self.image_names.append(ss[0])
                self.record_list.append([float(num) for num in ss[1:]])
                self.object_num_list.append(
                    min(len(self.record_list[-1]) // 5, max_objects_per_image)
                )

                # Padding or cropping the list as needed
                if len(self.record_list[-1]) < max_objects_per_image * 5:
                    self.record_list[-1] += [0.0, 0.0, 0.0, 0.0, 0.0] * (
                        max_objects_per_image - len(self.record_list[-1]) // 5
                    )
                elif len(self.record_list[-1]) > max_objects_per_image * 5:
                    self.record_list[-1] = self.record_list[-1][
                        : max_objects_per_image * 5
                    ]

        self.image_names = self.image_names * repeats
        self.record_list = self.record_list * repeats
        self.object_num_list = self.object_num_list * repeats

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
        object_num = self.object_num_list[idx]

        # Load and process image
        image = Image.open(self.image_dir + image_name)
        image = np.array(image)  # Convert to numpy for OpenCV processing

        # Random augmentations
        if self.apply_flip and random.random() < 0.5:
            image, raw_labels = flip_image_and_boxes(image, raw_labels)

        if self.apply_rotation:
            angle = (
                random.random() * 40 - 20
            )  # Random angle between -20 and 20
            image, raw_labels = rotate_image_and_boxes(
                image, raw_labels, angle
            )

        # Resize another image for cutmix if enabled and with a 50% chance
        if self.apply_cutmix and random.random() < 0.5:
            second_idx = random.randint(0, len(self.image_names) - 1)
            if second_idx != idx:
                second_image_name = self.image_names[second_idx]
                second_image = Image.open(self.image_dir + second_image_name)
                second_image = np.array(second_image)
                second_boxes = np.array(self.record_list[second_idx]).reshape(
                    -1, 5
                )
                image, raw_labels = cutmix_single_box(
                    image, raw_labels, second_image, second_boxes
                )

        # Apply final image transformations after augmentations
        image = Image.fromarray(
            image
        )  # Convert back to PIL for torch transform
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
        xcenter_scaled = (xmin + xmax) / 2.0 * width_ratio
        ycenter_scaled = (ymin + ymax) / 2.0 * height_ratio
        box_w_scaled = (xmax - xmin) * width_ratio
        box_h_scaled = (ymax - ymin) * height_ratio
        scaled_labels = torch.stack(
            [
                xcenter_scaled,
                ycenter_scaled,
                box_w_scaled,
                box_h_scaled,
                class_num,
            ],
            dim=1,
        )

        return image, scaled_labels, object_num


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
