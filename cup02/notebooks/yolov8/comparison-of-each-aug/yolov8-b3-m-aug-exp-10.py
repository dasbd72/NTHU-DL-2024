# %% [markdown]
# # DataLab Cup 2: CNN for Object Detection
#
# Sao-Hsuan Lin
#
# 113062532

# %%
import os
from uuid import uuid4
from models.yolov8.layers import (
    YoloV8,
    YoloV8WithResNet,
    YoloV8WithEfficientNetB3,
    YoloV8WithDenseNet121,
)

# ID
# TODO: change ID to the model name or experiment name
ID = uuid4().hex[:8]
# ID = "yolov8-b3-m-v2"
ID = "yolov8-b3-m-aug-exp-10"

# common params
# TODO: change DEVICE
DEVICE = "cuda:0"  # "cuda:i" or "cpu"
OMP_NUM_THREADS = 10
SEED = 42

# dataset params
# TODO: change the path to training data
TRAIN_DATA_PATH = "./dataset/pascal_voc_training_data.txt"
TRAIN_IMAGE_DIR = "./dataset/VOCdevkit_train/VOC2007/JPEGImages/"
# TRAIN_DATA_PATH = "./dataset/augmented_data.txt"
# TRAIN_IMAGE_DIR = "./dataset/AugmentedImage/"
# TRAIN_DATA_PATH = "./dataset/augmented_data_yolov8_1.txt"
# TRAIN_IMAGE_DIR = "./dataset/AugmentedImageYoloV8_1/"
TEST_DATA_PATH = "./dataset/pascal_voc_testing_data.txt"
TEST_IMAGE_DIR = "./dataset/VOCdevkit_test/VOC2007/JPEGImages/"

# model params I
IMAGE_SIZE = 640
BATCH_SIZE = 16
NUM_CLASSES = 20
REG_MAX = 16
MAX_OBJECTS_PER_IMAGE = 100

# model params II
# TODO: change the model class to the yolov8 model you want to train
MODEL_CLS = YoloV8WithEfficientNetB3
# TODO: change the model size, options: "n", "s", "m", "l", "x"
MODEL_SIZE = "m"
# TODO: change the model and weights
BOX_LOSS_WEIGHT = 7.5
CLS_LOSS_WEIGHT = 3.5
DFL_LOSS_WEIGHT = 1.5

# training params
# TODO: change the number of epochs
START_EPOCH = 0
EPOCHS = 1
# TODO: change lr
LEARNING_RATE = 1e-3
# TODO: change if freeze backbone or training all
FREEZE_BACKBONE = True

# checkpoint params
CHECKPOINT_DIR = os.path.join("./ckpts/", ID)
CHECKPOINT_NAME = "yolov8_checkpoint"

# evaluation params
OUTPUT_DIR = os.path.join("./output/", ID)

# %%
import os
import random
import torch
import warnings

# Check if CUDA is available
if torch.cuda.is_available():
    gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {gpus}")
    device = torch.device(DEVICE)
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")
print(f"Device: {device}")

os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)

random.seed(SEED)
torch.manual_seed(SEED)

warnings.filterwarnings("ignore")

# %%
if MODEL_SIZE == "n":
    model = MODEL_CLS.get_yolo_v8_n(
        num_classes=NUM_CLASSES,
        reg_max=REG_MAX,
        pred_max=MAX_OBJECTS_PER_IMAGE,
    )
elif MODEL_SIZE == "m":
    model = MODEL_CLS.get_yolo_v8_m(
        num_classes=NUM_CLASSES,
        reg_max=REG_MAX,
        pred_max=MAX_OBJECTS_PER_IMAGE,
    )
elif MODEL_SIZE == "x":
    model = MODEL_CLS.get_yolo_v8_x(
        num_classes=NUM_CLASSES,
        reg_max=REG_MAX,
        pred_max=MAX_OBJECTS_PER_IMAGE,
    )
model = model.to(device)

trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
total_params = sum(p.numel() for p in model.parameters())
print(
    "Trainable parameters: {}\nTotal parameters: {}\nRatio: {:5.3f}".format(
        trainable_params,
        total_params,
        trainable_params / total_params,
    )
)

# %%
from torch.utils.data import DataLoader, random_split
from models.yolov8.data import (
    TrainDatasetGenerator,
    AugmentedTrainDatasetGenerator,
    train_collate_fn,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2


# TODO:
class ModdedAugmentedTrainDatasetGenerator(AugmentedTrainDatasetGenerator):
    def __init__(self, data_path, image_dir, image_size):
        super().__init__(data_path, image_dir, image_size)
        # Define image transformations
        shape_transforms = [
            # A.HorizontalFlip(p=0.5),
            A.RandomCrop(width=420, height=420, p=0.3),
            # A.Affine(
            #     scale=(0.7, 1.3),
            #     rotate=(-30, 30),
            #     shear=(-10, 10),
            #     p=0.1,
            #     mode=cv2.BORDER_CONSTANT,
            # ),
        ]
        self.transform_no_pos = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.1),
        )
        self.transform = A.Compose(
            [
                A.Resize(self.image_size, self.image_size),
                *shape_transforms,
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.1),
        )


def create_data_loader(
    data_path,
    image_dir,
    batch_size,
    image_size,
    shuffle=True,
    num_workers=12,
    pin_memory=False,
    drop_last=False,
    device: str = "",
):
    # dataset = TrainDatasetGenerator(data_path, image_dir, image_size)
    # dataset = AugmentedTrainDatasetGenerator(data_path, image_dir, image_size)
    dataset = ModdedAugmentedTrainDatasetGenerator(
        data_path, image_dir, image_size
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        pin_memory_device=device,
        collate_fn=train_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        pin_memory_device=device,
        collate_fn=train_collate_fn,
    )
    return train_loader, val_loader


# Plot some images from the training loader
from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.data import CLASS_NAMES

ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic("matplotlib", "inline")
    train_loader, val_loader = create_data_loader(
        TRAIN_DATA_PATH,
        TRAIN_IMAGE_DIR,
        BATCH_SIZE,
        IMAGE_SIZE,
        pin_memory=True,
    )

    idx = 0
    plt.figure(figsize=(20, 8))
    for images, targets in train_loader:
        images = images.cpu().numpy()
        labels = targets.cpu().numpy()

        images = ((images + 1) / 2 * 255).astype(np.uint8)
        images = images.transpose(0, 2, 3, 1)
        labels[:, 1:5] = labels[:, 1:5] * IMAGE_SIZE
        for batch_idx in range(images.shape[0]):
            image = images[batch_idx].copy()
            for x1, y1, x2, y2, cls in labels[labels[:, 0] == batch_idx][
                :, 1:
            ]:
                image = cv2.rectangle(
                    image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 255),
                    5,
                )
                txt = "{}".format(CLASS_NAMES[int(cls)])
                cv2.putText(
                    image,
                    txt,
                    (int(x1) + 10, int(y1) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    6,
                )
                cv2.putText(
                    image,
                    txt,
                    (int(x1) + 10, int(y1) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    4,
                )
            plt.subplot(2, 5, idx + 1)
            plt.imshow(image)
            plt.axis("off")
            plt.title(f"Batch {batch_idx}")
            idx += 1
            if idx >= 10:
                break
        if idx >= 10:
            break
    plt.show()

# %% [markdown]
# ## Training

# %%
import os
import math
import time
import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler
from datetime import datetime
from utils.training import load_checkpoint, save_checkpoint
from models.yolov8.layers import YoloV8, YoloV8Loss
from models.yolov8.evaluate import predict_and_evaluate


# Training step function
def train_step(
    model: YoloV8,
    optimizer: optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Zero out gradients
    outputs = model(images)  # Forward pass
    loss = yolo_loss(outputs, labels)  # Compute loss
    loss_metric = loss.item()
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
    return loss_metric


def val_step(model: YoloV8, images: torch.Tensor, labels: torch.Tensor):
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        outputs = model(images)
        loss = yolo_loss(outputs, labels)
        loss_metric = loss.item()
    return loss_metric


# Directory for saving checkpoints
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_loader, val_loader = None, None
yolo_loss = None
optimizer = None
scheduler = None
error_breaking = False

if EPOCHS > 0:
    # Initialize objects if need training
    train_loader, val_loader = create_data_loader(
        TRAIN_DATA_PATH,
        TRAIN_IMAGE_DIR,
        BATCH_SIZE,
        IMAGE_SIZE,
        pin_memory=True,
    )
    yolo_loss = YoloV8Loss(
        model,
        box_loss_weight=BOX_LOSS_WEIGHT,
        cls_loss_weight=CLS_LOSS_WEIGHT,
        dfl_loss_weight=DFL_LOSS_WEIGHT,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # Load checkpoint if available
    if START_EPOCH > 0:
        load_checkpoint(
            model, CHECKPOINT_DIR, CHECKPOINT_NAME, optimizer, START_EPOCH
        )

    # Force set lr
    optimizer.param_groups[0]["lr"] = LEARNING_RATE

    # Freeze/Unfreeze backbone
    if FREEZE_BACKBONE:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    # Print parameters
    print("=== Info ===")
    print("Training ID: {}".format(ID))
    print("Training device: {}".format(str(device)))
    print("=== Dataset ===")
    print("Dataset path: {}".format(TRAIN_DATA_PATH))
    print("Training on {} images".format(len(train_loader.dataset)))
    print("Validating on {} images".format(len(val_loader.dataset)))
    print("Image size: {}".format(IMAGE_SIZE))
    print("Batch size: {}".format(BATCH_SIZE))
    print("Number of classes: {}".format(NUM_CLASSES))
    print("=== Model ===")
    print("Model class: {}".format(MODEL_CLS.__name__))
    print("Model size: {}".format(MODEL_SIZE))
    print("Box loss weight: {}".format(BOX_LOSS_WEIGHT))
    print("Class loss weight: {}".format(CLS_LOSS_WEIGHT))
    print("DFL loss weight: {}".format(DFL_LOSS_WEIGHT))
    print("=== Training ===")
    print("Start epoch: {}".format(START_EPOCH))
    print("Number of epochs: {}".format(EPOCHS))
    print("Learning rate: {}".format(LEARNING_RATE))
    print("Freeze backbone: {}".format(FREEZE_BACKBONE))


# Training loop
for epoch in range(START_EPOCH, EPOCHS + START_EPOCH):
    print(
        "{} - epoch {:3d}/{:3d}".format(
            datetime.now(), epoch + 1, EPOCHS + START_EPOCH
        )
    )
    start_ts = time.perf_counter()
    loss_metric_list = []
    loss_detail_list = []
    val_loss_metric_list = []

    for idx, (images, labels) in enumerate(train_loader):
        images, labels = (
            images.to(device),
            labels.to(device),
        )
        loss_metric = train_step(model, optimizer, images, labels)
        loss_metric_list.append(loss_metric)
        loss_detail_list.append(yolo_loss._prev_loss.cpu().detach().numpy())

        if (
            math.isnan(loss_metric)
            or math.isinf(loss_metric)
            or loss_metric < 0
            or (yolo_loss._prev_loss < 0).any()
        ):
            print("Loss is {:.4f}, stop training.".format(loss_metric))
            error_breaking = True
            break

        if idx % 100 == 0:
            lr = scheduler.get_lr()[0]
            print(
                "epoch {:3d}/{:3d}, batch: {:4d}/{:4d}, loss {:10.4f} [{:3f}, {:3f}, {:3f}], lr {:10.4e}".format(
                    epoch + 1,
                    EPOCHS + START_EPOCH,
                    idx + 1,
                    len(train_loader),
                    loss_metric,
                    yolo_loss._prev_loss[0],
                    yolo_loss._prev_loss[1],
                    yolo_loss._prev_loss[2],
                    lr,
                )
            )

    if error_breaking:
        break

    # Scheduler step after each epoch
    # scheduler.step()

    # Save checkpoint
    save_checkpoint(
        epoch + 1, model, optimizer, CHECKPOINT_DIR, CHECKPOINT_NAME
    )

    # Validation
    for idx, (images, labels) in enumerate(val_loader):
        images, labels = (
            images.to(device),
            labels.to(device),
        )
        val_loss_metric = val_step(model, images, labels)
        val_loss_metric_list.append(val_loss_metric)

    # Print info
    avg_train_loss = sum(loss_metric_list) / len(loss_metric_list)
    ave_train_loss_detail = np.mean(loss_detail_list, axis=0)
    avg_val_loss = sum(val_loss_metric_list) / len(val_loss_metric_list)
    lr = scheduler.get_last_lr()[0]
    print(
        "epoch {:3d}/{:3d}, train loss {:10.4f}, {}, val loss {:10.4f}, lr {:10.4e}, time {:.2f}s".format(
            epoch + 1,
            EPOCHS + START_EPOCH,
            avg_train_loss,
            ave_train_loss_detail,
            avg_val_loss,
            lr,
            time.perf_counter() - start_ts,
        )
    )

    # Evaluation
    conf_ratio = 0.1
    pred_output_path = os.path.join(
        OUTPUT_DIR, f"yolo_predictions_{epoch:03d}_{conf_ratio:.2f}.csv"
    )
    eval_output_path = os.path.join(
        OUTPUT_DIR, f"yolo_eval_results_{epoch:03d}_{conf_ratio:.2f}.csv"
    )
    score = predict_and_evaluate(
        model,
        TEST_DATA_PATH,
        TEST_IMAGE_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        pred_output_path,
        eval_output_path,
        pin_memory=True,
        conf_ratio=conf_ratio,
        device=device,
    )
    print(
        "epoch {:3d}/{:3d}, test mAP {:.4f}".format(
            epoch + 1, EPOCHS + START_EPOCH, score
        )
    )

# %% [markdown]
# ## Predict Test data

# %%
import random
from IPython import get_ipython
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.data import CLASS_NAMES
from models.yolov8.layers import YoloV8
from models.yolov8.utils import process_outputs


def predict_draw(image_path, model: YoloV8):
    np_img = cv2.imread(image_path)
    np_img = cv2.resize(np_img, (IMAGE_SIZE, IMAGE_SIZE))
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    resized_img = np_img
    np_img = np_img.astype(np.float32)
    np_img = np_img / 255.0 * 2 - 1
    np_img = np.reshape(np_img, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    np_img = np.transpose(np_img, (0, 3, 1, 2))

    with torch.no_grad():
        model.eval()
        y_pred = (
            model.inference(torch.tensor(np_img).to(device)).cpu().detach()
        )
    bbox_list, class_list, conf_list = process_outputs(
        y_pred, IMAGE_SIZE, conf_threshold=0.0, conf_ratio=0.01
    )

    bboxes, classes, confidences = (
        bbox_list[0],
        class_list[0],
        conf_list[0],
    )
    for idx, (bbox, class_idx, conf) in enumerate(
        zip(bboxes, classes, confidences)
    ):
        if idx >= 4:
            break
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(
            resized_img,
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            (0, 255, 255),
            2,
        )
        txt = f"{CLASS_NAMES[class_idx]}: {conf:.2f}"
        cv2.putText(
            resized_img,
            txt,
            (int(xmin) + 8, int(ymin) + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_8,
        )
        cv2.putText(
            resized_img,
            txt,
            (int(xmin) + 8, int(ymin) + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 0),
            2,
            cv2.LINE_8,
        )
    return resized_img


def visualize(data_path, image_dir, model, shuffle=False):
    # Retrieve image names
    image_names = []
    with open(data_path, "r") as f:
        for line in f:
            image_names.append(line.strip().split(" ")[0])
    image_names = (
        random.sample(image_names, 10) if shuffle else image_names[:10]
    )

    results = []
    for image_name in tqdm(image_names):
        image_path = os.path.join(image_dir, image_name)
        image = predict_draw(image_path, model)
        results.append((image, image_name))

    plt.figure(figsize=(20, 8))
    for i, (img, title) in enumerate(results):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
    plt.show()


ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic("matplotlib", "inline")

    # Load model from checkpoint
    load_checkpoint(model, CHECKPOINT_DIR, CHECKPOINT_NAME, epoch=None)
    visualize(TRAIN_DATA_PATH, TRAIN_IMAGE_DIR, model, shuffle=True)
    visualize(TEST_DATA_PATH, TEST_IMAGE_DIR, model)
