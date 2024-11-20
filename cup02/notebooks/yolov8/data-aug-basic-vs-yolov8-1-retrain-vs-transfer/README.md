# data-aug-basic-vs-yolov8-1-retrain-vs-transfer

This notebooks are comparisons between datasets.

## Datasets

- Data Aug Basic
  - One of data_aug_basic.ipynb
- Data Aug Yolov8-1
  - data-aug-yolov8-dasbd72.ipynb
  - Adds cutmix and mosaic to data aug basic

## Experiments

- Data Aug Basic Epoch 1 to 20
  - 1 to 10
    - Best 0.4199
    - Converges to 0.4133
  - 11 to 20
    - Best 0.3505
    - Converges to 0.3505
- Data Aug Yolov8-1 Epoch 1 to 10
  - 1 to 10
    - Best 0.4409
    - Convergence Converge to 0.4409
- Data Aug Basic Epoch 1 to 10 and Yolov8-1 Epoch 11 to 20
  - 11 to 20
    - Best 0.3541
    - Convergence Goes to 0.3541 at 13 epoch but goes up 0.39 to 0.41 at 14 to 20 epoch
