import os

import torch


def save_checkpoint(epoch, model, optimizer, checkpoint_dir, checkpoint_name):
    """
    Save a checkpoint to a specified directory.

    :param epoch: The epoch number to save in the checkpoint.
    :param model: The model to save in the checkpoint.
    :param optimizer: The optimizer to save in the checkpoint.
    :param checkpoint_dir: The directory to save the checkpoint in.
    :param checkpoint_name: The name of the checkpoint file, will be appended with the epoch number.
    """
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{checkpoint_name}_{epoch:03d}.pt"
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint for epoch {epoch} at {checkpoint_path}")


def load_checkpoint(
    model,
    checkpoint_dir,
    checkpoint_name,
    optimizer=None,
    epoch=None,
    device=None,
) -> bool:
    """
    Load a checkpoint from a specified directory.

    :param model: The model to load the checkpoint into.
    :param checkpoint_dir: The directory to search for the checkpoint.
    :param checkpoint_name: The name of the checkpoint file, will be appended with the epoch number.
    :param optimizer: The optimizer to load the checkpoint into. If None, the optimizer is not loaded.
    :param epoch: The epoch to load the checkpoint from. If None, the latest checkpoint is loaded.
    If no checkpoint is found, the function prints a message and returns.
    """
    if epoch is None:
        # Search for the latest checkpoint
        files = os.listdir(checkpoint_dir)
        files = [
            file
            for file in files
            if file.startswith(checkpoint_name) and file.endswith(".pt")
        ]
        if not files:
            print("No checkpoint found.")
            return False
        files.sort()
        checkpoint_path = os.path.join(checkpoint_dir, files[-1])
    else:
        checkpoint_path = os.path.join(
            checkpoint_dir, f"{checkpoint_name}_{epoch:03d}.pt"
        )
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint for epoch {epoch} not found.")
        return False
    checkpoint = torch.load(
        checkpoint_path, weights_only=False, map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return True
