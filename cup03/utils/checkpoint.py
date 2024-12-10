import os

import torch


def save_checkpoint(epoch, state_dict, checkpoint_dir, checkpoint_name):
    """
    Save a checkpoint to a specified directory.

    :param epoch: The epoch number to save in the checkpoint.
    :param state_dict: The state_dict to save in the checkpoint.
    :param checkpoint_dir: The directory to save the checkpoint in.
    :param checkpoint_name: The name of the checkpoint file, will be appended with the epoch number.
    """
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{checkpoint_name}_{epoch:03d}.pt"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        state_dict,
        checkpoint_path,
    )
    print(f"Saved checkpoint for epoch {epoch} at {checkpoint_path}")


def load_checkpoint(
    checkpoint_dir,
    checkpoint_name,
    epoch=None,
    device=None,
) -> dict:
    """
    Load a checkpoint from a specified directory.

    :param checkpoint_dir: The directory to search for the checkpoint.
    :param checkpoint_name: The name of the checkpoint file, will be appended with the epoch number.
    :param epoch: The epoch to load the checkpoint from. If None, the latest checkpoint is loaded.
    :param device: The device to load the checkpoint on.
    If no checkpoint is found, the function prints a message and returns.
    """
    if epoch is None:
        if not os.path.isdir(checkpoint_dir):
            print("Checkpoint directory not found.")
            return None
        # Search for the latest checkpoint
        files: list[str]
        files = os.listdir(checkpoint_dir)
        files = [
            file
            for file in files
            if file.startswith(checkpoint_name) and file.endswith(".pt")
        ]
        if not files:
            print("No checkpoint found.")
            return None
        files.sort()
        checkpoint_path = os.path.join(checkpoint_dir, files[-1])
    else:
        checkpoint_path = os.path.join(
            checkpoint_dir, f"{checkpoint_name}_{epoch:03d}.pt"
        )
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint for epoch {epoch} not found.")
        return None
    state_dict = torch.load(
        checkpoint_path, weights_only=False, map_location=device
    )
    print(f"Loaded checkpoint for epoch {epoch} from {checkpoint_path}")
    return state_dict
