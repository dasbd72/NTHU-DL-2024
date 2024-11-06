import os
import torch


# Function to save checkpoint
def save_checkpoint(epoch, model, optimizer, checkpoint_dir, checkpoint_name):
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


# Function to load checkpoint
def load_checkpoint(
    model, checkpoint_dir, checkpoint_name, optimizer=None, epoch=None
):
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
            return
        files.sort()
        checkpoint_path = os.path.join(checkpoint_dir, files[-1])
    else:
        checkpoint_path = os.path.join(
            checkpoint_dir, f"{checkpoint_name}_{epoch:03d}.pt"
        )
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint for epoch {epoch} not found.")
        return
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path}")
