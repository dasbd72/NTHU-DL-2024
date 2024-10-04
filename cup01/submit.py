import os
import argparse


class Args:
    file: str
    message: str = "Auto submit"
    competition: str = "2024-datalab-cup1"
    dry_run: bool = False


parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--dry-run",
    action="store_true",
    help="Dry run",
)
parser.add_argument(
    "-c",
    "--competition",
    type=str,
    help="Competition name",
    default=Args.competition,
)
parser.add_argument(
    "-m", "--message", type=str, help="Message to submit", default=Args.message
)
parser.add_argument("file", type=str, help="File to submit")
args: Args = parser.parse_args(namespace=Args)

if __name__ == "__main__":
    if args.file is None:
        raise ValueError("File is required")
    if args.message is None:
        raise ValueError("Message is required")
    if not os.path.exists(args.file):
        raise FileNotFoundError(f"File {args.file} not found")
    cmd = f'kaggle competitions submit -c 2024-datalab-cup1 -f {args.file} -m "{args.message}"'
    print(cmd)
    # Confirm
    ret = input("Submit? [y/n]: ")
    if ret.lower() != "y":
        print("Aborted")
        exit(0)
    if not args.dry_run:
        os.system(cmd)
