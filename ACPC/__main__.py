import argparse
import sys
from train import train_model
from test import test_model

def run(args: argparse.Namespace) -> None:
    if args.mode.lower() == 'train':
        train_model(args.model)
    elif args.mode.lower() == 'test':
        test_model(args.model)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        type=str,
                        help="Train or test",
                        choices=['train', 'test'],
                        required=True)
    
    parser.add_argument("--model",
                        type=str,
                        help='unet or highresnet',
                        choices=['unet', 'highresnet'],
                        required=True)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(args)