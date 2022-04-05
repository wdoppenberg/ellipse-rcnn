import argparse

import mlflow
import torch

from src import CraterDetector


def get_parser():
    parser = argparse.ArgumentParser(description='Save core weights to file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_id', type=str, default=None, nargs='?',
                        help='Resume from MLflow run checkpoint')
    parser.add_argument('--output_name', type=str, default="CraterRCNN", nargs='?',
                        help='Resume from MLflow run checkpoint')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment("crater-core")

    model = CraterDetector()
    checkpoint = mlflow.pytorch.load_state_dict(fr"runs:/{args.run_id}/checkpoint")
    model.load_state_dict(checkpoint['model_state_dict'])

    torch.save(model.state_dict(), fr"blobs/{args.output_name}.pth")
