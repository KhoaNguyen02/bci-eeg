import argparse
import json
import torch
import numpy as np
import random
import os
from src.data_loader import MotorImageryDataLoader
from src.networks.eegnet import EEGNet
from src.networks.decor_eegnet import DecorrelatedEEGNet
from src.train import train_model


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_model(model_type, config):
    """Create model based on type and config"""
    model_config = config['model']
    base_params = {
        'chunk_size': config['chunk_size'],
        'n_channels': config['num_channels'],
        'num_classes': config['num_classes'],
        **{k: v for k, v in model_config.items() if k != 'decor_lr' and k != 'kappa'}
    }

    if model_type == "decorr_eegnet":
        return DecorrelatedEEGNet(**base_params,
                                  decor_lr=model_config['decor_lr'],
                                  kappa=model_config['kappa'])
    return EEGNet(**base_params)


def main():
    parser = argparse.ArgumentParser(description='Train EEG Motor Imagery Classification Models')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--model', type=str,
                        choices=['eegnet', 'decorr_eegnet'], required=True)
    parser.add_argument('--seed', type=int, help='Random seed')
    args = parser.parse_args()

    # Load config and setup
    with open(f"config/{args.config}", 'r') as f:
        config = json.load(f)

    if args.seed is not None:
        set_seed(args.seed)
        config['random_state'] = args.seed
        print(f"Set random seed to: {args.seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f"Using device: {device} | Model: {args.model} | Config: {args.config}")

    # Load data
    print("Loading data...")
    data_loader = MotorImageryDataLoader(
        data_dir=config['data_dir'], chunk_size=config['chunk_size'],
        num_channel=config['num_channels'], test_size=config['test_size'],
        val_size=config['val_size'], random_state=config['random_state'],
        io_path=config.get('io_path'), split_path=config.get('split_path')
    )

    train_loader, val_loader, test_loader = data_loader.get_loaders(
        batch_size=config['batch_size'])

    # Train model
    print("\nStarting training...")
    results = train_model(
        model=create_model(args.model, config),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, epochs=config['epochs'], learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'], early_stopping_patience=config['early_stopping_patience'],
        wandb_project=config['wandb_project'], model_type=args.model,
        config_name=args.config, seed=args.seed
    )

    print(f"\nTraining completed !!!")
    print(f"Best val acc: {results['best_val_acc']:.2f}% | Test acc: {results['test_acc']:.2f}%")
    print(f"Model saved to: {results['model_path']}")


if __name__ == "__main__":
    main()