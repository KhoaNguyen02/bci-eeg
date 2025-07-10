import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import time
import os
from torchsummary import summary
from src.networks.decor_eegnet import decor_modules, decor_update


class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            return self.counter >= self.patience
        return False


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            total_loss += criterion(output, target).item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return 100. * correct / total, total_loss / len(data_loader)


def train_model(model, train_loader, val_loader, test_loader, device,
                epochs=50, learning_rate=0.001, weight_decay=1e-4,
                early_stopping_patience=15, wandb_project='bci-eeg',
                model_type=None, config_name=None, seed=None):

    # Setup wandb
    run_name = f"{model_type or model.__class__.__name__}_{int(time.time())}"
    if seed:
        run_name += f"_seed{seed}"

    wandb.init(project=wandb_project, name=run_name, tags=[model_type or model.__class__.__name__], 
                config={'model_name': model.__class__.__name__, 'model_type': model_type,
                       'config_file': config_name, 'seed': seed, 'epochs': epochs,
                       'learning_rate': learning_rate, 'weight_decay': weight_decay,
                       'optimizer': 'adam', 'scheduler': 'plateau',
                       'early_stopping_patience': early_stopping_patience, 'device': str(device)})

    # Setup model and training
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    decorrelators = decor_modules(model)
    has_decorrelation = len(decorrelators) > 0
    early_stopper = EarlyStopper(patience=early_stopping_patience)

    # Model saving setup
    os.makedirs('results/models', exist_ok=True)
    model_save_path = f'results/models/{model.__class__.__name__}_{wandb.run.name}.pth'

    print(f"Training {model.__class__.__name__} on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nModel Summary:")
    summary(model, input_size=(1, 22, 1750))

    best_val_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training loop
        model.train()
        train_loss = train_correct = train_total = decor_loss_total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if has_decorrelation:
                decor_loss_total += np.mean(decor_update(decorrelators))

            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        # Validation and metrics
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)
        train_acc = 100. * train_correct / train_total
        scheduler.step(val_loss)

        # Logging
        log_dict = {'epoch': epoch + 1, 'train_loss': train_loss / len(train_loader),
                    'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch_time': time.time() - epoch_start}

        if has_decorrelation:
            log_dict['decor_loss'] = decor_loss_total / len(train_loader)

        wandb.log(log_dict)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {log_dict["train_loss"]:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%'
              + (f', Decor Loss: {log_dict["decor_loss"]:.6f}' if has_decorrelation else ''))

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(),
                       'best_val_acc': best_val_acc, 'config': dict(wandb.config)}, model_save_path)
            wandb.save(model_save_path)

        # Early stopping
        if early_stopper.early_stop(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Final evaluation
    test_acc, test_loss = evaluate_model(model, test_loader, criterion, device)
    wandb.log({'test_acc': test_acc, 'test_loss': test_loss, 'best_val_acc': best_val_acc})
    wandb.config.update({'actual_epochs': epoch + 1}, allow_val_change=True)
    wandb.finish()

    return {'best_val_acc': best_val_acc, 'test_acc': test_acc,
            'model_path': model_save_path, 'epochs': epoch + 1}
