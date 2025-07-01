import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_data_loader
from model.timesformer import TimeSformer
from model.slow_fast_i3d import SlowFastI3D
from model.slow_fast_r3d import SlowFastR3D
from utils.metrics import compute_ap, compute_map
from utils.plot import plot_loss, plot_acc, plot_map, plot_class_ap


class Trainer:
    def __init__(self, config, model, train_loader, val_loader, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.config['train']['save_dir'] = os.path.normpath(self.config['train']['save_dir'])
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names

        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=1e-5
        )

        # Training state
        self.best_map = 0.0
        self.current_epoch = 0  # 初始化为0
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_map': [],
            'best_class_ap': None
        }

        # Create save directory
        os.makedirs(config['train']['save_dir'], exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        with tqdm(self.train_loader, desc=f'Train Epoch {self.current_epoch + 1}', leave=True) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total_loss += loss.item() * inputs.size(0)
                correct += (predicted == labels).sum().item()
                total += inputs.size(0)

                pbar.set_postfix({
                    'loss': total_loss / total,
                    'acc': 100. * correct / total,
                    'lr': self.optimizer.param_groups[0]['lr']
                })

        return total_loss / len(self.train_loader.dataset), 100. * correct / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad(), tqdm(self.val_loader, desc=f'Valid Epoch {self.current_epoch + 1}', leave=True) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_loss += loss.item() * inputs.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.append(outputs.softmax(dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                pbar.set_postfix({
                    'loss': total_loss / (pbar.n + 1),
                    'acc': 100. * correct / ((pbar.n + 1) * self.val_loader.batch_size)
                })

        # Calculate metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        class_ap = compute_ap(all_labels, all_preds, len(self.class_names))
        epoch_map = compute_map(all_labels, all_preds, len(self.class_names))

        return (
            total_loss / len(self.val_loader.dataset),
            100. * correct / len(self.val_loader.dataset),
            epoch_map,
            class_ap
        )

    def save_checkpoint(self, is_best=False):
        state = {
            'epoch': self.current_epoch,  # 保存当前epoch（从0开始）
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'class_names': self.class_names
        }

        filename = 'best_model.pth' if is_best else 'last_model.pth'
        torch.save(state, os.path.join(self.config['train']['save_dir'], filename))

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.metrics = checkpoint.get('metrics', {
                'train_loss': [], 'val_loss': [],
                'train_acc': [], 'val_acc': [],
                'val_map': [], 'best_class_ap': None
            })
            return True
        return False

    def plot_metrics(self):
        """Plot all metrics separately"""
        save_dir = self.config['train']['save_dir']
        plot_loss(self.metrics['train_loss'], self.metrics['val_loss'], save_dir)
        plot_acc(self.metrics['train_acc'], self.metrics['val_acc'], save_dir)
        plot_map(self.metrics['val_map'], save_dir)
        if self.metrics['best_class_ap'] is not None:
            plot_class_ap(self.metrics['best_class_ap'], self.class_names, save_dir)

    def train(self, resume=False):
        # Load checkpoint if resuming
        if resume:
            last_model_path = os.path.join(self.config['train']['save_dir'], 'last_model.pth')
            if self.load_checkpoint(last_model_path):
                print(f"Resuming training from epoch {self.current_epoch + 1}")  # 显示时+1
                self.plot_metrics()  # Plot previous metrics

        # Main training loop
        num_epochs = self.config['train']['epochs']
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch  # 直接使用当前epoch值

            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_map, class_ap = self.validate()

            # Update metrics
            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_acc'].append(val_acc)
            self.metrics['val_map'].append(val_map)

            # Check for best model
            if val_map > self.best_map:
                self.best_map = val_map
                self.metrics['best_class_ap'] = class_ap
                self.save_checkpoint(is_best=True)

            # Save last checkpoint
            self.save_checkpoint()

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%, Val mAP: {val_map:.4f}, "
                  f"Best mAP: {self.best_map:.4f}")

            # Plot metrics every 5 epochs
            if (epoch + 1) % 1 == 0:
                self.plot_metrics()

        # Final plots
        self.plot_metrics()
        print(f"\nTraining complete. Best Val mAP: {self.best_map:.4f}")


if __name__ == "__main__":
    config_path = "TimeSformer.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Prepare data loaders
    train_loader, val_loader, class_names = get_data_loader(
        dataset_json=config['dataloader']['dataset_json'],
        batch_size=config['train']['batch_size'],
        width=config['dataloader']['width'],
        height=config['dataloader']['height'],
        num_frames=config['dataloader']['frames'],
        mode=config['dataloader']['mode'],
        stride=config['dataloader']['stride'],
        frame_strategy=config['dataloader']['frame_strategy'],
        train_transform=None,
        val_transform=None
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Initialize model
    if config['model']['name'] == 'SlowFastR3D':
        model = SlowFastR3D(
            input_channels=3,
            a=config['model']['a'],
            b=config['model']['b'],
            number_classes=len(class_names),
            reshape_method=config['model']['reshape_method'],
            endpoint='logits'
        )
    elif config['model']['name'] == 'SlowFastI3D':
        model = SlowFastI3D(
            input_channels=3,
            a=config['model']['a'],
            b=config['model']['b'],
            number_classes=len(class_names),
            reshape_method=config['model']['reshape_method'],
            endpoint='logits'
        )
    elif config['model']['name'] == 'TimeSformer':
        assert config['dataloader']['width'] == config['dataloader']['height'],f"dataloader width should be equal to dataloader height"
        model = TimeSformer(
            dim=config['model']['dim'],
            frames=config['dataloader']['frames'],
            number_classes=len(class_names),
            image_size=config['dataloader']['width'],
            patch_size=config['model']['patch_size'],
            input_channels=3,
            depth=config['model']['depth'],
            heads=config['model']['heads'],
            attn_dropout=config['model']['attn_dropout'],
            ff_dropout=config['model']['ff_dropout'],
            rotary_emb=config['model']['rotary_emb'],
            shift_tokens=config['model']['shift_tokens'],
            endpoint='logits'
        )
    print("Model:"+config['model']['name'])
    # Create and run trainer
    trainer = Trainer(config, model, train_loader, val_loader, class_names)
    trainer.train(resume=config['train']['resume_training'])