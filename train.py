import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from data_loader import get_data_loader
from model.slow_fast_i3d import SlowFastI3D
from utils.plot import plot_acc,plot_loss


def train_model(model, train_loader, val_loader, class_names, config, resume_epoch=0):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 确保模型所有部分都移到设备上
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # 训练参数
    num_epochs = config['num_epochs']
    best_val_acc = 0.0
    best_model_path = os.path.join(config['save_dir'], 'best_model.pth')
    last_model_path = os.path.join(config['save_dir'], 'last_model.pth')

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 训练循环
    for epoch in range(resume_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        train_bar = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}', unit='batch')
        for inputs, labels in train_bar:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()

            train_loss += loss.item() * inputs.size(0)
            train_correct += correct
            total_train += inputs.size(0)

            train_bar.set_postfix({
                'loss': train_loss / total_train,
                'acc': 100. * train_correct / total_train
            })

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = 100. * train_correct / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        print(f"Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        val_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}', unit='batch')
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()

                val_loss += loss.item() * inputs.size(0)
                val_correct += correct
                total_val += inputs.size(0)

                val_bar.set_postfix({
                    'loss': val_loss / total_val,
                    'acc': 100. * val_correct / total_val
                })

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100. * val_correct / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        print(f"Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")

        scheduler.step(epoch_val_loss)

        # 保存最佳模型和最后一个模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'acc': epoch_val_acc
            }, best_model_path)
            print(f"New best model saved with val_acc: {best_val_acc:.2f}%")

        # Save last model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }, last_model_path)
        print(f"Last model saved at epoch {epoch + 1}")

        # Plot the curves after each epoch
        plot_loss(train_losses, val_losses, config['save_dir'])
        plot_acc(train_accs, val_accs, config['save_dir'])

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")
    return model


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model and optimizer from checkpoint"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        val_accs = checkpoint.get('val_accs', [])

        print(f"Loaded checkpoint from epoch {epoch}")
        return model, optimizer, epoch, train_losses, val_losses, train_accs, val_accs
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return model, optimizer, 0, [], [], [], []


if __name__ == "__main__":
    # 配置参数
    config = {
        'num_epochs': 300,
        'learning_rate': 0.001,
        'save_dir': 'checkpoints',
        'batch_size': 16,
        'resume_training': True  # Set to False to start fresh training
    }

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    # 获取数据加载器
    train_loader, val_loader, class_names = get_data_loader(
        config_path="config.json",
        batch_size=config['batch_size'],
        target_width=224,
        target_height=224,
        num_frames=64,
        stride=8,
        mode='single',
        frame_strategy='zero_pad',
        train_transform=None,
        val_transform=None,
        shuffle=True
    )

    # 初始化模型并立即转移到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlowFastI3D(3, 4, 0.125, len(class_names), reshape_method='time_to_channel', endpoint='prediction').to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 如果需要从上次训练恢复
    resume_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    if config['resume_training']:
        last_model_path = os.path.join(config['save_dir'], 'last_model.pth')
        model, optimizer, resume_epoch, train_losses, val_losses, train_accs, val_accs = load_checkpoint(
            model, optimizer, last_model_path, device)

        # If we loaded a checkpoint, plot the previous curves
        if resume_epoch > 0:
            plot_loss(train_losses, val_losses, config['save_dir'])
            plot_acc(train_accs, val_accs, config['save_dir'])

    # 打印模型结构（可选）
    print(model)

    # 训练模型
    trained_model = train_model(model, train_loader, val_loader, class_names, config, resume_epoch)