# train.py
import torch
import torch.optim as optim
import os
import json
from tqdm import tqdm
import random

from utils.flownet.FlowNetSD import FlowNetSD
from utils.flownet.data_loader import get_dataloader
from utils.flownet.losses import MultiScale
from utils.flownet.plot import plot_loss, plot_epe, plot_optical_flow


def load_config(config_path):
    """加载JSON配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train_model(model, train_loader, val_loader, criterion, optimizer, config, device='cuda'):
    """
    训练FlowNetSD模型

    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        config: 配置字典
        device: 训练设备 ('cuda' or 'cpu')
    """
    train_cfg = config['train']
    save_dir = train_cfg['save_dir']

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 初始化变量
    best_val_loss = float('inf')
    start_epoch = 0
    train_loss_history = []
    val_loss_history = []
    train_epe_history = []
    val_epe_history = []

    # 存储一个验证样本用于可视化
    vis_sample = None

    # 继续训练逻辑
    if train_cfg['resume_training']:
        last_checkpoint = os.path.join(save_dir, 'flownet_last.pth')
        if os.path.exists(last_checkpoint):
            checkpoint = torch.load(last_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_loss_history = checkpoint['train_loss']
            val_loss_history = checkpoint['val_loss']
            train_epe_history = checkpoint['train_epe']
            val_epe_history = checkpoint['val_epe']
            best_val_loss = min(val_loss_history) if val_loss_history else float('inf')
            print(f"Resuming training from epoch {start_epoch}")

    # 获取一个验证样本用于可视化
    if vis_sample is None:
        val_iter = iter(val_loader)
        vis_sample = next(val_iter)
        while vis_sample[0].shape[0] < 1:  # 确保batch不为空
            vis_sample = next(val_iter)
        vis_idx = random.randint(0, vis_sample[0].shape[0] - 1)
        vis_img1 = vis_sample[0][vis_idx, :3]  # 取前3通道作为第一帧
        vis_img2 = vis_sample[0][vis_idx, 3:]  # 取后3通道作为第二帧
        vis_flow = vis_sample[1][vis_idx]  # 光流真值

    # 训练循环
    for epoch in range(start_epoch, train_cfg['epochs']):
        print(f'\nEpoch {epoch + 1}/{train_cfg["epochs"]}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_epe = 0.0

        train_pbar = tqdm(train_loader, desc='Training', leave=False)
        for i, (inputs, targets) in enumerate(train_pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # 计算损失
            if isinstance(outputs, tuple):  # 多尺度输出
                loss = criterion(outputs, targets)
            else:  # 单一输出
                loss = criterion(outputs, targets)

            loss[0].backward()
            optimizer.step()

            running_loss += loss[0].item()
            running_epe += loss[1].item()

            train_pbar.set_postfix({
                'loss': running_loss / (i + 1),
                'epe': running_epe / (i + 1)
            })

        # 计算平均训练损失
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_epe = running_epe / len(train_loader)
        train_loss_history.append(epoch_train_loss)
        train_epe_history.append(epoch_train_epe)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_epe = 0.0

        val_pbar = tqdm(val_loader, desc='Validation', leave=False)
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                if isinstance(outputs, tuple):
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, targets)

                val_loss += loss[0].item()
                val_epe += loss[1].item()

                val_pbar.set_postfix({
                    'val_loss': val_loss / (i + 1),
                    'val_epe': val_epe / (i + 1)
                })

        # 计算平均验证损失
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_epe = val_epe / len(val_loader)
        val_loss_history.append(epoch_val_loss)
        val_epe_history.append(epoch_val_epe)

        # 打印epoch结果
        print(f'Train Loss: {epoch_train_loss:.4f} | EPE: {epoch_train_epe:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} | EPE: {epoch_val_epe:.4f}')

        # 保存当前模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss_history,
            'val_loss': val_loss_history,
            'train_epe': train_epe_history,
            'val_epe': val_epe_history,
            'config': config
        }

        torch.save(checkpoint, os.path.join(save_dir, 'last_model.pth'))

        # 如果是最好的模型，则保存并可视化
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f'>>> Best model saved with val loss: {best_val_loss:.4f}')

            # 可视化当前最佳模型在验证样本上的表现
            with torch.no_grad():
                vis_input = torch.cat([vis_img1, vis_img2], dim=0).unsqueeze(0).to(device)
                vis_output = model(vis_input)
                if isinstance(vis_output, tuple):
                    vis_output = vis_output[0]  # 取最精细尺度的输出

                # 保存可视化结果
                plot_optical_flow(
                    img1=vis_img1,
                    img2=vis_img2,
                    flow_gt=vis_flow,
                    flow_pred=vis_output.squeeze(0),
                    save_dir=save_dir
                )

        # 绘制损失和EPE曲线
        plot_loss(train_loss_history, val_loss_history, save_dir)
        plot_epe(train_epe_history, val_epe_history, save_dir)

    print('\nTraining complete')


if __name__ == '__main__':
    # 加载配置
    config = load_config(r"checkpoints/FlowNetSD/FlyingChairs/config.json")
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # 初始化模型
    model_cfg = config['model']
    if model_cfg['name'] == 'FlowNetSD':
        model = FlowNetSD(input_channels=6, batchNorm=True, training=True).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_cfg['name']}")
    print(f"Model {model_cfg['name']} initialized")

    # 数据加载
    train_loader, val_loader = get_dataloader(
        root_dir=config['dataloader']['root_dir'],
        batch_size=config['train']['batch_size'],
        val_ratio=config['dataloader']['val_ratio']
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 损失函数和优化器
    criterion = MultiScale(startScale=4, numScales=5, l_weight=0.32, norm='L1')
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    # 训练模型
    print("\nStarting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device
    )