# plot.py
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_optical_flow(img1, img2, flow_gt, flow_pred, save_dir):
    """
    绘制光流对比图

    Args:
        img1: 第一帧图像 (C,H,W)
        img2: 第二帧图像 (C,H,W)
        flow_gt: 真实光流 (2,H,W)
        flow_pred: 预测光流 (2,H,W)
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 转换为numpy并调整通道顺序
    img1 = img1.permute(1, 2, 0).cpu().numpy()
    img2 = img2.permute(1, 2, 0).cpu().numpy()
    flow_gt = flow_gt.permute(1, 2, 0).cpu().numpy()
    flow_pred = flow_pred.permute(1, 2, 0).cpu().numpy()

    # 绘制图像
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('Frame 1')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2)
    axes[0, 1].set_title('Frame 2')
    axes[0, 1].axis('off')

    # 绘制光流 (使用HSV颜色空间)
    def draw_flow(ax, flow, title):
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis('off')

    draw_flow(axes[1, 0], flow_gt, 'Ground Truth Flow')
    draw_flow(axes[1, 1], flow_pred, 'Predicted Flow')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'optical_flow.png')
    plt.savefig(save_path)
    plt.close()


def plot_loss(train_losses, val_losses, save_dir):
    """
    绘制训练和验证损失曲线

    Args:
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表
        save_dir (str): 保存目录
    """
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path)
    plt.close()


def plot_epe(train_epes, val_epes, save_dir):
    """
    绘制训练和验证EPE曲线

    Args:
        train_epes (list): 训练EPE列表
        val_epes (list): 验证EPE列表
        save_dir (str): 保存目录
    """
    plt.figure()
    plt.plot(train_epes, label='Training EPE')
    plt.plot(val_epes, label='Validation EPE')
    plt.title('Training and Validation EPE')
    plt.xlabel('Epoch')
    plt.ylabel('EPE')
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'epe_curve.png')
    plt.savefig(save_path)
    plt.close()