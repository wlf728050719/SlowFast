import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
from PIL import Image


# 自定义数据集类
class FlowDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 包含图片和光流文件的根目录
            transform (callable, optional): 可选的图像变换
        """
        self.root_dir = root_dir
        self.transform = transform

        # 收集所有匹配的文件对
        self.samples = []
        for filename in os.listdir(root_dir):
            if filename.endswith('_img1.ppm'):
                prefix = filename[:-9]  # 去掉 '_img1.ppm' 得到前缀
                img1_path = os.path.join(root_dir, f"{prefix}_img1.ppm")
                img2_path = os.path.join(root_dir, f"{prefix}_img2.ppm")
                flow_path = os.path.join(root_dir, f"{prefix}_flow.flo")

                if os.path.exists(img2_path) and os.path.exists(flow_path):
                    self.samples.append((img1_path, img2_path, flow_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img2_path, flow_path = self.samples[idx]

        # 加载图像
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # 加载光流文件 (.flo 格式)
        flow = self.read_flo_file(flow_path)

        if self.transform:
            # 同时对两帧图像应用相同的随机变换
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img1 = self.transform(img1)
            torch.random.manual_seed(seed)
            img2 = self.transform(img2)

        # 光流转换为Tensor (H,W,2) -> (2,H,W)
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        return torch.cat([img1, img2], dim=0), flow  # 返回 (6,H,W) 和 (2,H,W)

    @staticmethod
    def read_flo_file(filename):
        """
        读取 .flo 光流文件
        """
        with open(filename, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if magic != 202021.25:
                raise RuntimeError(f'Invalid .flo file: {filename}')
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
        return data.reshape(h, w, 2)


def get_dataloader(root_dir, batch_size=8, val_ratio=0.2, img_size=(384, 512)):
    """
    获取训练和验证的DataLoader

    Args:
        root_dir (str): 数据集根目录
        batch_size (int): 批量大小
        val_ratio (float): 验证集比例 (0.0-1.0)
        img_size (tuple): 图像目标尺寸 (H,W)

    Returns:
        train_loader, val_loader: 训练和验证的数据加载器
    """
    # 数据增强和归一化
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建完整数据集
    dataset = FlowDataset(root_dir, transform=transform)

    # 分割训练集和验证集
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # 丢弃最后不完整的batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


# 使用示例
if __name__ == "__main__":
    train_loader, val_loader = get_dataloader(
        root_dir=r"D:\Downloads\FlyingChairs_release\FlyingChairs",
        batch_size=8,
        val_ratio=0.2
    )

    # 检查第一个batch
    for batch in train_loader:
        images, flows = batch
        print("Images min/max:", images.min(), images.max())
        print("Flows min/max:", flows.min(), flows.max())
        print(f"Images shape: {images.shape}")  # 应为 (B,6,H,W)
        print(f"Flows shape: {flows.shape}")  # 应为 (B,2,H,W)
        break