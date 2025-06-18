import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize, center_crop
import numpy as np
import cv2


class VideoDataset(Dataset):
    def __init__(self, config_path, target_width=224, target_height=224,
                 num_frames=16, stride=8, mode='single', frame_strategy='zero_pad',
                 transform=None):
        """
        参数:
            config_path: config.json的路径
            target_width: 目标视频宽度
            target_height: 目标视频高度
            num_frames: 每个样本的帧数
            stride: 滑动窗口的步长
            mode: 'single'或'slide'，决定每个视频返回多少样本
            frame_strategy: 'zero_pad'或'discard'，处理不足长度视频的策略
            transform: 可选的额外数据增强
        """
        assert mode in ['single', 'slide'], "mode must be 'single' or 'slide'"
        assert frame_strategy in ['zero_pad', 'discard'], "frame_strategy must be 'zero_pad' or 'discard'"

        with open(config_path, 'r') as f:
            config = json.load(f)

        self.actions = config['actions']
        self.target_width = target_width
        self.target_height = target_height
        self.num_frames = num_frames
        self.stride = stride
        self.mode = mode
        self.frame_strategy = frame_strategy
        self.transform = transform

        # 收集所有视频文件路径和对应的标签
        self.video_paths = []
        self.labels = []

        for action in self.actions:
            video_dir = action['video_folder']
            label = action['label']

            for video_file in os.listdir(video_dir):
                if video_file.endswith(('.mp4', '.avi', '.mov')):  # 支持常见视频格式
                    self.video_paths.append(os.path.join(video_dir, video_file))
                    self.labels.append(label)

        # 对于slide模式，预计算每个视频的样本数
        self.video_samples = []
        if self.mode == 'slide':
            for video_path in self.video_paths:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if total_frames >= self.num_frames:
                    n_samples = (total_frames - self.num_frames) // self.stride + 1
                else:
                    if self.frame_strategy == 'zero_pad':
                        n_samples = 1
                    else:
                        n_samples = 0

                self.video_samples.append(n_samples)

            # 创建索引映射 (dataset_idx -> (video_idx, window_idx))
            self.index_map = []
            for video_idx, n_samples in enumerate(self.video_samples):
                for window_idx in range(n_samples):
                    self.index_map.append((video_idx, window_idx))

    def __len__(self):
        if self.mode == 'single':
            return len(self.video_paths)
        else:
            return len(self.index_map)

    def _process_frame(self, frame):
        """处理单个帧: 调整大小和裁剪"""
        # 转换为CHW格式
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # 调整大小保持宽高比
        h, w = frame.shape[1], frame.shape[2]
        if h / w > self.target_height / self.target_width:
            # 高度过大，先调整高度
            new_h = self.target_height
            new_w = int(w * (self.target_height / h))
        else:
            # 宽度过大，先调整宽度
            new_w = self.target_width
            new_h = int(h * (self.target_width / w))

        frame = resize(frame, (new_h, new_w))

        # 中心裁剪到目标尺寸
        frame = center_crop(frame, (self.target_height, self.target_width))

        return frame

    def _load_video_frames(self, video_path):
        """加载视频并返回所有帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 从BGR转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return np.array(frames)

    def _get_video_segment(self, frames, start_idx):
        """获取视频片段，根据frame_strategy处理不足长度的情况"""
        end_idx = start_idx + self.num_frames
        segment_frames = frames[start_idx:end_idx]

        if len(segment_frames) < self.num_frames:
            if self.frame_strategy == 'zero_pad':
                # 补零
                pad_frames = np.zeros((self.num_frames - len(segment_frames), *segment_frames.shape[1:]),
                                      dtype=segment_frames.dtype)
                segment_frames = np.concatenate([segment_frames, pad_frames])
            else:  # discard
                return None

        return segment_frames

    def __getitem__(self, idx):
        if self.mode == 'single':
            video_path = self.video_paths[idx]
            label = self.labels[idx]

            # 加载视频
            frames = self._load_video_frames(video_path)

            # 只取第一个窗口
            segment_frames = self._get_video_segment(frames, 0)
            if segment_frames is None:  # 只有discard策略会返回None
                # 返回一个空样本，需要在DataLoader中过滤
                return torch.zeros((3, self.num_frames, self.target_height, self.target_width)), -1

        else:  # slide模式
            video_idx, window_idx = self.index_map[idx]
            video_path = self.video_paths[video_idx]
            label = self.labels[video_idx]

            # 加载视频
            frames = self._load_video_frames(video_path)

            # 计算窗口起始位置
            if len(frames) >= self.num_frames:
                start_idx = window_idx * self.stride
            else:
                start_idx = 0  # 只有zero_pad策略会进入这里

            segment_frames = self._get_video_segment(frames, start_idx)
            if segment_frames is None:  # 只有discard策略会返回None
                # 返回一个空样本，需要在DataLoader中过滤
                return torch.zeros((3, self.num_frames, self.target_height, self.target_width)), -1

        # 处理每一帧
        processed_frames = []
        for frame in segment_frames:
            if frame.sum() == 0:  # 补零的帧
                processed_frame = torch.zeros((3, self.target_height, self.target_width))
            else:
                processed_frame = self._process_frame(frame)
                if self.transform:
                    processed_frame = self.transform(processed_frame)
            processed_frames.append(processed_frame)

        # 组合成张量 (C, T, H, W)
        video_tensor = torch.stack(processed_frames, dim=1)

        return video_tensor, label


def get_data_loader(config_path, batch_size=8, target_width=224, target_height=224,
                    num_frames=16, stride=8, mode='single', frame_strategy='zero_pad',
                    train_transform=None, val_transform=None, shuffle=True, num_workers=4,
                    val_ratio=0.2, random_seed=42):
    """
    创建训练集和验证集数据加载器，按80-20比例划分

    参数:
        config_path: 配置文件路径
        batch_size: 批量大小
        target_width: 目标宽度
        target_height: 目标高度
        num_frames: 每个样本的帧数
        stride: 滑动窗口步长
        mode: 'single'或'slide'
        frame_strategy: 'zero_pad'或'discard'
        train_transform: 训练集数据增强
        val_transform: 验证集数据增强
        shuffle: 是否打乱数据
        num_workers: 数据加载工作线程数
        val_ratio: 验证集比例
        random_seed: 随机种子

    返回:
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        class_names: 类别名称列表
    """
    # 创建完整数据集
    full_dataset = VideoDataset(
        config_path=config_path,
        target_width=target_width,
        target_height=target_height,
        num_frames=num_frames,
        stride=stride,
        mode=mode,
        frame_strategy=frame_strategy,
        transform=None  # 不在数据集级别应用变换，分别在训练和验证中应用
    )

    # 获取类别名称
    with open(config_path, 'r') as f:
        config = json.load(f)
    class_names = [action['name'] for action in config['actions']]

    # 计算划分大小
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    # 随机划分数据集
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # 为训练集和验证集分别设置transform
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # 对于discard策略，需要过滤标签为-1的样本
    if frame_strategy == 'discard':
        def collate_fn(batch):
            batch = [item for item in batch if item[1] != -1]
            if len(batch) == 0:
                return torch.zeros((0, 3, num_frames, target_height, target_width)), torch.zeros(0)
            return torch.utils.data.dataloader.default_collate(batch)
    else:
        collate_fn = None

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要shuffle
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, class_names


# 示例使用
if __name__ == "__main__":
    # 定义数据转换
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    val_transform = None  # 验证集通常不需要数据增强

    print("=== 测试get_data_loader函数 ===")
    train_loader, val_loader, class_names = get_data_loader(
        config_path="config.json",
        batch_size=4,
        target_width=224,
        target_height=224,
        num_frames=256,
        stride=8,
        mode='single',
        frame_strategy='zero_pad',
        train_transform=train_transform,
        val_transform=val_transform,
        shuffle=True
    )

    print(f"类别名称: {class_names}")
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"验证集批次数量: {len(val_loader)}")

    # 检查一个训练批次
    for videos, labels in train_loader:
        print(f"\n训练批次 - 视频张量形状: {videos.shape}")  # (batch_size, 3, 16, 224, 224)
        print(f"训练批次 - 标签: {labels}")
        break

    # 检查一个验证批次
    for videos, labels in val_loader:
        print(f"\n验证批次 - 视频张量形状: {videos.shape}")
        print(f"验证批次 - 标签: {labels}")
        break
