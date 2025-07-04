import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize, center_crop
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable


class FrameSamplingStrategy(ABC):
    """Abstract base class for frame sampling strategies"""

    def __init__(self, num_frames: int, frame_strategy: str = 'zero_pad'):
        self.num_frames = num_frames
        self.frame_strategy = frame_strategy
        assert frame_strategy in ['zero_pad', 'discard'], "Invalid frame strategy"

    @abstractmethod
    def sample_frames(self, frames: np.ndarray) -> Optional[np.ndarray]:
        """Sample frames from video according to the strategy"""
        pass

    def handle_short_video(self, frames: np.ndarray) -> Optional[np.ndarray]:
        """Handle videos shorter than required number of frames"""
        print("frames length: ", len(frames), "expected length: ", self.num_frames, "use strategy: ",
              self.frame_strategy)
        if self.frame_strategy == 'zero_pad':
            pad_frames = np.zeros((self.num_frames - len(frames), *frames.shape[1:]),
                                  dtype=frames.dtype)
            return np.concatenate([frames, pad_frames])
        elif self.frame_strategy == 'discard':
            return None
        else:
            raise ValueError("Invalid frame strategy")


class HeadSegmentSampler(FrameSamplingStrategy):
    """Sample first N frames from video"""

    def sample_frames(self, frames: np.ndarray) -> Optional[np.ndarray]:
        segment = frames[:self.num_frames]
        return segment if len(segment) >= self.num_frames else self.handle_short_video(segment)


class SlideWindowSampler(FrameSamplingStrategy):
    """Slide window sampling with specified stride"""

    def __init__(self, window_size: int, stride: int, frame_strategy: str = 'zero_pad'):
        super().__init__(window_size, frame_strategy)
        self.stride = stride

    def sample_frames(self, frames: np.ndarray, start_idx: int = 0) -> Optional[np.ndarray]:
        end_idx = start_idx + self.num_frames
        segment = frames[start_idx:end_idx]
        return segment if len(segment) >= self.num_frames else self.handle_short_video(segment)


class UniformSampler(FrameSamplingStrategy):
    """Uniformly sample frames across the video"""

    def sample_frames(self, frames: np.ndarray) -> Optional[np.ndarray]:
        total_frames = len(frames)

        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, num=self.num_frames, dtype=np.int32)
            return frames[indices]
        return self.handle_short_video(frames)


class GroupSampler(FrameSamplingStrategy):
    """Sample n groups of x frames each, uniformly distributed across the video"""

    def __init__(self, num_groups: int, frames_per_group: int, frame_strategy: str = 'zero_pad'):
        super().__init__(num_groups * frames_per_group, frame_strategy)
        self.num_groups = num_groups
        self.frames_per_group = frames_per_group

    def sample_frames(self, frames: np.ndarray) -> Optional[np.ndarray]:
        total_frames = len(frames)

        if total_frames >= self.num_frames:
            group_boundaries = np.linspace(0, total_frames, self.num_groups + 1, dtype=np.int32)
            sampled_frames = []
            for i in range(self.num_groups):
                start, end = group_boundaries[i], group_boundaries[i + 1]
                if self.frames_per_group == 1:
                    indices = [start]
                else:
                    indices = np.linspace(start, end - 1, self.frames_per_group, dtype=np.int32)
                sampled_frames.extend(frames[indices])
            return np.array(sampled_frames)
        return self.handle_short_video(frames)


class VideoProcessor:
    """Handles video frame processing and transformations"""

    def __init__(self, target_width: int, target_height: int,
                 transform: Optional[Callable] = None):
        self.target_width = target_width
        self.target_height = target_height
        self.transform = transform

    def process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Process a single frame"""
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # Resize maintaining aspect ratio
        h, w = frame_tensor.shape[1], frame_tensor.shape[2]
        if h / w > self.target_height / self.target_width:
            new_h = self.target_height
            new_w = int(w * (self.target_height / h))
        else:
            new_w = self.target_width
            new_h = int(h * (self.target_width / w))

        frame_tensor = resize(frame_tensor, [new_h, new_w])
        frame_tensor = center_crop(frame_tensor, [self.target_height, self.target_width])

        if self.transform:
            frame_tensor = self.transform(frame_tensor)

        return frame_tensor

    def process_segment(self, segment: np.ndarray) -> torch.Tensor:
        """Process a segment of frames"""
        processed_frames = []
        for frame in segment:
            if frame.sum() == 0:  # Zero-padded frame
                processed_frame = torch.zeros((3, self.target_height, self.target_width))
            else:
                processed_frame = self.process_frame(frame)
            processed_frames.append(processed_frame)

        return torch.stack(processed_frames, dim=1)  # (C, T, H, W)


class VideoDataset(Dataset):
    def __init__(self, sampler_args, dataset_json: str, width: int = 224, height: int = 224,
                 transform: Optional[Callable] = None):
        """
        Args:
            dataset_json: Path to dataset.json
            width: Target video width
            height: Target video height
            transform: Optional data augmentation
        """
        self.target_width = width
        self.target_height = height
        self.sampler_args = sampler_args
        self.transform = transform

        # Load dataset configuration
        with open(dataset_json, 'r') as f:
            dataset = json.load(f)

        self.actions = dataset['actions']
        self._setup_sampler()
        self._load_video_metadata()
        self.video_processor = VideoProcessor(width, height, transform)

    def _setup_sampler(self):
        """Initialize the appropriate frame sampler"""
        self.mode = self.sampler_args['mode']
        if self.mode == 'head':
            self.sampler = HeadSegmentSampler(self.sampler_args['num_frames'], self.sampler_args['frame_strategy'])
        elif self.mode == 'slide':
            self.sampler = SlideWindowSampler(self.sampler_args['window_size'], self.sampler_args['stride'],
                                              self.sampler_args['frame_strategy'])
        elif self.mode == 'uniform':
            self.sampler = UniformSampler(self.sampler_args['num_frames'], self.sampler_args['frame_strategy'])
        elif self.mode == 'group':
            self.sampler = GroupSampler(self.sampler_args['num_groups'], self.sampler_args['frames_per_group'],
                                        self.sampler_args['frame_strategy'])
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        self.num_frames = self.sampler.num_frames

    def _load_video_metadata(self):
        """Load video paths and labels, and precompute sample counts"""
        self.video_paths = []
        self.labels = []
        self.video_samples = []  # Number of samples per video

        for action in self.actions:
            video_dir = action['video_folder']
            label = action['label']

            for video_file in os.listdir(video_dir):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(video_dir, video_file)
                    self.video_paths.append(video_path)
                    self.labels.append(label)

                    # For sliding window, precompute number of samples
                    if self.mode == 'slide':
                        cap = cv2.VideoCapture(video_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()

                        if total_frames >= self.num_frames:
                            n_samples = (total_frames - self.num_frames) // self.sampler.stride + 1
                        else:
                            n_samples = 0 if self.sampler.frame_strategy == 'discard' else 1
                        self.video_samples.append(n_samples)
                    else:
                        self.video_samples.append(1)  # 1 sample per video for other modes

        # Create index mapping for sliding window mode
        if self.mode == 'slide':
            self.index_map = []
            for video_idx, n_samples in enumerate(self.video_samples):
                for window_idx in range(n_samples):
                    self.index_map.append((video_idx, window_idx))

    def __len__(self) -> int:
        if self.mode == 'slide':
            return len(self.index_map)
        return len(self.video_paths)

    @staticmethod
    def load_video_frames(video_path: str) -> np.ndarray:
        """Load all frames from a video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return np.array(frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.mode == 'slide':
            video_idx, window_idx = self.index_map[idx]
            start_idx = window_idx * self.sampler.stride if self.video_samples[video_idx] > 1 else 0
        else:
            video_idx = idx

        video_path = self.video_paths[video_idx]
        label = self.labels[video_idx]

        # Load video frames
        frames = self.load_video_frames(video_path)

        # Sample frames according to strategy
        if self.mode == 'slide':
            segment = self.sampler.sample_frames(frames, start_idx)
        else:
            segment = self.sampler.sample_frames(frames)

        if segment is None:  # Only happens with discard strategy
            return torch.zeros((3, self.num_frames, self.target_height, self.target_width)), -1

        # Process the segment
        video_tensor = self.video_processor.process_segment(segment)
        return video_tensor, label


def get_data_loader(sampler_args, dataset_json: str, batch_size: int = 8, width: int = 224, height: int = 224,
                    train_transform: Optional[Callable] = None,
                    val_transform: Optional[Callable] = None,
                    num_workers: int = 4, val_ratio: float = 0.2,
                    random_seed: int = 42) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create train and validation data loaders with 80-20 split

    Args:
        dataset_json: Path to dataset config file
        batch_size: Batch size
        width: Target width
        height: Target height
        train_transform: Training data augmentation
        val_transform: Validation data augmentation
        num_workers: Number of worker processes
        val_ratio: Validation set ratio
        random_seed: Random seed for reproducibility

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        class_names: List of class names
    """
    print(sampler_args)
    # Load class names from config
    with open(dataset_json, 'r') as f:
        config = json.load(f)
    class_names = [action['name'] for action in config['actions']]

    # Create full dataset
    full_dataset = VideoDataset(
        sampler_args=sampler_args,
        dataset_json=dataset_json,
        width=width,
        height=height,
        transform=None  # Will be set separately for train/val
    )

    # Split dataset
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # Set transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Collate function to filter out invalid samples (for discard strategy)
    def collate_fn(batch):
        batch = [item for item in batch if item[1] != -1]
        if len(batch) == 0:
            return torch.zeros((0, 3, full_dataset.num_frames, height, width)), torch.zeros(0)
        return torch.utils.data.dataloader.default_collate(batch)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn if full_dataset.sampler.frame_strategy == 'discard' else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn if full_dataset.sampler.frame_strategy == 'discard' else None
    )

    return train_loader, val_loader, class_names


# Example usage
if __name__ == "__main__":
    # Define data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    # {'mode': 'head', 'num_frames': 16, 'frame_strategy': 'zero_pad'}
    # {'mode': 'slide', 'window_size': 16, 'stride': 8, 'frame_strategy': 'discard'}
    # {'mode': 'uniform', 'num_frames': 16, 'frame_strategy': 'zero_pad'}
    # {'mode': 'group', 'num_groups': 4, 'frames_per_group': 4, 'frame_strategy': 'zero_pad'}

    print("=== Testing get_data_loader ===")
    train_loader, val_loader, class_names = get_data_loader(
        sampler_args={'mode': 'slide', 'window_size': 64, 'stride': 64, 'frame_strategy': 'discard'},
        dataset_json="dataset.json",
        batch_size=4,
        width=224,
        height=224,
        train_transform=train_transform,
        val_transform=None,
    )

    print(f"Class names: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Check a training batch
    for videos, labels in train_loader:
        print(f"\nTrain batch - video shape: {videos.shape}")  # (batch_size, 3, 16, 224, 224)
        print(f"Train batch - labels: {labels}")
        break

    # Check a validation batch
    for videos, labels in val_loader:
        print(f"\nVal batch - video shape: {videos.shape}")
        print(f"Val batch - labels: {labels}")
        break
