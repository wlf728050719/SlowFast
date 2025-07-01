import torch
import json
import cv2
from torchvision.transforms.functional import resize, center_crop
from model.slow_fast_i3d import SlowFastI3D
from tqdm import tqdm


class VideoPredictor:
    def __init__(self, config_path, checkpoint_path, num_frames=64, target_size=224):
        """
        初始化视频预测器

        参数:
            config_path: 配置文件路径（包含类别信息）
            checkpoint_path: 模型权重路径
            num_frames: 输入模型的帧数
            target_size: 输入模型的帧尺寸（正方形）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.target_size = target_size

        # 加载类别信息
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.class_names = [action['name'] for action in config['actions']]

        # 初始化模型
        self.model = SlowFastI3D(
            input_channels=3,
            a=4,
            b=0.125,
            number_classes=len(self.class_names),
            reshape_method='time_to_channel',
            endpoint='prediction'
        ).to(self.device)

        # 加载模型权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded model from {checkpoint_path}")

    def preprocess_frame(self, frame):
        """预处理单个帧"""
        # 转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # 调整大小保持宽高比
        h, w = frame.shape[1], frame.shape[2]
        if h / w > 1:
            new_h = self.target_size
            new_w = int(w * (self.target_size / h))
        else:
            new_w = self.target_size
            new_h = int(h * (self.target_size / w))

        frame = resize(frame, (new_h, new_w))
        frame = center_crop(frame, (self.target_size, self.target_size))
        return frame

    def get_video_frames(self, video_path):
        """从视频中提取帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 计算采样间隔
        frame_interval = max(1, int(total_frames / self.num_frames))

        frames = []
        frame_count = 0
        with tqdm(total=self.num_frames, desc="Extracting frames") as pbar:
            while len(frames) < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    processed_frame = self.preprocess_frame(frame)
                    frames.append(processed_frame)
                    pbar.update(1)

                frame_count += 1

        cap.release()

        # 如果帧数不足，用黑帧补齐
        if len(frames) < self.num_frames:
            pad_frames = [torch.zeros((3, self.target_size, self.target_size))
                          for _ in range(self.num_frames - len(frames))]
            frames.extend(pad_frames)

        # 组合成张量 (C, T, H, W)
        video_tensor = torch.stack(frames, dim=1)
        return video_tensor.unsqueeze(0)  # 添加batch维度

    def predict(self, video_path, top_k=5):
        """
        对视频进行行为预测

        参数:
            video_path: 输入视频路径
            top_k: 返回前k个预测结果

        返回:
            list: 包含(top_class_name, probability)元组的列表
        """
        # 预处理视频
        video_tensor = self.get_video_frames(video_path).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        # 转换为可读结果
        results = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            class_name = self.class_names[class_idx]
            prob = top_probs[0][i].item()
            results.append((class_name, prob))

        return results


if __name__ == "__main__":
    # 配置参数
    CONFIG_PATH = "SlowFastR3D.json"  # 与训练时相同的配置文件
    CHECKPOINT_PATH = "checkpoints/best_model.pth"  # 训练好的模型权重
    VIDEO_PATH = r"D:\Desktop\UCF-101\BlowDryHair\v_ApplyLipstick_g15_c02.avi"  # 待预测视频

    # 初始化预测器
    predictor = VideoPredictor(
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        num_frames=64,
        target_size=224
    )

    # 执行预测
    predictions = predictor.predict(VIDEO_PATH, top_k=5)

    # 打印结果
    print("\nTop-5 Predictions:")
    for i, (class_name, prob) in enumerate(predictions, 1):
        print(f"{i}. {class_name}: {prob * 100:.2f}%")