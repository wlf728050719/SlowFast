import torch.nn as nn
import torch

from model.inception import Unit3D, InceptionModule


class SlowFastI3D(nn.Module):
    def __init__(self, input_channels, a, b,number_classes,reshape_method='time_to_channel',endpoint='feature'):
        super().__init__()
        self._a = a  # 时间维度缩放因子α
        self._b = b  # 通道数缩放因子β
        self._input_channels = input_channels
        self.reshape_method = reshape_method
        self._number_classes = number_classes
        self._endpoint = endpoint
        reshape_coefficient_rules = {
            'time_to_channel': lambda a, b: 1 + a * b,
            'time_strided_sampling': lambda a, b: 1 + b,
            'time_strided_conv': lambda a, b: 1 + 2 * b
        }
        self.extra_channel_coefficient = reshape_coefficient_rules[self.reshape_method](a, b)

        self.slow_end_points = nn.ModuleDict(
            {
                'Conv3d_1a_7x7': Unit3D(input_channels, 64, kernel_shape=(7, 7, 7), stride=(2 * a, 2, 2)),
                'MaxPool3d_2a_3x3': nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                'Conv3d_2b_1x1': Unit3D(64 * self.extra_channel_coefficient, 64, kernel_shape=(1, 1, 1)),
                'Conv3d_2c_3x3': Unit3D(64, 192, kernel_shape=(3, 3, 3)),
                'MaxPool3d_3a_3x3': nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                'Mixed_3b': InceptionModule(192 * self.extra_channel_coefficient, [64, 96, 128, 16, 32, 32]),
                'Mixed_3c': InceptionModule(64 + 128 + 32 + 32, [128, 128, 192, 32, 96, 64]),
                'MaxPool3d_4a_3x3': nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
                'Mixed_4b': InceptionModule((128 + 192 + 96 + 64) * self.extra_channel_coefficient, [192, 96, 208, 16, 48, 64]),
                'Mixed_4c': InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64]),
                'Mixed_4d': InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64]),
                'Mixed_4e': InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64]),
                'Mixed_4f': InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128]),
                'MaxPool3d_5a_2x2': nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
                'Mixed_5b': InceptionModule((256 + 320 + 128 + 128) * self.extra_channel_coefficient, [256, 160, 320, 32, 128, 128]),
                'Mixed_5c': InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128])
            })

        self.fast_end_points = nn.ModuleDict(
            {
                'Conv3d_1a_7x7': Unit3D(input_channels, 64 * b, kernel_shape=(7, 7, 7), stride=(2, 2, 2)),
                'MaxPool3d_2a_3x3': nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                'Conv3d_2b_1x1': Unit3D(64 * b, 64 * b, kernel_shape=(1, 1, 1)),
                'Conv3d_2c_3x3': Unit3D(64 * b, 192 * b, kernel_shape=(3, 3, 3)),
                'MaxPool3d_3a_3x3': nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                'Mixed_3b': InceptionModule(192 * b, [64, 96, 128, 16, 32, 32],reduction=b),
                'Mixed_3c': InceptionModule((64 + 128 + 32 + 32)*b, [128, 128, 192, 32, 96, 64],reduction=b),
                'MaxPool3d_4a_3x3': nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
                'Mixed_4b': InceptionModule((128 + 192 + 96 + 64)*b, [192, 96, 208, 16, 48, 64],reduction=b),
                'Mixed_4c': InceptionModule((192 + 208 + 48 + 64)*b, [160, 112, 224, 24, 64, 64],reduction=b),
                'Mixed_4d': InceptionModule((160 + 224 + 64 + 64)*b, [128, 128, 256, 24, 64, 64],reduction=b),
                'Mixed_4e': InceptionModule((128 + 256 + 64 + 64)*b, [112, 144, 288, 32, 64, 64],reduction=b),
                'Mixed_4f': InceptionModule((112 + 288 + 64 + 64)*b, [256, 160, 320, 32, 128, 128],reduction=b),
                'MaxPool3d_5a_2x2': nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
                'Mixed_5b': InceptionModule((256 + 320 + 128 + 128)*b, [256, 160, 320, 32, 128, 128],reduction=b),
                'Mixed_5c': InceptionModule((256 + 320 + 128 + 128)*b, [384, 192, 384, 48, 128, 128],reduction=b)
            })
        self.prediction_layer = nn.LazyLinear(self._number_classes)

    def reshape(self, x):
        batch_size, alpha_T, S_squared, beta_C = x.shape

        if self.reshape_method == 'time_to_channel':
            # [batch, αT, S², βC] -> [batch, T, S², αβC]
            # (2, 8, 16, 32) -> (2, 2, 16, 128)
            return x.view(batch_size, -1, self._a, S_squared, beta_C) \
                .permute(0, 1, 3, 2, 4) \
                .contiguous() \
                .view(batch_size, -1, S_squared, self._a * beta_C)

        elif self.reshape_method == 'time_strided_sampling':
            # [batch, αT, S², βC] -> [batch, T, S², βC]
            # (2, 8, 16, 32) -> (2, 2, 16, 32)
            return x[:, ::self._a, :, :]

        elif self.reshape_method == 'time_strided_conv':
            # [batch, αT, S², βC] -> [batch, T, S², 2βC]
            # (2, 8, 16, 32) -> (2, 2, 16, 64)
            conv3d = nn.Conv3d(
                in_channels=beta_C,
                out_channels=2 * beta_C,
                kernel_size=(5, 1, 1),
                stride=(self._a, 1, 1),
                padding=(2, 0, 0),
                device=x.device
            )
            return conv3d(x.permute(0, 3, 1, 2).unsqueeze(-1)) \
                .squeeze(-1) \
                .permute(0, 2, 3, 1)

        else:
            raise ValueError(f"不支持的reshape方法: {self.reshape_method}")

    def forward(self, x):
        # 初始输入
        x_slow = x
        x_fast = x

        # 同步处理两个分支
        for (slow_name, slow_module), (fast_name, fast_module) in zip(
                self.slow_end_points.items(),
                self.fast_end_points.items()
        ):
            # 同步前向传播
            x_slow = slow_module(x_slow)
            x_fast = fast_module(x_fast)

            # 在指定层添加横向连接作为下层输入
            if slow_name in ['MaxPool3d_2a_3x3', 'MaxPool3d_3a_3x3', 'MaxPool3d_4a_3x3','MaxPool3d_5a_2x2']:
                # 调整Fast分支特征维度
                fast_reshaped = self._lateral_conn(
                    fast_feat=x_fast,
                    slow_feat=x_slow
                )
                # 特征融合（通道拼接）
                x_slow = torch.cat([x_slow, fast_reshaped], dim=1)
            # print("x_slow shape:", x_slow.shape,"x_fast shape:", x_fast.shape)

        # 最终特征处理
        slow_pool = x_slow.mean(dim=[2, 3, 4])
        fast_pool = x_fast.mean(dim=[2, 3, 4])

        feature = torch.cat([slow_pool, fast_pool], dim=1)
        if self._endpoint == 'feature':
            return feature
        else:
            logits = self.prediction_layer(feature)
            prediction = torch.softmax(logits, dim=1)
            return prediction

    def _lateral_conn(self, fast_feat, slow_feat):
        """横向连接维度转换"""
        B, C_fast, T_fast, H, W = fast_feat.shape
        _, C_slow, T_slow, _, _ = slow_feat.shape

        # 转换为[T, H*W, C]格式以适应reshape
        fast_4d = fast_feat.permute(0, 2, 3, 4, 1).reshape(B, T_fast, H * W, C_fast)

        # 应用预设的reshape方法
        reshaped = self.reshape(fast_4d)  # [B, T_slow, H*W, new_C]

        # 转回3D格式 [B, new_C, T_slow, H, W]
        return reshaped.reshape(B, T_slow, H, W, -1).permute(0, 4, 1, 2, 3)

if __name__ == '__main__':
    # 初始化模型

    model = SlowFastI3D(
        number_classes= 4,
        input_channels=3, #3通道RGB
        a=4,    #时间缩放因子 FastWay时间维度是SlowWay的a倍 建议time Mod (2a)
        b=0.125, #通道缩放因子 FastWay通道维度是SlowWay的b倍 建议为16 * b为整数
        reshape_method='time_to_channel', #横向连接策略  time_to_channel time_strided_sampling time_strided_conv
        endpoint='prediction'
    )
    #(batch, channel, time, height, width) time过小，a过大时会导致特征图过小无法卷积
    x = torch.randn(2, 3, 32, 224, 224)
    with torch.no_grad():
        output = model(x)
    print(output.shape)
