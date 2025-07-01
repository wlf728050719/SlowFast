import torch.nn as nn
import torch
from model.common import Unit3D, R3D
from torchsummary import summary

class SlowFastR3D(nn.Module):
    def __init__(self, input_channels, number_classes, a, b, reshape_method='time_to_channel', endpoint='feature',
                 debug='none'):
        super(SlowFastR3D, self).__init__()
        self._a = a
        self._b = b
        self._reshape_method = reshape_method
        self._endpoint = endpoint
        self._num_classes = number_classes
        self._debug = debug

        if self._debug == 'none':
            self.r3d_inner_debug = False
            self.layer_debug = False
        elif self._debug == 'simple':
            self.r3d_inner_debug = False
            self.layer_debug = True
        else:
            self.r3d_inner_debug = True
            self.layer_debug = True

        reshape_coefficient_rules = {
            'time_to_channel': lambda a, b: 1 + a * b,
            'time_strided_sampling': lambda a, b: 1 + b,
            'time_strided_conv': lambda a, b: 1 + 2 * b
        }
        self.extra_channel_coefficient = reshape_coefficient_rules[self._reshape_method](a, b)
        if self._debug != 'none':
            print('extra_channel_coefficient:' + str(self.extra_channel_coefficient))
        self.slow_end_points = nn.ModuleDict(
            {
                # 3*64*224*224
                'DataLayer': Unit3D(input_channels, input_channels, kernel_shape=(1, 1, 1), stride=(2 * a, 1, 1)),
                # 3*4*224*224
                'Conv1': Unit3D(input_channels, 64, kernel_shape=(1, 7, 7), stride=(1, 2, 2)),
                # 64*4*112*112
                'Pool1': nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                # 64*4*56*56
                'Res2': nn.Sequential(
                    R3D(64 * self.extra_channel_coefficient, [64, 64, 256], [(1, 1, 1), (1, 3, 3), (1, 1, 1)],
                        debug=self.r3d_inner_debug),
                    # 256*4*56*56
                    R3D(256, [64, 64, 256], [(1, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 256*4*56*56
                    R3D(256, [64, 64, 256], [(1, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug)
                ),
                # 256*4*56*56
                'Res3': nn.Sequential(
                    R3D(256 * self.extra_channel_coefficient, [128, 128, 512], [(1, 1, 1), (1, 3, 3), (1, 1, 1)],
                        [(1, 1, 1), (1, 2, 2), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 512*4*28*28
                    R3D(512, [128, 128, 512], [(1, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 512*4*28*28
                    R3D(512, [128, 128, 512], [(1, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 512*4*28*28
                    R3D(512, [128, 128, 512], [(1, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug)
                ),
                # 512*4*28*28
                'Res4': nn.Sequential(
                    R3D(512 * self.extra_channel_coefficient, [256, 256, 1024], [(3, 1, 1), (1, 3, 3), (1, 1, 1)],
                        [(1, 1, 1), (1, 2, 2), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 1024*4*14*14
                    R3D(1024, [256, 256, 1024], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 1024*4*14*14
                    R3D(1024, [256, 256, 1024], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 1024*4*14*14
                    R3D(1024, [256, 256, 1024], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 1024*4*14*14
                    R3D(1024, [256, 256, 1024], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 1024*4*14*14
                    R3D(1024, [256, 256, 1024], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug)
                ),
                # 1024*4*14*14
                'Res5': nn.Sequential(
                    R3D(1024 * self.extra_channel_coefficient, [512, 512, 2048], [(3, 1, 1), (1, 3, 3), (1, 1, 1)],
                        [(1, 1, 1), (1, 2, 2), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 2048*4*7*7
                    R3D(2048, [512, 512, 2048], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 2048*4*7*7
                    R3D(2048, [512, 512, 2048], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug)
                )
                # 2048*4*7*7
            }
        )
        self.fast_end_points = nn.ModuleDict(
            {
                # 3*64*224*224
                'DataLayer': Unit3D(input_channels, input_channels, kernel_shape=(1, 1, 1), stride=(2, 1, 1)),
                # 3*32*224*224
                'Conv1': Unit3D(input_channels, 64 * b, kernel_shape=(5, 7, 7), stride=(1, 2, 2)),
                # 8*32*112*112
                'Pool1': nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                # 8*32*56*56
                'Res2': nn.Sequential(
                    R3D(64 * b, [64 * b, 64 * b, 256 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 256*4*56*56
                    R3D(256 * b, [64 * b, 64 * b, 256 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 256*4*56*56
                    R3D(256 * b, [64 * b, 64 * b, 256 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug)
                ),
                # 32*32*56*56
                'Res3': nn.Sequential(
                    R3D(256 * b, [128 * b, 128 * b, 512 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)],
                        [(1, 1, 1), (1, 2, 2), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 512*4*28*28
                    R3D(512 * b, [128 * b, 128 * b, 512 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 512*4*28*28
                    R3D(512 * b, [128 * b, 128 * b, 512 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 512*4*28*28
                    R3D(512 * b, [128 * b, 128 * b, 512 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)], debug=self.r3d_inner_debug)
                ),
                # 64*32*28*28
                'Res4': nn.Sequential(
                    R3D(512 * b, [256 * b, 256 * b, 1024 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)],
                        [(1, 1, 1), (1, 2, 2), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 1024*4*14*14
                    R3D(1024 * b, [256 * b, 256 * b, 1024 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)],
                        debug=self.r3d_inner_debug),
                    # 1024*4*14*14
                    R3D(1024 * b, [256 * b, 256 * b, 1024 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)],
                        debug=self.r3d_inner_debug),
                    # 1024*4*14*14
                    R3D(1024 * b, [256 * b, 256 * b, 1024 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)],
                        debug=self.r3d_inner_debug),
                    # 1024*4*14*14
                    R3D(1024 * b, [256 * b, 256 * b, 1024 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)],
                        debug=self.r3d_inner_debug),
                    # 1024*4*14*14
                    R3D(1024 * b, [256 * b, 256 * b, 1024 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)],
                        debug=self.r3d_inner_debug)
                ),
                # 128*32*14*14
                'Res5': nn.Sequential(
                    R3D(1024 * b, [512 * b, 512 * b, 2048 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)],
                        [(1, 1, 1), (1, 2, 2), (1, 1, 1)], debug=self.r3d_inner_debug),
                    # 2048*4*7*7
                    R3D(2048 * b, [512 * b, 512 * b, 2048 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)],
                        debug=self.r3d_inner_debug),
                    # 2048*4*7*7
                    R3D(2048 * b, [512 * b, 512 * b, 2048 * b], [(3, 1, 1), (1, 3, 3), (1, 1, 1)],
                        debug=self.r3d_inner_debug)
                )
                # 256*32*7*7
            })
        self.classify_head = nn.LazyLinear(self._num_classes)

    def forward(self, x):
        x_slow = x
        x_fast = x
        for (slow_name, slow_module), (fast_name, fast_module) in zip(
                self.slow_end_points.items(),
                self.fast_end_points.items()
        ):
            # 同步前向传播
            x_slow = slow_module(x_slow)
            x_fast = fast_module(x_fast)
            if self.layer_debug:
                print("after:",slow_name, "x_slow shape:", x_slow.shape, "x_fast shape:", x_fast.shape)
            # 在指定层添加横向连接作为下层输入
            if slow_name in ['Pool1', 'Res2', 'Res3', 'Res4']:
                # 调整Fast分支特征维度
                fast_reshaped = self._lateral_conn(
                    fast_feat=x_fast,
                    slow_feat=x_slow
                )
                # 特征融合（通道拼接）
                x_slow = torch.cat([x_slow, fast_reshaped], dim=1)
                if self.layer_debug:
                    print("after concat x_slow shape:", x_slow.shape, "x_fast shape:", x_fast.shape)


        slow_pool = x_slow.mean(dim=[2, 3, 4])
        fast_pool = x_fast.mean(dim=[2, 3, 4])
        feature = torch.cat([slow_pool, fast_pool], dim=1)
        if self._endpoint == 'feature':
            return feature
        logits = self.classify_head(feature)
        if self._endpoint == 'logits':
            return logits
        prediction = torch.softmax(logits, dim=1)
        return prediction

    def reshape(self, x):
        batch_size, alpha_T, S_squared, beta_C = x.shape

        if self._reshape_method == 'time_to_channel':
            # [batch, αT, S², βC] -> [batch, T, S², αβC]
            # (2, 8, 16, 32) -> (2, 2, 16, 128)
            return x.view(batch_size, -1, self._a, S_squared, beta_C) \
                .permute(0, 1, 3, 2, 4) \
                .contiguous() \
                .view(batch_size, -1, S_squared, self._a * beta_C)

        elif self._reshape_method == 'time_strided_sampling':
            # [batch, αT, S², βC] -> [batch, T, S², βC]
            # (2, 8, 16, 32) -> (2, 2, 16, 32)
            return x[:, ::self._a, :, :]

        elif self._reshape_method == 'time_strided_conv':
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
            raise ValueError(f"不支持的reshape方法: {self._reshape_method}")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SlowFastR3D(
        number_classes=4,
        input_channels=3,  # 3通道RGB
        a=8,  # 时间缩放因子 FastWay时间维度是SlowWay的a倍 建议time Mod (2a)
        b=0.125,  # 通道缩放因子 FastWay通道维度是SlowWay的b倍 建议为16 * b为整数
        reshape_method='time_to_channel',  # 横向连接策略  time_to_channel time_strided_sampling time_strided_conv
        endpoint='prediction',  # 端点  feature/logits/prediction
        debug='simple'  # debug模式 none/simple/all
    ).to(device)

    summary(model, (3, 64, 224, 224), device=device.type)

    # (batch, channel, time, height, width)
    x = torch.randn(1, 3, 64, 224, 224).to(device)
    with torch.no_grad():
        output = model(x)
    print(output.shape)

