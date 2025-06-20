import torch.nn as nn
import torch


class Unit3D(nn.Module):
    """Basic unit containing Conv3D + BatchNorm + non-linearity."""

    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation_fn='relu',
                 padding='same_scaling',  # same_scaling 输出特征图大小为输入特征图/对应维度步长
                 use_batch_norm=True,
                 use_bias=False):
        super(Unit3D, self).__init__()
        self._input_channels = int(input_channels)
        self._output_channels = int(output_channels)
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self._padding = padding

        if self._padding == 'same_scaling':
            self._padding = tuple(max(0, ((k - s + 1) // 2)) for k, s in zip(self._kernel_shape, self._stride))
        elif self._padding == 'none':
            self._padding = (0, 0, 0)

        self.conv3d = nn.Conv3d(
            in_channels=self._input_channels,
            out_channels=self._output_channels,
            kernel_size=kernel_shape,
            stride=self._stride,
            padding=self._padding,
            bias=self._use_bias,
        )

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels)
        else:
            self.bn = None

        if self._activation_fn == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif self._activation_fn == 'none':
            self.activation = None

    def forward(self, inputs):
        net = self.conv3d(inputs)
        if self.bn is not None:
            net = self.bn(net)
        if self.activation is not None:
            net = self.activation(net)
        return net

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1,debug = False):
        super(InceptionModule, self).__init__()
        self._reduction = reduction
        self._out_channels = out_channels
        self._in_channels = int(in_channels)
        self._debug = debug
        # Branch 0
        self.branch0 = Unit3D(self._in_channels, out_channels[0], kernel_shape=(1, 1, 1))

        # Branch 1
        self.branch1_0 = Unit3D(self._in_channels, out_channels[1], kernel_shape=(1, 1, 1))
        self.branch1_1 = Unit3D(out_channels[1], out_channels[2], kernel_shape=(3, 3, 3))

        # Branch 2
        self.branch2_0 = Unit3D(self._in_channels, out_channels[3], kernel_shape=(1, 1, 1))
        self.branch2_1 = Unit3D(out_channels[3], out_channels[4], kernel_shape=(3, 3, 3))

        # Branch 3
        self.branch3_0 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.branch3_1 = Unit3D(self._in_channels, out_channels[5], kernel_shape=(1, 1, 1))

        # Reduction
        total = out_channels[0] + out_channels[2] + out_channels[4] + out_channels[5]
        self.reduction_layer = Unit3D(total, int(total * reduction), kernel_shape=(1, 1, 1))

    def forward(self, x):
        branch0 = self.branch0(x)
        if self._debug:
            print(f"branch0: {branch0.shape}")
        branch1 = self.branch1_0(x)
        branch1 = self.branch1_1(branch1)
        if self._debug:
            print(f"branch1: {branch1.shape}")
        branch2 = self.branch2_0(x)
        branch2 = self.branch2_1(branch2)
        if self._debug:
            print(f"branch2: {branch2.shape}")
        branch3 = self.branch3_0(x)
        branch3 = self.branch3_1(branch3)
        if self._debug:
            print(f"branch3: {branch3.shape}")
        result = torch.cat([branch0, branch1, branch2, branch3], 1)
        if self._debug:
            print(f"branch-concat: {result.shape}")
        if self._reduction < 1:
            result = self.reduction_layer(result)
            if self._debug:
                print(f"reduction: {result.shape}")
        return result

class R3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_shapes, strides=None, paddings=None, debug=False):
        super(R3D, self).__init__()
        if strides is None:
            strides = [(1, 1, 1), (1, 1, 1), (1, 1, 1)]
        if paddings is None:
            paddings = ['same_scaling', 'same_scaling', 'same_scaling']
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_shapes = kernel_shapes
        self._debug = debug

        # 主分支
        self.conv1 = Unit3D(in_channels, out_channels[0], kernel_shape=kernel_shapes[0], stride=strides[0],
                            padding=paddings[0])
        self.conv2 = Unit3D(out_channels[0], out_channels[1], kernel_shape=kernel_shapes[1], stride=strides[1],
                            padding=paddings[1])
        self.conv3 = Unit3D(out_channels[1], out_channels[2], kernel_shape=kernel_shapes[2], stride=strides[2],
                            padding=paddings[2], activation_fn='none')
        # 下采样调整identity
        self.downsample = Unit3D(in_channels, out_channels[2], kernel_shape=kernel_shapes[1], stride=strides[1],
                                 padding=paddings[1], activation_fn='none')

    def forward(self, x):
        identity = x
        if self._debug:
            print('identity:', identity.shape)
        x = self.conv1(x)
        if self._debug:
            print('conv1:', x.shape)
        x = self.conv2(x)
        if self._debug:
            print('conv2:', x.shape)
        x = self.conv3(x)
        if self._debug:
            print('conv3:', x.shape)
        identity = self.downsample(identity)
        if self._debug:
            print('downsample:', identity.shape)
        out = identity + x
        if self._debug:
            print('res-concat:', out.shape)
        out = nn.ReLU(inplace=True)(out)
        return out
