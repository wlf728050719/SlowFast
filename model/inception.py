import torch
import torch.nn as nn


class Unit3D(nn.Module):
    """Basic unit containing Conv3D + BatchNorm + non-linearity."""

    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation_fn=nn.ReLU,
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

        self.conv3d = nn.Conv3d(
            in_channels=self._input_channels,
            out_channels=self._output_channels,
            kernel_size=kernel_shape,
            stride=stride,
            padding=tuple((k - 1) // 2 for k in kernel_shape),
            bias=use_bias
        )

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels)

        if self._activation_fn is not None:
            self.activation = activation_fn(inplace=True)

    def forward(self, inputs):
        net = self.conv3d(inputs)
        if self._use_batch_norm:
            net = self.bn(net)
        if self._activation_fn is not None:
            net = self.activation(net)
        return net


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1):
        super(InceptionModule, self).__init__()

        self._reduction = reduction
        self._out_channels = out_channels
        self._in_channels = int(in_channels)
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

        branch1 = self.branch1_0(x)
        branch1 = self.branch1_1(branch1)

        branch2 = self.branch2_0(x)
        branch2 = self.branch2_1(branch2)

        branch3 = self.branch3_0(x)
        branch3 = self.branch3_1(branch3)

        result = torch.cat([branch0, branch1, branch2, branch3], 1)
        if self._reduction < 1:
            result = self.reduction_layer(result)
        return result

