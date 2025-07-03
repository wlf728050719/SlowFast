import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from model.common import Unit3D, InceptionModule


class I3D(nn.Module):
    """Inception-v1 I3D architecture."""

    def __init__(self, input_channels, number_classes=400, spatial_squeeze=True,
                 endpoint='logits', debug='none'):
        super(I3D, self).__init__()
        self._debug = debug
        if self._debug == 'none':
            self.inception_inner_debug = False
            self.layer_debug = False
        elif self._debug == 'simple':
            self.inception_inner_debug = False
            self.layer_debug = True
        else:
            self.inception_inner_debug = True
            self.layer_debug = True

        self._input_channels = input_channels
        self._num_classes = number_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = endpoint

        self.end_points = nn.ModuleDict(
            {
                'Conv3d_1a_7x7': Unit3D(input_channels, 64, kernel_shape=(7, 7, 7), stride=(2, 2, 2)),
                'MaxPool3d_2a_3x3': nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                'Conv3d_2b_1x1': Unit3D(64, 64, kernel_shape=(1, 1, 1)),
                'Conv3d_2c_3x3': Unit3D(64, 192, kernel_shape=(3, 3, 3)),
                'MaxPool3d_3a_3x3': nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                'Mixed_3b': InceptionModule(192, [64, 96, 128, 16, 32, 32], debug=self.inception_inner_debug),
                'Mixed_3c': InceptionModule(256, [128, 128, 192, 32, 96, 64], debug=self.inception_inner_debug),
                'MaxPool3d_4a_3x3': nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
                'Mixed_4b': InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64],
                                            debug=self.inception_inner_debug),
                'Mixed_4c': InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64],
                                            debug=self.inception_inner_debug),
                'Mixed_4d': InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64],
                                            debug=self.inception_inner_debug),
                'Mixed_4e': InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64],
                                            debug=self.inception_inner_debug),
                'Mixed_4f': InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                            debug=self.inception_inner_debug),
                'MaxPool3d_5a_2x2': nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
                'Mixed_5b': InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                            debug=self.inception_inner_debug),
                'Mixed_5c': InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                            debug=self.inception_inner_debug)
            })

        # Logits
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1), padding=0)
        self.dropout = nn.Dropout3d(p=0.5)
        self.classify_head = Unit3D(384 + 384 + 128 + 128, number_classes, kernel_shape=(1, 1, 1), activation_fn='none',
                                    use_batch_norm=False,
                                    use_bias=True)

    def forward(self, x, dropout_keep_prob=1.0):
        for layer, module in self.end_points.items():
            x = module(x)
            if self.layer_debug:
                print("after:", layer, "x shape:", x.shape)

        x = self.avg_pool(x)
        if self._final_endpoint == 'feature':
            return x
        if dropout_keep_prob < 1.0:
            x = self.dropout(x)
        logits = self.classify_head(x)

        if self._spatial_squeeze:
            logits = logits.squeeze(3).squeeze(3)

        averaged_logits = torch.mean(logits, dim=2)

        if self._final_endpoint == 'logits':
            return averaged_logits

        predictions = F.softmax(averaged_logits, dim=1)
        return predictions


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 64, 224, 224).to(device)
    model = I3D(
        input_channels=3, #输入通道
        number_classes=400,  #分类类别
        endpoint='prediction', #端点 feature/logits/prediction
        debug='simple' #debug模式 none/simple/all
    ).to(device)
    summary(model, (3, 64, 224, 224), device=device.type)
    out = model(x)
    print(out.shape)
