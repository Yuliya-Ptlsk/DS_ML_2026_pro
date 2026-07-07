import torch
import torch.nn as nn

class UNetModel(nn.Module):
    class _ConvLayerDbl(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )

        def forward(self, x):
            return self.model(x)

    class _EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = UNetModel._ConvLayerDbl(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.block(x)
            y = self.max_pool(x)

            return y, x

    class _DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.block = UNetModel._ConvLayerDbl(in_channels, out_channels)

        def forward(self, x, y):
            x = self.transpose(x)
            u = torch.cat([x, y], dim=1)
            u = self.block(u)

            return u

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc_1 = self._EncoderBlock(in_channels, 32)
        self.enc_2 = self._EncoderBlock(32, 64)
        self.enc_3 = self._EncoderBlock(64, 128)
        self.enc_4 = self._EncoderBlock(128, 256)

        self.bottleneck = self._ConvLayerDbl(256, 512)

        self.dec_1 = self._DecoderBlock(512, 256)
        self.dec_2 = self._DecoderBlock(256, 128)
        self.dec_3 = self._DecoderBlock(128, 64)
        self.dec_4 = self._DecoderBlock(64, 32)

        self.out = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x, y_1 = self.enc_1(x)
        x, y_2 = self.enc_2(x)
        x, y_3 = self.enc_3(x)
        x, y_4 = self.enc_4(x)

        x = self.bottleneck(x)

        x = self.dec_1(x, y_4)
        x = self.dec_2(x, y_3)
        x = self.dec_3(x, y_2)
        x = self.dec_4(x, y_1)

        return self.out(x)

