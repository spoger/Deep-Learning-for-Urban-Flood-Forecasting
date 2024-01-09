import torch
import torch.nn as nn


class conv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class conv7(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # 添加一个1x1的卷积层以减半通道数
        )

    def forward(self, x):
        return self.upconv(x)


class conv_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_out, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.convp1 = conv7(14, 8)
        self.convp2 = conv7(8, 16)
        self.convp3 = conv7(16, 32)
        self.convp4 = conv7(32, 64)
        self.convp5 = conv7(64, 128)
        self.convp6 = conv7(128, 256)

        self.conv1 = conv3(256, 512)

        self.up1 = up(512, 256)
        self.upconv1 = conv3(512, 256)

        self.up2 = up(256, 128)
        self.upconv2 = conv3(256, 128)

        self.up3 = up(128, 64)
        self.upconv3 = conv3(128, 64)

        self.up4 = up(64, 32)
        self.upconv4 = conv3(64, 32)

        self.up5 = up(32, 16)
        self.upconv5 = conv3(32, 16)

        self.up6 = up(16, 8)
        self.upconv6 = conv3(16, 8)

        self.conv_out = conv_out(8, 1)

    def forward(self, x):
        x1 = self.convp1(x)
        m1 = self.Maxpool(x1)
        # print("m1 shape:", m1.shape)

        x2 = self.convp2(m1)
        m2 = self.Maxpool(x2)
        # print("m2 shape:", m2.shape)

        x3 = self.convp3(m2)
        m3 = self.Maxpool(x3)
        # print("m3 shape:", m3.shape)

        x4 = self.convp4(m3)
        m4 = self.Maxpool(x4)
        # print("m4 shape:", m4.shape)

        x5 = self.convp5(m4)
        m5 = self.Maxpool(x5)
        # print("m5 shape:", m5.shape)

        x6 = self.convp6(m5)
        m6 = self.Maxpool(x6)
        # print("m6 shape:", m6.shape)

        x = self.conv1(m6)
        # print("x shape:", x.shape)

        x = self.up1(x)
        x = torch.cat((x6, x), dim=1)
        x = self.upconv1(x)
        # print("x shape:", x.shape)

        x = self.up2(x)
        x = torch.cat((x5, x), dim=1)
        x = self.upconv2(x)
        # print("x shape:", x.shape)

        x = self.up3(x)
        x = torch.cat((x4, x), dim=1)
        x = self.upconv3(x)
        # print("x shape:", x.shape)

        x = self.up4(x)
        x = torch.cat((x3, x), dim=1)
        x = self.upconv4(x)
        # print("x shape:", x.shape)

        x = self.up5(x)
        x = torch.cat((x2, x), dim=1)
        x = self.upconv5(x)
        # print("x shape:", x.shape)

        x = self.up6(x)
        x = torch.cat((x1, x), dim=1)
        x = self.upconv6(x)
        # print("x shape:", x.shape)

        x = self.conv_out(x)
        return x

# if __name__ == "__main__":
#     model = UNet()
#     print(model)
#
#     input_tensor = torch.randn(1, 14, 1024, 1024)  # Batch size = 1, channels = 4, width = 1024, height = 1024
#     output_tensor = model(input_tensor)
#     print("Output tensor shape:", output_tensor.shape)  # Output tensor shape: [1, 1, 1024, 1024]
