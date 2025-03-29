import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class DoubleConv(nn.Module):
    """U-Net의 기본 구조: (Conv → BN → ReLU) × 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """2D U-Net 모델"""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # 다운샘플링 (Contracting path)
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # 업샘플링 (Expanding path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        # 출력 레이어
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 인코딩 경로
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # 디코딩 경로 (업샘플 + Skip Connection)
        dec4 = self.decoder4(torch.cat([self.upconv4(bottleneck), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([self.upconv3(dec4), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.upconv1(dec2), enc1], dim=1))

        # 최종 출력
        return self.final_conv(dec1)
    
# efficientnet + unet
class effi_UNet(torch.nn.Module):
    def __init__(self, encoder='timm-efficientnet-b7', encoder_weights='advprop',
                 in_channels=1, num_classes=1, activation='sigmoid', device='cuda'):
        super(effi_UNet, self).__init__()

        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
        self.device = device
        self.to(device)
    
    def forward(self, x):
        return self.model(x)

def modeltype(model):
    if model == "unet":
        return UNet(in_channels=1, out_channels=1)
    elif model == "effi_unet":
        return effi_UNet()