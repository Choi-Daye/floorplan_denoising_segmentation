import torch
import torch.nn as nn


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)     # 3x3 conv 변환
        self.bn1 = nn.BatchNorm2d(out_channels)     # batch norm
        self.relu = nn.ReLU()       # activation func
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # for dimension matching (잔차를 만들기 위해, 입력을 바로 출력으로 만드는 '지름길' 생성)
        if in_channels != out_channels:     # 입출력 채널 개수가 다르면
            self.skip = nn.Conv2d(in_channels, out_channels, 1)     # 1x1 conv (입력 채널 수를 출력 채널 수에 맞춰 변경)
        else:
            self.skip = nn.Identity()       # identity 연결
    def forward(self, x):
        identity = self.skip(x)     # skip 경로로 변환
        out = self.relu(self.bn1(self.conv1(x)))       # conv -> norm -> ReLU
        out = self.bn2(self.conv2(out))     # conv -> norm
        out += identity         # Residual Learning : skip + main 경로 결과
        out = self.relu(out)        # ReLU
        return out


# Attention Gate
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):        # Decoder(gate), Encoder(local), intermediate feature
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int)     # 1x1 conv -> norm
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int)     # 1x1 conv -> norm
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid()     # 두 입력결과를 더해 비선형 처리 -> 1x1 conv -> sigmoid : gate mask 생성
        )
        self.relu = nn.ReLU()
    def forward(self, g, x):        # Decoder의 feature와 Encoder의 skip을 받아 gate 출력
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)        # 합산 후 ReLU
        psi = self.psi(psi)     # Attention mask (squeeze channel)
        return x * psi


# U-Net with Residual Blocks and Attention Gates
class AttResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):        # 입력 채널, 출력 채널, feature 크기 리스트 
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock(in_channels, features[0]),
            ResidualBlock(features[0], features[1]),
            ResidualBlock(features[1], features[2]),
            ResidualBlock(features[2], features[3]),
        ])
        self.pool = nn.MaxPool2d(2)     # 2x2 max pooling -> 이미지 축소
        self.bottleneck = ResidualBlock(features[3], features[3]*2)
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(features[3]*2, features[3], 2, stride=2),
            nn.ConvTranspose2d(features[3], features[2], 2, stride=2),
            nn.ConvTranspose2d(features[2], features[1], 2, stride=2),
            nn.ConvTranspose2d(features[1], features[0], 2, stride=2),
        ])
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(features[3]*2, features[3]),
            ResidualBlock(features[2]*2, features[2]),
            ResidualBlock(features[1]*2, features[1]),
            ResidualBlock(features[0]*2, features[0]),
        ])
        # Attention Gates
        self.attention_gates = nn.ModuleList([
            AttentionGate(F_g=features[3], F_l=features[3], F_int=features[2]),
            AttentionGate(F_g=features[2], F_l=features[2], F_int=features[1]),
            AttentionGate(F_g=features[1], F_l=features[1], F_int=features[0]),
            AttentionGate(F_g=features[0], F_l=features[0], F_int=features[0]//2),
        ])
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
    def forward(self, x):       # batch, channel, height, width
        skips = []
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            # Attention Gate on skip connection
            attn = self.attention_gates[i](g=x, x=skips[-(i+1)])
            x = torch.cat([attn, x], dim=1)
            x = self.decoder_blocks[i](x)
        return self.final_conv(x)
