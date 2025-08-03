import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, embedding_dim=32):
        super(UNet, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=20, embedding_dim=embedding_dim)

        # Initial convolution: image + embedded color
        self.inc = self.double_conv(n_channels + embedding_dim, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)

        # Upsampling layers with proper channel math
        self.up1_upsample = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1_conv = self.double_conv(128 + 128, 128)

        self.up2_upsample = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2_conv = self.double_conv(64 + 64, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x_img, x_text):
        # Embed color and expand to match image height/width
        emb = self.embedding(x_text).unsqueeze(2).unsqueeze(3)
        emb = emb.expand(-1, -1, x_img.size(2), x_img.size(3))

        # Concatenate image and color embedding
        x = torch.cat([x_img, emb], dim=1)

        # Encoder
        x1 = self.inc(x)        # -> [B, 64, 128, 128]
        x2 = self.down1(x1)     # -> [B, 128, 64, 64]
        x3 = self.down2(x2)     # -> [B, 256, 32, 32]

        # Decoder
        x = self.up1_upsample(x3)           # -> [B, 128, 64, 64]
        x = torch.cat([x, x2], dim=1)       # -> [B, 256, 64, 64]
        x = self.up1_conv(x)                # -> [B, 128, 64, 64]

        x = self.up2_upsample(x)            # -> [B, 64, 128, 128]
        x = torch.cat([x, x1], dim=1)       # -> [B, 128, 128, 128]
        x = self.up2_conv(x)                # -> [B, 64, 128, 128]

        return self.outc(x)                 # -> [B, 3, 128, 128]

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def down(self, in_ch, out_ch):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_ch, out_ch)
        )
