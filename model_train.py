import torch
import torch.nn as nn
import torch.nn.parallel


################################# Encoder ######################################

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

def downBlock(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=False)
    )
    return block

def sameBlock(in_planes, out_planes):
    block = nn.Sequential( conv3x3(in_planes, out_planes),
                           nn.BatchNorm2d(out_planes),
                           nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def upBlock(in_channels, out_channels):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=False)
    )
    return block

class NetworkShadow(nn.Module):
    def __init__(self):
        super(NetworkShadow, self).__init__()

        self.encoder_geometry1 = nn.Sequential(
            # 4 * 240 * 320
            downBlock(4, 32),
            sameBlock(32, 32)
        )
        self.encoder_geometry2 = nn.Sequential(
            # 32 * 120 * 160
            downBlock(32, 64),
            sameBlock(64, 64)
        )
        self.encoder_geometry3 = nn.Sequential(
            # 64 * 60 * 80
            downBlock(64, 128),
            sameBlock(128, 128)
        )
        self.encoder_geometry4 = nn.Sequential(
            # 128 * 30 * 40
            downBlock(128, 64),
            sameBlock(64, 64)
            # 64 * 15 * 20
        )


        self.encoder_global1 = nn.Sequential(
            # 4 * 240 * 320
            downBlock(4, 32),
            sameBlock(32, 32)
        )
        self.encoder_global2 = nn.Sequential(
            # 64 * 120 * 160
            downBlock(64, 64),
            sameBlock(64, 64)
        )
        self.encoder_global3 = nn.Sequential(
            # 128 * 60 * 80
            downBlock(128, 128),
            sameBlock(128, 128),
        )
        self.encoder_global4 = nn.Sequential(
            # 256 * 30 * 40
            downBlock(256, 128),
            sameBlock(128, 64)
            # 64 * 15 * 20
        )

        self.decoder_combine = nn.Sequential(
            # 128 * 15 * 20
            upBlock(128, 128),
            sameBlock(128, 64),
            # 64 * 30 * 40
            upBlock(64, 32),
            sameBlock(32, 32),
            # 32 * 60 * 80
            upBlock(32, 16),
            sameBlock(16, 16),
            # 16 * 120 * 160
            upBlock(16, 1),
            # 32 * 240 * 320
            conv3x3(1, 1),
            # 1 * 240 * 320
            nn.Sigmoid()
        )


    def forward(self, depth_model, normal_model, image_background, mask_shadow_background):

        feature_geometry = torch.cat((depth_model, normal_model), 1)                   # 1 3
        feature_geometry1 = self.encoder_geometry1(feature_geometry)
        feature_geometry2 = self.encoder_geometry2(feature_geometry1)
        feature_geometry3 = self.encoder_geometry3(feature_geometry2)
        feature_geometry4 = self.encoder_geometry4(feature_geometry3)

        feature_global = torch.cat((mask_shadow_background, image_background), 1)   # 1 3
        # feature_global = image_background # 3
        feature_global1 = self.encoder_global1(feature_global)
        feature_global1 = torch.cat((feature_global1, feature_geometry1), 1)
        feature_global2 = self.encoder_global2(feature_global1)
        feature_global2 = torch.cat((feature_global2, feature_geometry2), 1)
        feature_global3 = self.encoder_global3(feature_global2)
        feature_global3 = torch.cat((feature_global3, feature_geometry3), 1)
        feature_global4 = self.encoder_global4(feature_global3)
        # 64 * 15 * 20

        feature_combine = torch.cat((feature_geometry4, feature_global4), 1)



        shadow_img = self.decoder_combine(feature_combine)

        return shadow_img
