import torch.nn as nn
import torch
import Models.Conv3dReLU as Conv3dReLU
import Models.LWSA as LWSA
import Models.LWCA as LWCA
import Models.Decoder as Decoder
import utils.configs_TransMatch as configs


class TransMatch(nn.Module):
    def __init__(self, args):
        super(TransMatch, self).__init__()

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU.Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU.Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        config2 = configs.get_TransMatch_LPBA40_config()
        self.moving_lwsa = LWSA.LWSA(config2)
        self.fixed_lwsa = LWSA.LWSA(config2)

        self.lwca1 = LWCA.LWCA(config2, dim_diy=96)
        self.lwca2 = LWCA.LWCA(config2, dim_diy=192)
        self.lwca3 = LWCA.LWCA(config2, dim_diy=384)
        self.lwca4 = LWCA.LWCA(config2, dim_diy=768)

        self.up0 = Decoder.DecoderBlock(768, 384, skip_channels=384, use_batchnorm=False)
        self.up1 = Decoder.DecoderBlock(384, 192, skip_channels=192, use_batchnorm=False)
        self.up2 = Decoder.DecoderBlock(192, 96, skip_channels=96, use_batchnorm=False)
        self.up3 = Decoder.DecoderBlock(96, 48, skip_channels=48, use_batchnorm=False)
        self.up4 = Decoder.DecoderBlock(48, 16, skip_channels=16, use_batchnorm=False)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.reg_head = Decoder.RegistrationHead(
            in_channels=48,
            out_channels=3,
            kernel_size=3,
        )

    def forward(self, moving_Input, fixed_Input):

        input_fusion = torch.cat((moving_Input, fixed_Input), dim=1)

        x_s1 = self.avg_pool(input_fusion)  # 用于concat AB后下采样1/2后的卷积的input

        f4 = self.c1(x_s1)  # 下采样后的卷积
        f5 = self.c2(input_fusion)  # 原始图像的卷积

        B, _, _, _, _ = moving_Input.shape  # Batch, channel, height, width, depth

        moving_fea_4, moving_fea_8, moving_fea_16, moving_fea_32 = self.moving_lwsa(moving_Input)
        fixed_fea_4, fixed_fea_8, fixed_fea_16, fixed_fea_32 = self.moving_lwsa(fixed_Input)

        moving_fea_4_cross = self.lwca1(moving_fea_4, fixed_fea_4)
        moving_fea_8_cross = self.lwca2(moving_fea_8, fixed_fea_8)
        moving_fea_16_cross = self.lwca3(moving_fea_16, fixed_fea_16)
        moving_fea_32_cross = self.lwca4(moving_fea_32, fixed_fea_32)

        fixed_fea_4_cross = self.lwca1(fixed_fea_4, moving_fea_4)
        fixed_fea_8_cross = self.lwca2(fixed_fea_8, moving_fea_8)
        fixed_fea_16_cross = self.lwca3(fixed_fea_16, moving_fea_16)
        fixed_fea_32_cross = self.lwca4(fixed_fea_32, moving_fea_32)


        x = self.up0(moving_fea_32_cross, moving_fea_16_cross, fixed_fea_16_cross)
        x = self.up1(x, moving_fea_8_cross, fixed_fea_8_cross)
        x = self.up2(x, moving_fea_4_cross, fixed_fea_4_cross)
        x = self.up3(x, f4)
        x = self.up(x)
        outputs = self.reg_head(x)

        return outputs
