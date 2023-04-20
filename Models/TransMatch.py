import torch.nn as nn
import torch
import Models.configs_LWSA_LWCA as configs
import Models.LWSA as LWSA
import Models.LWCA as LWCA
import Models.Decoder as Decoder




class TransMatch(nn.Module):
    def __init__(self, args):
        super(TransMatch, self).__init__()

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.opt_conv = LWSA.Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)


        config_LWSA = configs.get_LWSA_config()
        config_LWCA = configs.get_LWCA_config()

        #LWSA
        self.moving_backbone = LWSA.LWSA(config_LWSA)
        # !TODO: parameter sharing
        # self.fixed_backbone = LWSA.LWSA(config_LWSA)
        self.fixed_backbone = self.moving_backbone

        # LWCA
        self.crossattn1 = LWCA.LWCA(config_LWCA, dim_diy=96)
        self.crossattn2 = LWCA.LWCA(config_LWCA, dim_diy=192)
        self.crossattn3 = LWCA.LWCA(config_LWCA, dim_diy=384)
        self.crossattn4 = LWCA.LWCA(config_LWCA, dim_diy=768)


        self.up0 = Decoder.DecoderBlock(768, 384, skip_channels=384, use_batchnorm=False)
        self.up1 = Decoder.DecoderBlock(384, 192, skip_channels=192, use_batchnorm=False)
        self.up2 = Decoder.DecoderBlock(192, 96, skip_channels=96, use_batchnorm=False)
        self.up3 = Decoder.DecoderBlock(96, 48, skip_channels=48, use_batchnorm=False)
        self.up4 = Decoder.DecoderBlock(48, 16, skip_channels=16, use_batchnorm=False)  # 384, 160, 160, 256
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.reg_head = Decoder.RegistrationHead(
            in_channels=48,
            out_channels=3,
            kernel_size=3,
        )

    def forward(self, moving_Input, fixed_Input):

        input_fusion = torch.cat((moving_Input, fixed_Input), dim=1)
        # print(input_fusion.shape)
        # print('The shape of x_s0(X.clone):', x_s0.shape)
        x_s1 = self.avg_pool(input_fusion)  # 用于concat AB后下采样1/2后的卷积的input

        # print('The shape of x_s1:', x_s1.shape)
        f4 = self.opt_conv(x_s1)  # 下采样后的卷积
        # print('The shape of f4', f4.shape)
        # f5 = self.c2(input_fusion)  # 原始图像的卷积
        # print('The shape of f5:', f5.shape)

        B, _, _, _, _ = moving_Input.shape  # Batch, channel, height, width, depth

        moving_fea_4, moving_fea_8, moving_fea_16, moving_fea_32 = self.moving_backbone(moving_Input)
        # print('The out of moving image after swin transformer:', moving_fea_4.shape, moving_fea_8.shape, moving_fea_16.shape, moving_fea_32.shape)
        fixed_fea_4, fixed_fea_8, fixed_fea_16, fixed_fea_32 = self.moving_backbone(fixed_Input)
        # print('The out of fixed iamge after swin transformer:', fixed_fea_4.shape, fixed_fea_8.shape, fixed_fea_16.shape, fixed_fea_32.shape)

        # LWCA module
        moving_fea_4_cross = self.crossattn1(moving_fea_4, fixed_fea_4)
        # print('The shape of moving_fea_4_cross:', moving_fea_4_cross.shape)
        moving_fea_8_cross = self.crossattn2(moving_fea_8, fixed_fea_8)
        # print('The shape of moving_fea_8_cross:', moving_fea_8_cross.shape)
        moving_fea_16_cross = self.crossattn3(moving_fea_16, fixed_fea_16)
        # print('The shape of moving_fea_16_cross:', moving_fea_16_cross.shape)
        moving_fea_32_cross = self.crossattn4(moving_fea_32, fixed_fea_32)
        # print('The shape of moving_fea_32_cross:', moving_fea_32_cross.shape)
        fixed_fea_4_cross = self.crossattn1(fixed_fea_4, moving_fea_4)
        # print('The shape of moving_fea_4_cross:', moving_fea_4_cross.shape)
        fixed_fea_8_cross = self.crossattn2(fixed_fea_8, moving_fea_8)
        # print('The shape of moving_fea_8_cross:', moving_fea_8_cross.shape)
        fixed_fea_16_cross = self.crossattn3(fixed_fea_16, moving_fea_16)
        # print('The shape of moving_fea_16_cross:', moving_fea_16_cross.shape)
        fixed_fea_32_cross = self.crossattn4(fixed_fea_32, moving_fea_32)
        # print('The shape of moving_fea_32_cross:', moving_fea_32_cross.shape)

        # 这里没有concat下采样到32的moving image和fixed image, 后面可以尝试一下
        # 这里上采样最后直接线性插值了，因为显存不够了
        x = self.up0(moving_fea_32_cross, moving_fea_16_cross, fixed_fea_16_cross)
        x = self.up1(x, moving_fea_8_cross, fixed_fea_8_cross)
        x = self.up2(x, moving_fea_4_cross, fixed_fea_4_cross)
        x = self.up3(x, f4)
        x = self.up(x)
        outputs = self.reg_head(x)
        # print(outputs.shape)
        return outputs
