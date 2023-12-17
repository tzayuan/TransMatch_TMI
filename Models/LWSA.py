'''
LWSA module

A partial code was retrieved from:
https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration

Swin-Transformer code was retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

Original Swin-Transformer paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.
'''

import Models.basic_LWSA as basic
import Models.Conv3dReLU as Conv3dReLU
import torch.nn as nn
import utils.configs_TransMatch as configs

class LWSA(nn.Module):
    def __init__(self, config):
        super(LWSA, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = basic.SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf
                                           )
        self.c1 = Conv3dReLU.Conv3dReLU(1, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU.Conv3dReLU(1, config.reg_head_chan, 3, 1, use_batchnorm=False)

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, x):
        # print('The shape of x(Input of transformer function):', x.shape)
        source = x[:, 0:1, :, :, :]
        #print('The shape of source:', source.shape)
        if self.if_convskip:
            x_s0 = x.clone()  # 用于concat AB的直接卷积的input
            # print('The shape of x_s0(X.clone):', x_s0.shape)
            x_s1 = self.avg_pool(x)  # 用于concat AB后下采样1/2后的卷积的input
            # print('The shape of x_s1:', x_s1.shape)
            f4 = self.c1(x_s1)  # 下采样后的卷积
            # print('The shape of f4', f4.shape)
            f5 = self.c2(x_s0)  # 原始图像的卷积
            # print('The shape of f5:', f5.shape)
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        return f3, f2, f1, out_feats[-1]


CONFIGS = {
    'TransMatch_LPBA40': configs.get_TransMatch_LPBA40_config()
}
