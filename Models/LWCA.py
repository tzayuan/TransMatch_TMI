'''
LWCA module

A partial code was retrieved from:
https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration

Swin-Transformer code was retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

Original Swin-Transformer paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.
'''

import Models.basic_LWCA as basic
import torch.nn as nn
import utils.configs_TransMatch as configs

class LWCA(nn.Module):
    def __init__(self, config, dim_diy):
        super(LWCA, self).__init__()
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
                                           pat_merg_rf=config.pat_merg_rf,
                                           dim_diy=dim_diy
                                           )

    def forward(self, x, y):
        moving_fea_cross = self.transformer(x, y)
        return moving_fea_cross


CONFIGS = {
    'TransMatch_LPBA40': configs.get_TransMatch_LPBA40_config(),
}
