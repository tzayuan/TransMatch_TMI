# Support multimodal image inference, this inference example code is based on the SRSEG dataset as an example

import os
import time
import glob
import argparse

from scipy.ndimage import zoom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transmatch.STN import SpatialTransformer
from dataset import load_vol
from utils import dice, jacobian_determinant,minmax_norm

parser = argparse.ArgumentParser()
# Please modify the path of 'transmatchbest.pt'. It is recommended to use an absolute path 
parser.add_argument("--load_model", type=str, default="/home/snowball/Documents/Code/TransMatch_mm/RegistrationNets/save_model_pt/transmatchbest.pt")
parser.add_argument("--gpu_id", type=str, default="0")

args = parser.parse_args()
load_model = args.load_model
test_data = args.test_data
gpu_id = args.gpu_id

def test():
    # Please modify the path of SR-REG/vol
    vol_path = "/home/snowball/Documents/Code/TransMatch_mm/SR-REG/vol/"
    vol_all_data = os.listdir(vol_path)
    vol_dir_data = []

    # Please modify the path of SR-REG/seg
    seg_path = "/home/snowball/Documents/Code/TransMatch_mm/SR-REG/seg/"
    seg_all_data = os.listdir(seg_path)
    seg_dir_data = []

    for file in vol_all_data:
        if not os.path.isdir(vol_path + file):
            vol_dir_data.append(vol_path + file)

    for file in seg_all_data:
        if not os.path.isdir(seg_path + file):
            seg_dir_data.append(seg_path + file)

    val_mr_database = []
    val_ct_database = []
    val_seg_database = []

    for data in vol_dir_data:
        if "_mr.nii.gz" in data:
            val_mr_database.append(data)
        if "_ct.nii.gz" in data:
            val_ct_database.append(data)
    for data in seg_dir_data:
        val_seg_database.append(data)

    val_mr_database.sort()
    val_mr_database = val_mr_database[150:160]
    val_ct_database.sort()
    val_ct_database = val_ct_database[150:160]
    val_seg_database.sort()
    val_seg_database = val_seg_database[150:160]

    model_stn = SpatialTransformer((192, 224, 192), "nearest")

    # 在加载模型前添加模块映射
    # line 67~76 代码的作用: Torh1.x下训练的模型兼容到Torch2.x下的推理
    import sys
    from types import ModuleType

    class LayersDropStub(ModuleType):
        def __init__(self):
            super().__init__('timm.models.layers.drop')
            from timm.layers import DropPath
            self.DropPath = DropPath

    sys.modules['timm.models.layers.drop'] = LayersDropStub()

    model = torch.load(load_model, map_location='cuda:0')
   
    device = torch.device("cuda:" + gpu_id)

    model.to(device)
    model_stn.to(device)

    count = 0
    avg_dice = []
    avg_time = []
    avg_jac = []

    init_dice = []

    with torch.no_grad(): 
        for ii in range(0, len(val_mr_database), 2):
            moving_image, fixed_image = load_vol(val_mr_database[ii]), load_vol(val_ct_database[ii+1])
            moving_image = minmax_norm(moving_image)
            fixed_image = minmax_norm(fixed_image)
            moving_label, fixed_label = load_vol(val_seg_database[ii]), load_vol(val_seg_database[ii+1])
            
            moving_image = np.pad(moving_image, ((8, 8),(8, 8),(0, 0)), 'constant', constant_values=0)
            moving_label = np.pad(moving_label, ((8, 8),(8, 8),(0, 0)), 'constant', constant_values=0)
            fixed_image = np.pad(fixed_image, ((8, 8),(8, 8),(0, 0)), 'constant', constant_values=0)
            fixed_label = np.pad(fixed_label, ((8, 8),(8, 8),(0, 0)), 'constant', constant_values=0)
            
            moving = torch.from_numpy(moving_image).to(device)[None, None, ...].float()
            mov_lab = torch.from_numpy(moving_label).to(device)[None, None, ...].float()
            fixed = torch.from_numpy(fixed_image).to(device)[None, None, ...].float()
            fix_lab = torch.from_numpy(fixed_label).to(device)[None, None, ...].float()
            
            model_in = torch.cat((moving, fixed), dim=1)
            before_time = time.time()
            warped, flow = model(model_in)
            after_time = time.time()

            warp_lab = model_stn(mov_lab, flow)

            avg_time.append(after_time - before_time)

            mov_lab = mov_lab.detach().cpu().numpy().squeeze()
            fix_lab = fix_lab.detach().cpu().numpy().squeeze()
            warp_lab = warp_lab.detach().cpu().numpy().squeeze()
            flow = flow.squeeze().permute(1, 2, 3, 0).cuda().data.cpu().numpy()
            good_labels = np.intersect1d(mov_lab, fix_lab)[1:]
            #good_labels = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28]
            dice_val = dice(fix_lab, mov_lab, good_labels)
            init_dice.append(dice_val.mean())
            dice_val = dice(fix_lab, warp_lab, good_labels)
            avg_dice += [dice_val.mean()]
            avg_jac += [np.sum(jacobian_determinant(flow) < 0)]
            #print(avg_dice[-1], avg_jac[-1])

            """
            count += 1
            moving_image = load_vol(val_ct_database[jj])
            fixed_image = load_vol(val_mr_database[ii])
            moving_image = minmax_norm(moving_image)
            fixed_image = minmax_norm(fixed_image)
            moving_label = load_vol(val_seg_database[jj])
            fixed_label = load_vol(val_seg_database[ii])

            fixed = torch.from_numpy(fixed_image).to(device)[None, None, ...].float()
            fix_lab = torch.from_numpy(fixed_label).to(device)[None, None, ...].float()
            moving = torch.from_numpy(moving_image).to(device)[None, None, ...].float()
            mov_lab = torch.from_numpy(moving_label).to(device)[None, None, ...].float()
            """

            model_in = torch.cat((fixed, moving), dim=1)
            warped, flow = model(model_in)
            fix_lab = torch.from_numpy(fix_lab).to(device)[None, None, ...].float()
            warp_lab = model_stn(fix_lab, flow)

            # mov_lab = mov_lab.detach().cpu().numpy().squeeze()
            fix_lab = fix_lab.detach().cpu().numpy().squeeze()
            warp_lab = warp_lab.detach().cpu().numpy().squeeze()
            flow = flow.squeeze().permute(1, 2, 3, 0).cuda().data.cpu().numpy()

            dice_val = dice(mov_lab, warp_lab, good_labels)
            # print("after dice values:", dice_val.mean())
            avg_dice += [dice_val.mean()]
            avg_jac += [np.sum(jacobian_determinant(flow) < 0)]
            #print(avg_dice[-1], avg_jac[-1])

    print("Dice before registration: ")
    print(np.mean(init_dice), np.std(init_dice))
    print("Dice after registration: ")
    print(np.mean(avg_dice), np.std(avg_dice))
    print("Ave time (s): ")
    print(np.mean(avg_time), np.std(avg_time))
    print("Ave folding: ")
    print(np.mean(avg_jac), np.std(avg_jac))

if __name__ == "__main__":
    test()
