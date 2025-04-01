# Train script for multimodal registration (SR-REG)
# You should modifiled TransMatch.py, line 39 to 'self.stn =SpatialTransformer((192, 224, 192))'

import os
import time
import copy
import glob
import argparse

import torch
from torch import optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from Models.STN import SpatialTransformer
import utils.losses_mm as losses
import utils.utils as utils
from utils.dataset_mm import data_generator_double, load_vol

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="transmatch")
parser.add_argument("--max_epoches", type=int, default=500)
parser.add_argument("--base_lr", type=float, default=0.0001)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--load_pt", type=bool, default=False)

args = parser.parse_args()
model_name = args.model_name
lr = args.base_lr
max_epoches = args.max_epoches
gpu_id = args.gpu_id
load_pt = args.load_pt


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def train():
    vol_path = "/home/snowball/Documents/Code/TransMatch_mm/SR-REG/vol/"
    vol_all_data = os.listdir(vol_path)
    vol_dir_data = []

    seg_path = "/home/snowball/Documents/Code/TransMatch_mm/SR-REG/seg/"
    seg_all_data = os.listdir(seg_path)
    seg_dir_data = []

    for file in vol_all_data:
        if not os.path.isdir(vol_path + file):
            vol_dir_data.append(vol_path + file)

    for file in seg_all_data:
        if not os.path.isdir(seg_path + file):
            seg_dir_data.append(seg_path + file)

    train_mr_database = []
    train_ct_database = []
    train_seg_database = []

    for data in vol_dir_data:
        if "_mr.nii.gz" in data:
            train_mr_database.append(data)
        if "_ct.nii.gz" in data:
            train_ct_database.append(data)
    for data in seg_dir_data:
        train_seg_database.append(data)

    train_mr_database.sort()
    train_mr_database = train_mr_database[:150]
    train_ct_database.sort()
    train_ct_database = train_ct_database[:150]
    train_seg_database.sort()
    train_seg_database = train_seg_database[:150]

    in_shape= (192, 224, 192)
    model_stn = SpatialTransformer(in_shape, "bilinear")
    device = torch.device("cuda:" + gpu_id)
    model_stn.to(device)
    if model_name == "transmatch":
        from Models.TransMatch_mm import TransMatch
        model = TransMatch("123")

    lr = 0.0001
    max_epoches = 500
    steps_per_epoch = 100
    criterions = [losses.Dice_mambamorph().loss, losses.Grad3d(penalty='l2'), losses.SpatialSegmentSmoothness()]
    weights = [1.0, 0.1]

    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    train_gen = data_generator_double(train_ct_database, train_mr_database, train_seg_database)

    labels = [0., 2., 3., 4., 5., 7., 8., 10., 11., 12., 13., 14., 15., 16., 17., 18., 24., 26., 28.]
    # ignore_label = [0, 5, 24]
    for epoch in range(0, max_epoches):

        for idx in range(steps_per_epoch):
            start_time = time.time()

            data, _ = next(train_gen)
            moving = data[0]
            fixed = data[1]
            moving_mask = data[2]
            fixed_mask = data[3]

            moving = moving[None, None, ...]
            fixed = fixed[None, None, ...]
            moving_mask = moving_mask[None, ..., None]
            fixed_mask = fixed_mask[None, ..., None]
            moving_mask = utils.split_seg_global(moving_mask, labels)
            fixed_mask = utils.split_seg_global(fixed_mask, labels)

            moving = torch.from_numpy(moving).to(device).float()
            fixed = torch.from_numpy(fixed).to(device).float()
            moving_mask = torch.from_numpy(moving_mask).to(device).float().permute(0, 4, 1, 2, 3)
            fixed_mask = torch.from_numpy(fixed_mask).to(device).float().permute(0, 4, 1, 2, 3)

            model_in = torch.cat((moving, fixed), dim=1)
            _, flow = model(model_in)

            loss = 0.0
            curr_loss = [0, 0]
            warped_mask = model_stn(moving_mask, flow)
            curr_loss[0] += criterions[0](warped_mask, fixed_mask) * weights[0]
            curr_loss[1] += criterions[1](flow, flow) * weights[1]
            loss += curr_loss[0] + curr_loss[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del model_in
            del warped_mask, flow

            model_in = torch.cat((fixed, moving), dim=1)
            _, flow = model(model_in)

            loss = 0.0
            curr_loss = [0, 0]
            warped_mask = model_stn(fixed_mask, flow)
            curr_loss[0] += criterions[0](moving_mask, warped_mask) * weights[0]
            curr_loss[1] += criterions[1](flow, flow) * weights[1]
            loss += curr_loss[0] + curr_loss[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_time = time.time()
            print("{} of Epoch {}, Loss: {:.4f}, Sim: {:.4f}, Reg: {:.4f}. Cost Time: {:.2f}".format(idx, epoch,
                                                                                                     loss.item(),
                                                                                                     curr_loss[
                                                                                                         0].item(),
                                                                                                     curr_loss[
                                                                                                         1].item(),
                                                                                                     end_time - start_time))

        if epoch % 10 == 0:
            torch.save(model, os.path.join("./save_model_pt/srseg", model_name + str(epoch) + ".pt"))

    torch.save(model, os.path.join("./save_model_pt/srseg", model_name + str(max_epoches) + ".pt"))


if __name__ == "__main__":
    train()
