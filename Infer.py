# python imports
import os
import glob
import warnings
warnings.filterwarnings("ignore")
# external imports
# import cv2
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from Model import losses
# from Model.config import args
from Models.config import args
from Model.datagenerators_atlas import Dataset
from Model.model import U_Network, SpatialTransformer

from Models.TransMatch import TransMatch
import utils
from natsort import natsorted
import time
#from torchkeras import summary

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    # cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
    #         63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
    #         163, 164, 165, 166]
    # cls_lst = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 77, 85]
    cls_lst = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
    # cls_lst = [182]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


def infer():

    # 创建需要的文件夹并指定gpu
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 日志文件
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    # f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")
    # f_val = open(os.path.join(args.log_dir, "val_log.txt"), "w")

    # 读入fixed图像 [D, W, H] = 160×192×160
    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    # print('The shape of fixed image:', input_fixed.shape, type(input_fixed))
    # slide = cv2.imshow('test', input_fixed[0, 0, 80, :, :])
    # cv2.waitKey()
    vol_size = input_fixed.shape[2:]
    # [B, C, D, W, H]
    input_fixed_eval = torch.from_numpy(input_fixed).to(device).float()
    input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)
    input_fixed = torch.from_numpy(input_fixed).to(device).float()
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, "fixed.delineation.structure.label.nii.gz")))[np.newaxis, np.newaxis, ...]
    fixed_label = torch.from_numpy(fixed_label).to(device).float()



    # 创建配准网络（net）和STN
    net = TransMatch(args).to(device)
    best_model = torch.load('./experiments/dsc0.765epoch104.pth.tar')['state_dict']
    net.load_state_dict(best_model)

    # # 定义总参数量、可训练参数量及非可训练参数量变量
    # Total_params = 0
    # Trainable_params = 0
    # NonTrainable_params = 0
    # # 遍历model.parameters()返回的全局参数列表
    # for param in net.parameters():
    #     mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
    #     Total_params += mulValue  # 总参数量
    #     if param.requires_grad:
    #         Trainable_params += mulValue  # 可训练参数量
    #     else:
    #         NonTrainable_params += mulValue  # 非可训练参数量
    # print(f'Total params: {Total_params}')
    # print(f'Trainable params: {Trainable_params}')
    # print(f'Non-trainable params: {NonTrainable_params}')
    # #
    # # summary(net.cuda(), [[160, 192, 160], [160, 192, 160]], batch_size=1)
    # return 0
    iterEpoch = 1
    contTrain = False
    if contTrain:
        checkpoint = torch.load('./Checkpoint/500.pth')
        net.load_state_dict(checkpoint)
        iterEpoch = 501

    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    # UNet.train()
    net.train()
    STN.train()
    # 模型参数个数
    # print("UNet: ", count_parameters(UNet))
    # print("Transformer: ", count_parameters(net))
    # print("STN: ", count_parameters(STN))

    # Set optimizer and losses
    # opt = Adam(UNet.parameters(), lr=args.lr)
    opt = Adam(net.parameters(), lr=args.lr)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss
    # zero_loss_fn = nn.MSELoss().to(device)

    # zero = np.zeros((1, 3, 160, 192, 160), dtype=int)
    # zero = torch.from_numpy(zero).to(device).float()

    # Get all the names of the training data
    train_files = glob.glob(os.path.join(args.train_dir, '*.nii.gz'))
    DS = Dataset(files=train_files)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Test file and anatomical labels we want to evaluate
    test_file_lst = glob.glob(os.path.join('../../Dataset/IXIIXI_delineation/delineation_l_norm/test', "*.nii.gz"))
    # print("The number of test data: ", len(test_file_lst))

    # Set up model
    # UNet = ImageDepthNet(args).to(device)
    # UNet.load_state_dict(torch.load(args.checkpoint_path))
    # STN_img = SpatialTransformer(vol_size).to(device)
    net.eval()
    STN.eval()
    STN_label.eval()

    # Test Folder
    TIME = []
    for file in test_file_lst:
        # Train
        # fig_name = file[59:61]
        # Test.
        # fig_name = file[58:60]
        
        name = os.path.split(file)[1]
        fig_name = name[:4]
        # 读入moving图像
        input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
        input_moving = torch.from_numpy(input_moving).to(device).float()
        # 读入moving图像对应的label
        label_file = glob.glob(os.path.join(args.label_dir, name[:4] + "*"))[0]
        input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))[np.newaxis, np.newaxis, ...]
        input_label = torch.from_numpy(input_label).to(device).float()

        # 获得配准后的图像和label
        start = time.time()
        pred_flow = net(input_fixed_eval, input_moving)
        pred_img = STN(input_fixed_eval, pred_flow)
        TIME.append(time.time()-start)
        pred_label = STN_label(fixed_label, pred_flow)
        # pred_label = input_label # 用于测试初始的dice值

        # 计算DSC
        # dice = compute_label_dice(input_label.cpu(), pred_label[0, 0, ...].cpu().detach().numpy())
        # dice = utils.dice_val(pred_label.long(), fixed_label.long(), 46)
        # print("{0}" .format(dice))
        # DSC.append(dice)

        # Save image file
        # Train
        # tmpName = str(file[58:61])
        # Test
        tmpName = fig_name
        # print(tmpName)
        save_image(pred_img, f_img, tmpName + '_warpped.nii.gz')
        save_image(pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, tmpName + "_flow.nii.gz")
        save_image(pred_label, f_img, tmpName + "_label.nii.gz")
        # save_image(input_moving, f_img, tmpName + "_original.nii.gz")
        del pred_flow, pred_label, pred_img, input_moving, input_label
        print('ok')
    print(TIME)
    print(np.mean(TIME))


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    model_lists = natsorted(glob.glob(save_dir+  '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))
    torch.save(state, save_dir+filename)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    infer()
