# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from utils import losses
from utils.config import args
from utils.datagenerators_atlas import Dataset
from Models.STN import SpatialTransformer
from natsort import natsorted

from Models.TransMatch import TransMatch

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
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
            63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
            163, 164, 165, 166]
    # cls_lst = [182]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


def train():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 日志文件
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    # 读入fixed图像 [D, W, H] = 160×192×160
    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    # [B, C, D, W, H]
    input_fixed_eval = torch.from_numpy(input_fixed).to(device).float()
    input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)
    input_fixed = torch.from_numpy(input_fixed).to(device).float()
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, "S01.delineation.structure.label.nii.gz")))[np.newaxis, np.newaxis, ...]
    fixed_label = torch.from_numpy(fixed_label).to(device).float()


    # 创建配准网络（net）和STN
    net = TransMatch(args).to(device)

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

    opt = Adam(net.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # Get all the names of the training data
    train_files = glob.glob(os.path.join(args.train_dir, '*.nii.gz'))
    DS = Dataset(files=train_files)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Training loop.
    for i in range(iterEpoch, args.n_iter + 1):
        # Generate the moving images and convert them to tensors.
        net.train()
        STN.train()
        print('epoch:', i)
        input_moving_all = iter(DL)
        for input_moving, fig_name in input_moving_all:
            # [B, C, D, W, H]
            fig_name = fig_name[0]
            input_moving = input_moving.to(device).float()

            # Run the data through the model to produce warp and flow field

            flow_m2f = net(input_fixed, input_moving)
            m2f = STN(input_fixed, flow_m2f)

            # Calculate loss
            sim_loss = sim_loss_fn(m2f, input_moving)
            grad_loss = grad_loss_fn(flow_m2f)
            # zero_loss = zero_loss_fn(flow_m2f, zero)
            loss = sim_loss + args.alpha * grad_loss #  + zero_loss

            print("%d, %s, %f, %f, %f" % (i, fig_name, loss.item(), sim_loss.item(), grad_loss.item()), file=f)

            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            # inverse fixed image and moving image
            flow_m2f = net(input_moving, input_fixed)
            m2f = STN(input_moving, flow_m2f)


            # Calculate loss
            sim_loss = sim_loss_fn(m2f, input_fixed)
            grad_loss = grad_loss_fn(flow_m2f)
            # zero_loss = zero_loss_fn(flow_m2f, zero)
            loss = sim_loss + args.alpha * grad_loss #  + zero_loss

            print("%d, %s, %f, %f, %f" % (i, fig_name, loss.item(), sim_loss.item(), grad_loss.item()), file=f)

            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

        test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))

        net.eval()
        STN.eval()
        STN_label.eval()

        DSC = []
        for file in test_file_lst:
            fig_name = file[58:60]
            name = os.path.split(file)[1]
            # 读入moving图像
            input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
            input_moving = torch.from_numpy(input_moving).to(device).float()
            # 读入moving图像对应的label
            label_file = glob.glob(os.path.join(args.label_dir, name[:3] + "*"))[0]
            input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))

            # 获得配准后的图像和label
            pred_flow = net(input_fixed_eval, input_moving)
            pred_img = STN(input_fixed_eval, pred_flow)
            pred_label = STN_label(fixed_label, pred_flow)
            # pred_label = input_label # 用于测试初始的dice值

            # 计算DSC
            dice = compute_label_dice(input_label, pred_label[0, 0, ...].cpu().detach().numpy())
            print("{0}" .format(dice))
            DSC.append(dice)

            del pred_flow, pred_img, pred_label, input_moving

        print(np.mean(DSC), np.std(DSC))
        save_checkpoint({
            'epoch': i+1,
            'state_dict': net.state_dict(),
            'optimizer': opt.state_dict(),
            }, save_dir='experiments/1212firstrunorigincode/', filename='dsc{:.4f}epoch{:0>3d}.pth.tar'.format(np.mean(DSC), i+1))

    f.close()

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    model_lists = natsorted(glob.glob(save_dir+  '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, save_dir+filename)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
