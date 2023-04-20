import argparse

parser = argparse.ArgumentParser()

# 公共参数
parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')
parser.add_argument("--atlas_file", type=str, help="gpu id number",
                    dest="atlas_file", default='../../Dataset/IXIIXI_delineation/delineation_l_norm/fixed.nii.gz')
parser.add_argument("--model", type=str, help="voxelmorph 1 or 2",
                    dest="model", choices=['vm1', 'vm2'], default='vm2')
parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='./Result')

# train时参数
parser.add_argument("--train_dir", type=str, help="data folder with training vols",
                    dest="train_dir", default="../../Dataset/IXIIXI_delineation/delineation_l_norm/train")
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=4e-4)
parser.add_argument("--n_iter", type=int, help="number of iterations",
                    dest="n_iter", default=8000)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                    dest="sim_loss", default='ncc')
parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=4)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
parser.add_argument("--n_save_iter", type=int, help="frequency of model saves",
                    dest="n_save_iter", default=100)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='./Checkpoint')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='./Log')
# train
parser.add_argument('--Training', default=False, type=bool, help='Training or not')
parser.add_argument('--init_method', default='tcp://127.0.0.1:33111', type=str, help='init_method')
parser.add_argument('--data_root', default='./Data/', type=str, help='data path')
parser.add_argument('--train_steps', default=40000, type=int, help='train_steps')
parser.add_argument('--img_size', default=224, type=int, help='network input size')
parser.add_argument('--pretrained_model', default='./pretrained_model/80.7_T2T_ViT_t_14.pth.tar', type=str,
                    help='load pretrained model')
parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
# parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
# parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
parser.add_argument('--stepvalue1', default=20000, type=int, help='the step 1 for adjusting lr')
parser.add_argument('--stepvalue2', default=30000, type=int, help='the step 2 for adjusting lr')
parser.add_argument('--trainset', default='NJUD+NLPR+DUTLF-Depth', type=str, help='Trainging set')
parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')


# test
parser.add_argument('--Testing', default=False, type=bool, help='Testing or not')
parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
parser.add_argument('--test_paths', type=str, default='NJUD+NLPR+DUTLF-Depth+ReDWeb-S+STERE+SSD+SIP+RGBD135+LFSD')
parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
parser.add_argument('--methods', type=str, default='RGBD_VST', help='evaluated method name')
parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='../../Dataset/IXIIXI_delineation/delineation_l_norm/test')
parser.add_argument("--label_dir", type=str, help="label data directory",
                    dest="label_dir", default='../../Dataset/IXIIXI_delineation/label')
parser.add_argument("--checkpoint_path", type=str, help="model weight file",
                    dest="checkpoint_path", default="./Checkpoint/1800.pth")

args = parser.parse_args()
