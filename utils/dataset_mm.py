import pickle
import random

import numpy as np
import nibabel as nib

import utils

def load_vol_double(data_names, types="vol"):
    vol_data1 = None
    vol_data2 = None

    if data_names.endswith(".pkl"):
       
        f = open(data_names, "rb")
        all_data = pickle.load(f)
        if types == "vol":
            vol_data1 = all_data[0]
            vol_data2 = all_data[1]
        else:
            vol_data1 = all_data[2]
            vol_data2 = all_data[3]
    else:
        assert "unkonwn files"
    
    return vol_data1, vol_data2   



def load_vol(data_name, types="vol"):
    vol_data = None
    if data_name.endswith((".nii.gz", ".nii")):
        vol_data = nib.load(data_name).get_fdata()
    elif data_name.endswith(".pkl"):
        f = open(data_name, "rb")
        if types == "vol":
            vol_data = pickle.load(f)[0]
        else:
            vol_data = pickle.load(f)[1]
    elif data_name.endswith((".npz", ".npy")):
        vol_data = np.load(data_name)["vol_data"]
    else:
        assert "unkown files"

    return vol_data

def save_vol(x, data_name, affine=None):
    if affine is None:
        affine = np.array([[-1, 0, 0, 0], [0, 0, 1, 0],  [0, -1, 0, 0],  [0, 0, 0, 1]], dtype=float)  
        
    nib.Nifti1Image(x, affine).to_filename(data_name)

        
  
  
def data_generator(data, vol_shape=(160, 192, 224), batch_size=1):
    vol_len = len(data)
    ndims = len(vol_shape)

    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    while True:
        idx1 = random.randint(0, vol_len-1)
        idx2 = random.randint(0, vol_len-1)
        moving_name = data[idx1]
        fixed_name = data[idx2]
        moving_image, fixed_image = load_vol(moving_name), load_vol(fixed_name)

        inputs = [moving_image.squeeze(), fixed_image.squeeze()]
        outputs = [fixed_image, zero_phi]

        yield inputs, outputs

def data_generator_dirlab(data, vol_shape=(160, 192, 224), batch_size=1):
    vol_len = len(data)
    ndims = len(vol_shape)

    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    while True:
        idx1 = random.randint(0, vol_len - 1)
        idx2 = random.randint(0, vol_len - 1)
        moving_name = data[idx1]
        fixed_name = data[idx2]
        if moving_name[-12] == fixed_name[-12] and moving_name != fixed_name:
            pass
        else:
            continue
        #print(moving_name, fixed_name)
        moving_image, fixed_image = load_vol(moving_name), load_vol(fixed_name)
        inputs = [moving_image.squeeze(), fixed_image.squeeze()]
        outputs = [fixed_image, zero_phi]

        yield inputs, outputs
        
        
def data_generator_double(data_ct, data_mr, data_seg, vol_shape=(160, 192, 224), batch_size=1):
    vol_len = len(data_ct)
    ndims = len(vol_shape)

    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    while True:
        idx1 = random.randint(0, vol_len - 1)
        idx2 = random.randint(0, vol_len - 1)
        if idx1 == idx2:
          idx2 = (idx2 + 1) % vol_len
          
        #idx3 = random.ranindt(0, 3)
        #(moving_name, fixed_name) = (data_ct[idx1], data_mr[idx2]) if idx3 % 3==0 else (data_mr[idx1], data_mr[idx2]) if idx3 % 3 ==1 else (data_ct[idx1], data_ct[idx2]) 
        (moving_name, fixed_name) = (data_ct[idx1], data_mr[idx2])

        moving_seg_name = data_seg[idx1]
        fixed_seg_name = data_seg[idx2]
        moving_image, fixed_image = load_vol(moving_name), load_vol(fixed_name)
        moving_image = utils.minmax_norm(moving_image)
        fixed_image = utils.minmax_norm(fixed_image)
        moving_mask, fixed_mask = load_vol(moving_seg_name), load_vol(fixed_seg_name)
        fixed_image = np.pad(fixed_image, ((8, 8),(8, 8),(0, 0)), 'constant', constant_values=0)
        moving_image = np.pad(moving_image, ((8, 8),(8, 8),(0, 0)), 'constant', constant_values=0)
        fixed_mask = np.pad(fixed_mask, ((8, 8),(8, 8),(0, 0)), 'constant', constant_values=0)
        moving_mask = np.pad(moving_mask, ((8, 8),(8, 8),(0, 0)), 'constant', constant_values=0)

        
        inputs = [moving_image.squeeze(), fixed_image.squeeze(), moving_mask.squeeze(), fixed_mask.squeeze()]
        outputs = [fixed_image, zero_phi]

        yield inputs, outputs
