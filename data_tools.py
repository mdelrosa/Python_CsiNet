# data_tools.py
# functions for importing/manipulating data for training/validation
import numpy as np

def add_batch(data_down, batch, type_str):
    # concatenate batch data onto end of data
    # Inputs:
    # -> data_up = np.array for uplink
    # -> data_up = np.array for downlink
    # -> batch = mat file to add to np.array
    # -> type_str = part of key to select for training/validation
    x_down = batch['HD_{}'.format(type_str)]
    if data_down is None:
        return x_down
    else:
        return np.vstack((data_down,x_down))

def split_complex(data):
    re = np.expand_dims(np.real(data).astype('float32'),axis=2) # real portion
    im = np.expand_dims(np.imag(data).astype('float32'),axis=2) # imag portion
    return np.concatenate((re,im),axis=2)

def get_data_shape(samples,T,img_channels,img_height,img_width,data_format):
    if(data_format=="channels_last"):
        shape = (samples, T, img_height, img_width, img_channels) if T != 1 else (samples, img_height, img_width, img_channels)
    elif(data_format=="channels_first"):
        shape = (samples, T, img_channels, img_height, img_width) if T != 1 else (samples, img_channels, img_height, img_width)
    return shape

def subsample_data(data,T,T_max=10):
    # shorten along T axis
    if (T < T_max):
        data = data[:,0:T,:]
    return data

def batch_str(base,num):
        return base+'_'+str(num)+'.mat'

