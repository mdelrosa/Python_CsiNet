import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
import json
from CsiNet_LSTM import *
tf.reset_default_graph()

envir = 'indoor' #'indoor' or 'outdoor'
# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
T = 1
data_format = "channels_last"

json_config = 'config/csinet_aux_test_10_31.json'
with open(json_config) as json_file:
    data = json.load(json_file)
    encoded_dims = data['encoded_dims']
    dates = data['dates']
    model_dir = data['model_dir']
    aux_bool = data['aux_bool']
    M_1 = data['M_1']
    print("Loaded json file: {}".format(json_config))

# # Data loading
# if envir == 'indoor':
#     # mat = sio.loadmat('data/DATA_Htestin.mat')
#     mat = sio.loadmat('../Bi-Directional-Channel-Reciprocity/data/indoor53/Data100_Hvalin.mat')
#     X_test = x_test = mat['HD_val'] # array

# elif envir == 'outdoor':
#     mat = sio.loadmat('../Bi-Directional-Channel-Reciprocity/data/outdoor53/Data100_Hvalin.mat') # don't have this dataset at the moment
#     X_test = x_test = mat['HD_val'] # array

# x_test = x_test.astype('float32')
# x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

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

def get_data_shape(samples,img_channels,img_height,img_width,data_format):
    if(data_format=="channels_last"):
        shape = (samples, img_height, img_width, img_channels)
    elif(data_format=="channels_first"):
        shape = (samples, img_channels, img_height, img_width)
    return shape

def subsample_data(data,T,T_max=10):
    # shorten along T axis
    (samples, T_orig, img_len) = data.shape
    if (T < T_max):
        data = data[:,0:T,:]
        if T == 1:
            data = data.reshape(samples,img_len)
    return data

# Data loading
batch_num = 10 # we'll use batch_num-1 for training and 1 for validation
x_train = x_train_up = x_val = x_val_up = None
if envir == 'indoor':
    for batch in range(1,batch_num+1):
        print("Adding batch #{}".format(batch))
        mat = sio.loadmat('data/data_001/Data100_Htrainin_down_FDD_32ant_{}.mat'.format(batch))
        x_train  = add_batch(x_train, mat, 'train')
        mat = sio.loadmat('data/data_001/Data100_Hvalin_down_FDD_32ant_{}.mat'.format(batch))
        x_val  = add_batch(x_val, mat, 'val')

    x_test = x_val
    x_test_up = x_val_up

elif envir == 'outdoor':
    mat = sio.loadmat('../Bi-Directional-Channel-Reciprocity/data/urban3/Data100_Htrainin_down_FDD.mat')
    mat1 = sio.loadmat('../Bi-Directional-Channel-Reciprocity/data/urban3/Data100_Htrainin_up_FDD.mat')
    x_train = mat['HD_train']
    x_train_up = mat1['HU_train']
    mat = sio.loadmat('../Bi-Directional-Channel-Reciprocity/data/urban3/Data100_Hvalin_down_FDD.mat')
    mat1 = sio.loadmat('../Bi-Directional-Channel-Reciprocity/data/urban3/Data100_Hvalin_up_FDD.mat')
    x_val = mat['HD_val']
    x_val_up = mat1['HU_val']

    x_test = x_val
    x_test_up = x_val_up

# evaluate short T for speed of training and fair comparison with DualNet-TEMP
print('pre x_train.shape: {}'.format(x_train.shape))
x_train = subsample_data(x_train,T)
x_val = subsample_data(x_val,T)
x_test = subsample_data(x_test,T)
print('post x_train.shape: {}'.format(x_train.shape))

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

print('x_train.shape: {} - x_val.shape: {} - x_test.shape: {}'.format(x_train.shape, x_val.shape, x_test.shape))

# data_shape = (len(tensor), T, img_height, img_width, img_channels)

x_train = np.reshape(x_train, get_data_shape(len(x_train), img_channels, img_height, img_width,data_format))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, get_data_shape(len(x_val), img_channels, img_height, img_width,data_format))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, get_data_shape(len(x_test), img_channels, img_height, img_width,data_format))  # adapt this if using `channels_first` image data format

# Calcaulating the NMSE and rho
# if envir == 'indoor':
#     mat = sio.loadmat('data/DATA_HtestFin_all.mat')
#     X_test = mat['HF_all']# array

# elif envir == 'outdoor':
#     mat = sio.loadmat('data/DATA_HtestFout_all.mat')
#     X_test = mat['HF_all']# array

# get a dummy tensor for zeroed out aux input
aux_test = np.zeros((len(x_test),M_1))

x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
# encoded_dims = [512,128,64,32]  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
# dates = ['10_14','10_14','10_14','10_14']
power_arr = []
mse_arr = []
# TO-DO; load these params from json file
for i in range(len(encoded_dims)):
    encoded_dim = encoded_dims[i]
    date = dates[i]
    file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+'_'+date

    # load json and create model
    outfile = "{}/model_{}.json".format(model_dir,file)
    json_file = open(outfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    CsiNet_model = model_from_json(loaded_model_json)

    # load weights outto new model
    outfile = "{}/model_{}.h5".format(model_dir,file)
    CsiNet_model.load_weights(outfile)
    
    print('loading model from {} with encoded_dim={}'.format(outfile,encoded_dim))
    #Testing data
    tStart = time.time()
    if aux_bool == 1:
        x_hat = CsiNet_model.predict([aux_test,x_test])
    else:
        x_hat = CsiNet_model.predict(x_test)
    tEnd = time.time()
    print ("It cost %f sec per sample (%f samples)" % ((tEnd - tStart)/x_test.shape[0],x_test.shape[0]))

    # X_test = np.reshape(X_test, (len(X_test), img_height, 125))
    # print("In "+envir+" environment")
    power_list = []
    mse_list = []
    # for t in range(T):
    x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)
    power = np.sum(abs(x_test_C)**2, axis=1)
    # power_d = np.sum(abs(X_hat)**2, axis=1)
    mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)
    power_list.append(power)
    mse_list.append(mse)

    print("-> NMSE for encoded_dim={} is {}".format(encoded_dim, 10*math.log10(np.mean(mse/power))))
    # print("Correlation is ", np.mean(rho))
    power_arr.append(power_list)
    mse_arr.append(mse_list)
    # TO-DO: make dataframe with models and timeslots as rows/cols

power_arr = np.array((power_arr))
mse_arr = np.array((mse_arr))
print('power_arr:{}'.format(power_arr))
print('mse_arr:{}'.format(mse_arr))

#filename = "result/decoded_%s.csv"%file
#x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
#np.savetxt(filename, x_hat1, delimiter=",")

# import matplotlib.pyplot as plt
# '''abs'''
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display origoutal
#     ax = plt.subplot(2, n, i + 1 )
#     x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
#     plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ax.invert_yaxis()
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5 
#                           + 1j*(x_hat[i, 1, :, :]-0.5))
#     plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ax.invert_yaxis()
# plt.show()
