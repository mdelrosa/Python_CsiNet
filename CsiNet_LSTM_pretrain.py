# CsiNet_LSTM_pretrain.py
from unpack_json import *
json_config = 'config/indoor0001/T10/replication/csinet_lstm_pretrain_04_04.json' 

encoded_dims, dates, model_dir, aux_bool, M_1, data_format, epochs, t1_train, t2_train, gpu_num, lstm_latent_bool, conv_lstm_bool = unpack_json(json_config)
norm_range, minmax_file, T, dataset_spec, batch_num, lrs, batch_sizes, envir = get_keys_from_json(json_config, keys=['norm_range', 'minmax_file', 'T', 'dataset_spec', 'batch_num', 'lrs', 'batch_sizes', 'envir'])
t1_train, t2_train = get_keys_from_json(json_config, keys=['t1_train', 't2_train'],is_bool=True) # import these as booleans rather than int

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";  # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gpu_num);  # Do other imports now...
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
# from tensorflow.keras import Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
import sys
# import os
from CsiNet_LSTM import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras import initializers 
# tf.reset_default_graph()

from tensorflow.core.protobuf import rewriter_config_pb2
from NMSE_performance import calc_NMSE, denorm_H3, denorm_H4

# minmax_file = get_minmax()
# norm_range = get_norm_range(json_config)

def reset_keras():
    sess = tf.keras.backend.get_session()
    tf.keras.backend.clear_session()
    sess.close()
    # limit gpu resource allocation
    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.visible_device_list = '1'
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    
    # disable arithmetic optimizer
    off = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.rewrite_options.arithmetic_optimization = off
    
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

reset_keras()

# envir = 'indoor' #'indoor' or 'outdoor'
# fit params
# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim = 512  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
# snippet testing CSINet_LSTM
debug_flag = 0

# early stopping callback
# es = EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=1)

def default_vals():
    # codeword lengths: CR=1/4 -> 512, CR=1/16 -> 128, CR=1/32 -> 64, CR=1/64 -> 32
    return 10, 1 # T, LSTM_depth_lim, debug_flag

if len(sys.argv):
    try:
        T = int(sys.argv[1])
        debug_flag = int(sys.argv[2])
    except:
        T, debug_flag = default_vals()
else:
    T, debug_flag = default_vals()
print("T: {} - debug_flag: {}".format(T, debug_flag))
epochs = 10 if debug_flag else epochs
batch_num = batch_num if debug_flag==0 else 1 # we'll use batch_num-1 for training and 1 for validation

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
        shape = (samples, T, img_height, img_width, img_channels)
    elif(data_format=="channels_first"):
        shape = (samples, T, img_channels, img_height, img_width)
    return shape

def subsample_data(data,T,T_max=10):
    # shorten along T axis
    if (T < T_max):
        data = data[:,0:T,:]
    return data

def batch_str(base,num):
        return base+'_'+str(batch)+'.mat'

# data loading
batch_num = batch_num if debug_flag == 0 else 1 # we'll use batch_num-1 for training and 1 for validation
epochs = 10 if debug_flag else epochs
x_train = x_train_up = x_val = x_val_up = x_test = x_test_up = None
if dataset_spec:
    train_str = dataset_spec[0]
    val_str = dataset_spec[1]
    if len(dataset_spec) ==3:
        test_str = dataset_spec[2]
else:
    train_str = 'data/data_001/Data100_Htrainin_down_FDD_32ant'
    val_str = 'data/data_001/Data100_Hvalin_down_FDD_32ant'
for batch in range(1,batch_num+1):
    print("Adding batch #{}".format(batch))
    # mat = sio.loadmat('data/data_001/Data100_Htrainin_down_FDD_32ant_{}.mat'.format(batch))
    mat = sio.loadmat(batch_str(train_str,batch))
    x_train  = add_batch(x_train, mat, 'train')
    mat = sio.loadmat(batch_str(val_str,batch))
    x_val  = add_batch(x_val, mat, 'val')
    if len(dataset_spec) ==3:
        print("test_str: {}".format(test_str))
        mat = sio.loadmat(batch_str(test_str,batch))
        print("test matr: {}".format(mat))
        x_test  = add_batch(x_test, mat, 'test')
if len(dataset_spec) < 3:
    x_test = x_val
    x_test_up = x_val_up

# evaluate short T for speed of training and fair comparison with DualNet-TEMP
print('pre x_train.shape: {}'.format(x_train.shape))
x_train = subsample_data(x_train,T)
x_val = subsample_data(x_val,T)
x_test = subsample_data(x_test,T)
print('post x_train.shape: {}'.format(x_train.shape))

# data are complex samples; split into re/im and concatenate

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

print('x_train.shape: {} - x_val.shape: {} - x_test.shape: {}'.format(x_train.shape, x_val.shape, x_test.shape))

# data_shape = (len(tensor), T, img_height, img_width, img_channels)

x_train = np.reshape(x_train, get_data_shape(len(x_train), T, img_channels, img_height, img_width,data_format))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, get_data_shape(len(x_val), T, img_channels, img_height, img_width,data_format))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, get_data_shape(len(x_test), T, img_channels, img_height, img_width,data_format))  # adapt this if using `channels_first` image data format

aux_train = np.zeros((len(x_train),M_1))
aux_val = np.zeros((len(x_val),M_1))
aux_test = np.zeros((len(x_test),M_1))

for LSTM_depth in encoded_dims:
    for lr in lrs:
        reset_keras()
        optimizer = Adam(learning_rate=lr)
        for batch_size in batch_sizes:
            print('=====================================')
            print("LSTM Depth={} // Adam with lr={:1.1e} // batch_size={}".format(LSTM_depth,lr,batch_size))
            print('=====================================')
            if conv_lstm_bool: 
                file = 'convLSTM_'+(envir)+'_T{}'.format(T)+'_depth{}'.format(LSTM_depth)+time.strftime('_%m_%d')
                print("Convolutional recurrent activations.")
                LSTM_model = stacked_convLSTM(img_channels, img_height, img_width, T, lstm_latent_bool,LSTM_depth=LSTM_depth, data_format=data_format)
            else:
                file = 'LSTM_'+(envir)+'_T{}'.format(T)+'_depth{}'.format(LSTM_depth)+time.strftime('_%m_%d')
                print("Non-convolutional recurrent activations.")
                # LSTM_model = stacked_LSTM(img_channels, img_height, img_width, T, lstm_latent_bool,LSTM_depth=LSTM_depth, data_format=data_format)
                # LSTM_model = stacked_LSTM(img_channels, img_height, img_width, T, lstm_latent_bool,LSTM_depth=LSTM_depth, data_format=data_format,recurrent_initializer=initializers.Identity(gain=1.0),kernel_initializer=initializers.Identity(gain=1.0))
                LSTM_model = stacked_LSTM(img_channels, img_height, img_width, T, lstm_latent_bool,LSTM_depth=LSTM_depth, data_format=data_format)
            LSTM_model.compile(optimizer=optimizer, loss='mse')
            
            class LossHistory(Callback):
                def on_train_begin(self, logs={}):
                    self.losses_train = []
                    self.losses_val = []
            
                def on_batch_end(self, batch, logs={}):
                    self.losses_train.append(logs.get('loss'))
                    
                def on_epoch_end(self, epoch, logs={}):
                    self.losses_val.append(logs.get('val_loss'))
                    
            history = LossHistory()
            if conv_lstm_bool:
                file = 'CsiNet_convLSTM_{}_D{}_{}'.format(envir,LSTM_depth,time.strftime('%m_%d'))
            else:
                file = 'CsiNet_LSTM_{}_D{}_{}'.format(envir,LSTM_depth,time.strftime('%m_%d'))
            path = '{}/TensorBoard_{}_{:1.1e}_bs{}'.format(model_dir, file, lr, batch_size)
            
            model_json = LSTM_model.to_json()
            outfile = "{}/model_{}_{:1.1e}_bs{}.json".format(model_dir,file, lr, batch_size)
            with open(outfile, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            outfile = "{}/model_{}_{:1.1e}_bs{}.h5".format(model_dir,file, lr, batch_size)
            checkpoint = ModelCheckpoint(outfile, monitor="val_loss",verbose=1)
    
            LSTM_model.fit(x_train, x_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(x_val, x_val),
                            callbacks=[history, checkpoint, 
                                       TensorBoard(log_dir = path)])
           
            filename = '{}/trainloss_{}_{:1.1e}_bs{}.csv'.format(model_dir,file, lr, batch_size)
            loss_history = np.array(history.losses_train)
            np.savetxt(filename, loss_history, delimiter=",")
            
            filename = '{}/valloss_{}_{:1.1e}_bs{}.csv'.format(model_dir,file, lr, batch_size)
            loss_history = np.array(history.losses_val)
            np.savetxt(filename, loss_history, delimiter=",")
            
            #Testing data
            print("reload best weights...".format(outfile))
            LSTM_model.load_weights(outfile)
            tStart = time.time()
            x_hat = LSTM_model.predict(x_test)
            tEnd = time.time()
            print ("It cost %f sec per sample (%f samples)" % ((tEnd - tStart)/x_test.shape[0],x_test.shape[0]))
            
            print("For Adam with lr={:1.1e} // batch_size={} // norm_range={}".format(lr,batch_size,norm_range))
            if norm_range == "norm_H3":
                x_hat_denorm = denorm_H3(x_hat,minmax_file)
                x_test_denorm = denorm_H3(x_test,minmax_file)
            if norm_range == "norm_H4":
                x_hat_denorm = denorm_H4(x_hat,minmax_file)
                x_test_denorm = denorm_H4(x_test,minmax_file)
            calc_NMSE(x_hat_denorm,x_test_denorm,T=T)
