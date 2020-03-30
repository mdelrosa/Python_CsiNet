# CsiNet_LSTM_train.py

from NMSE_performance import calc_NMSE, denorm_H3, denorm_H4
from unpack_json import *
# json_config = 'config/indoor0001/T10/csinet_lstm_v2_02_22.json' # VALIDATED 
# json_config = 'config/outdoor300/T10/csinet_lstm_v2_03_02.json' # VALIDATED 
# json_config = 'config/outdoor300/T5/csinet_lstm_v2_03_17.json' # VALIDATED 
json_config = 'config/indoor0001/T10/replication/csinet_lstm_v2_03_28.json' # IN PROGRESS
encoded_dims, dates, model_dir, aux_bool, M_1, data_format, epochs, t1_train, t2_train, gpu_num, lstm_latent_bool, conv_lstm_bool = unpack_json(json_config)
network_name, norm_range, minmax_file, share_bool, T, dataset_spec, batch_num, lrs, batch_sizes, envir = get_keys_from_json(json_config, keys=['network_name', 'norm_range', 'minmax_file', 'share_bool', 'T', 'dataset_spec', 'batch_num', 'lrs', 'batch_sizes', 'envir'])
load_bool, pass_through_bool, t1_train, t2_train = get_keys_from_json(json_config, keys=['load_bool', 'pass_through_bool', 't1_train', 't2_train'],is_bool=True) # import these as booleans rather than int
lr = lrs[0]
batch_size = batch_sizes[0]

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";  # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gpu_num);  # Do other imports now...
import scipy.io as sio 
import numpy as np
import math
import time
import sys
# import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, Callback, ModelCheckpoint
from tensorflow.core.protobuf import rewriter_config_pb2
from CsiNet_LSTM import *
# tf.reset_default_graph()
# from tensorflow.keras.backend import manual_variable_initialization
# manual_variable_initialization(True)

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
    # tf.global_variables_initializer()

reset_keras()

# # limit gpu resource allocation
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
# 
# # disable arithmetic optimizer
# off = rewriter_config_pb2.RewriterConfig.OFF
# config.graph_options.rewrite_options.arithmetic_optimization = off
# 
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)

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
def default_vals():
    # codeword lengths: CR=1/4 -> 512, CR=1/16 -> 128, CR=1/32 -> 64, CR=1/64 -> 32
    return 0, 1 # T, M_2, LSTM_depth, convCsiNet_bool, debug_flag

if len(sys.argv):
    try:
        convCsiNet_bool = int(sys.argv[1])
        debug_flag = int(sys.argv[2])
    except:
        convCsiNet_bool, debug_flag = default_vals()
else:
    convCsiNet_bool, debug_flag = default_vals()
print("T: {} - convCsiNet_bool: {}- debug_flag: {}".format(T, convCsiNet_bool, debug_flag))

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

# Data loading
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
# x_train = split_complex(x_train)
# x_val = split_complex(x_val)
# x_test = split_complex(x_test)

print('x_train.shape: {} - x_val.shape: {} - x_test.shape: {}'.format(x_train.shape, x_val.shape, x_test.shape))

# data_shape = (len(tensor), T, img_height, img_width, img_channels)

x_train = np.reshape(x_train, get_data_shape(len(x_train), T, img_channels, img_height, img_width,data_format))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, get_data_shape(len(x_val), T, img_channels, img_height, img_width,data_format))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, get_data_shape(len(x_test), T, img_channels, img_height, img_width,data_format))  # adapt this if using `channels_first` image data format

aux_train = np.zeros((len(x_train),M_1))
aux_val = np.zeros((len(x_val),M_1))
aux_test = np.zeros((len(x_test),M_1))

# CRs = [128,64,32] # sweep compression ratios for latent space
for i in range(len(encoded_dims)):
    M_2 = encoded_dims[i]
    date = dates[i]
    reset_keras()
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    # optimizer = Adam(learning_rate=lr)
    print('-------------------------------------')
    print("Build CsiNet-LSTM for CR2={}".format(M_2))
    print('-------------------------------------')
    LSTM_depth=3
    
    # def callbacks
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses_train = []
            self.losses_val = []
    
        def on_batch_end(self, batch, logs={}):
            self.losses_train.append(logs.get('loss'))
            
        def on_epoch_end(self, epoch, logs={}):
            self.losses_val.append(logs.get('val_loss'))
    
    if aux_bool:
        data_train = [aux_train, x_train]
        data_val = [aux_val, x_test]
        data_test = [aux_test, x_test]
    else:
        data_train = x_train
        data_test = x_test
    print('-> model_dir: {}'.format(model_dir))
    if convCsiNet_bool:
        file_base = 'CsiNet_ConvLSTM_'
    else:
        file_base = 'CsiNet_LSTM_'

    if convCsiNet_bool:
        CsiNet_LSTM_model = CsiNet_ConvLSTM(img_channels, img_height, img_width, T, M_1, M_2, LSTM_depth=LSTM_depth, data_format=data_format)
    else:
        if load_bool:
            if (network_name != 'model_weights_test'):
                file = file_base+(envir)+'_dim'+str(M_2)+"_{}".format(date)
            else:
                file = "weights_test" 
            CsiNet_LSTM_model = CsiNet_LSTM(img_channels, img_height, img_width, T, M_1, M_2, LSTM_depth=LSTM_depth, data_format=data_format, t1_trainable=t1_train, t2_trainable=t2_train, share_bool=share_bool, pass_through_bool=pass_through_bool, conv_lstm_bool=conv_lstm_bool)
            outfile = "{}/model_{}.h5".format(model_dir,file)
            CsiNet_LSTM_model.load_weights(outfile)
            temp = CsiNet_LSTM_model.get_weights()
            print(temp[0])
            print ("--- Pre-loaded network performance is... ---")
            x_hat = CsiNet_LSTM_model.predict(data_test)

            print("For Adam with lr={:1.1e} // batch_size={} // norm_range={}".format(lr,batch_size,norm_range))
            if norm_range == "norm_H3":
                x_hat_denorm = denorm_H3(x_hat,minmax_file)
                x_test_denorm = denorm_H3(x_test,minmax_file)
            if norm_range == "norm_H4":
                x_hat_denorm = denorm_H4(x_hat,minmax_file)
                x_test_denorm = denorm_H4(x_test,minmax_file)
            print('-> x_hat range is from {} to {}'.format(np.min(x_hat_denorm),np.max(x_hat_denorm)))
            print('-> x_test range is from {} to {} '.format(np.min(x_test_denorm),np.max(x_test_denorm)))
            calc_NMSE(x_hat_denorm,x_test_denorm,T=T)
        else:
            CsiNet_LSTM_model = CsiNet_LSTM(img_channels, img_height, img_width, T, M_1, M_2, LSTM_depth=LSTM_depth, data_format=data_format, t1_trainable=t1_train, t2_trainable=t2_train, share_bool=share_bool, pass_through_bool=pass_through_bool, conv_lstm_bool=conv_lstm_bool)
            CsiNet_LSTM_model.compile(optimizer=optimizer, loss='mse')

        if (network_name != 'model_weights_test'):
            file = file_base+(envir)+'_dim'+str(M_2)+"_{}".format(date)
        else:
            file = "weights_test" 
        # save+serialize model to JSON
        model_json = CsiNet_LSTM_model.to_json()
        outfile = "{}/model_{}.json".format(model_dir,file)
        with open(outfile, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        outfile = "{}/model_{}.h5".format(model_dir,file)
        # outfile = "{}/weights_test.h5".format(model_dir,file)
        checkpoint = ModelCheckpoint(outfile, monitor="val_loss",verbose=1)
        # checkpoint = ModelCheckpoint(outfile, monitor="val_loss",verbose=1,save_best_only=True,mode="min")
        history = LossHistory()
        # path = '{}/TensorBoard_{}'.format(model_dir, file)
        
        # CsiNet_LSTM_model.compile(optimizer=optimizer, loss='mse')
        # print(CsiNet_LSTM_model)
        
        CsiNet_LSTM_model.fit(data_train, x_train,
                         epochs=epochs,
                         batch_size=batch_size,
                         shuffle=True,
                         validation_data=(data_val, x_val),
                         callbacks=[checkpoint,
                                    history])
                                    # TensorBoard(log_dir = path),
        
        
        filename = '{}/trainloss_{}.csv'.format(model_dir,file)
        loss_history = np.array(history.losses_train)
        np.savetxt(filename, loss_history, delimiter=",")
        
        filename = '{}/valloss_{}.csv'.format(model_dir,file)
        loss_history = np.array(history.losses_val)
        np.savetxt(filename, loss_history, delimiter=",")
        
        #Testing data
        tStart = time.time()
        # x_hat = CsiNet_LSTM_model.predict(data_test)
        x_hat = CsiNet_LSTM_model.predict(data_test)
        tEnd = time.time()
        print ("It cost %f sec per sample (%f samples)" % ((tEnd - tStart)/x_test.shape[0],x_test.shape[0]))
        
        print("For Adam with lr={:1.1e} // batch_size={} // norm_range={}".format(lr,batch_size,norm_range))
        if norm_range == "norm_H3":
            x_hat_denorm = denorm_H3(x_hat,minmax_file)
            x_test_denorm = denorm_H3(x_test,minmax_file)
        elif norm_range == "norm_H4":
            x_hat_denorm = denorm_H4(x_hat,minmax_file)
            x_test_denorm = denorm_H4(x_test,minmax_file)
        print('-> x_hat range is from {} to {}'.format(np.min(x_hat_denorm),np.max(x_hat_denorm)))
        print('-> x_test range is from {} to {} '.format(np.min(x_test_denorm),np.max(x_test_denorm)))
        calc_NMSE(x_hat_denorm,x_test_denorm,T=T)
