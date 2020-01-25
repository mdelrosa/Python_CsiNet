# NMSE_performance.py
from unpack_json import *
dataset_spec = get_dataset_spec(json_config)
batch_num = get_batch_num(json_config)
network_name = get_network_name(json_config)
minmax_file = get_minmax_file(json_config)
norm_range = get_norm_range(json_config)
lrs, batch_sizes = get_hyperparams(json_config)
T, dataset_spec, batch_num, lrs, batch_sizes, envir = get_keys_from_json(json_config, keys=['T', 'dataset_spec', 'batch_num', 'lrs', 'batch_sizes', 'envir'])
encoded_dims, dates, model_dir, aux_bool, M_1, data_format, epochs, t1_train, t2_train, gpu_num, lstm_latent_bool, conv_lstm_bool = unpack_json(json_config)
print(type(lrs),type(batch_sizes),type(encoded_dims))
if __name__=="__main__":
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";  # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gpu_num);  # Do other imports now...
    import tensorflow as tf
import data_tools as dt
import scipy.io as sio
import numpy as np
import math
import time
import sys
import csv

### load original normalized dataset (original)
def load_dataset(dataset_spec,T=3,batch_num=1,img_channels=2,img_height=32,img_width=32,data_format="channels_first"):
    x_train = x_train_up = x_val = x_val_up = None
    if dataset_spec:
        train_str = dataset_spec[0]
        val_str = dataset_spec[1]
    else:
        train_str = 'data/data_001/Data100_Htrainin_down_FDD_32ant'
        val_str = 'data/data_001/Data100_Hvalin_down_FDD_32ant'
    for batch in range(1,batch_num+1):
        print("Adding batch #{}".format(batch))
        # mat = sio.loadmat('data/data_001/Data100_Htrainin_down_FDD_32ant_{}.mat'.format(batch))
        mat = sio.loadmat(dt.batch_str(train_str,batch))
        x_train  = dt.add_batch(x_train, mat, 'train')
        mat = sio.loadmat(dt.batch_str(val_str,batch))
        x_val  = dt.add_batch(x_val, mat, 'val')
    x_test = x_val
    x_test_up = x_val_up
    
    # evaluate short T for speed of training and fair comparison with DualNet-TEMP
    print('pre x_train.shape: {}'.format(x_train.shape))
    x_train = dt.subsample_data(x_train,T)
    x_val = dt.subsample_data(x_val,T)
    x_test = dt.subsample_data(x_test,T)
    print('post x_train.shape: {}'.format(x_train.shape))
    
    # data are complex samples; split into re/im and concatenate
    
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    
    x_train = np.reshape(x_train, dt.get_data_shape(len(x_train), T, img_channels, img_height, img_width,data_format))
    x_val = np.reshape(x_val, dt.get_data_shape(len(x_val), T, img_channels, img_height, img_width,data_format))
    x_test = np.reshape(x_test, dt.get_data_shape(len(x_test), T, img_channels, img_height, img_width,data_format))
    print('x_train.shape: {} - x_val.shape: {} - x_test.shape: {}'.format(x_train.shape, x_val.shape, x_test.shape))

    aux_train = np.zeros((len(x_train),M_1))
    aux_test = np.zeros((len(x_test),M_1))
    d = {
            'train': x_train,
            'val': x_val,
            'test': x_test,
            'aux_train': aux_train,
            'aux_test': aux_test
        }
    return d

### load appropriate minmax file
def get_minmax():
    return minmax_file

### load model with weights: supersedes same func in unpack_json
def model_with_weights(model_dir,network_name,weights_bool=True):
    # load json and create model
    outfile = "{}/{}.json".format(model_dir,network_name)
    json_file = open(outfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    print('-> Loading model from {}...'.format(outfile))
    model = model_from_json(loaded_model_json)
    # load weights outto new model
    outfile = "{}/{}.h5".format(model_dir,network_name)
    print('-> Loading weights from {}...'.format(outfile))
    model.load_weights(outfile)
    return model

def predict_with_model(model,d,aux_bool=0):
    #Testing data
    tStart = time.time()
    print('-> Predicting on test set...')
    if aux_bool:
        x_hat = model.predict([d['aux_test'],d["test"]])
    else:
        x_hat = model.predict(d["test"])
    tEnd = time.time()
    print ("It cost %f sec per sample (%f samples)" % ((tEnd - tStart)/d["test"].shape[0],d["test"].shape[0]))
    return x_hat

### helper function: denormalize H3
def denorm_H3(data,minmax_file,link_type='down'):
    fieldnames = ['link','min','max']
    with open(minmax_file) as csv_file:
        csv_reader = csv.DictReader(csv_file,fieldnames=fieldnames,delimiter=',')
        for row in csv_reader:
            if row['link']==link_type:
                    d_min = float(row['min'])
                    d_max = float(row['max'])
    data = data*(d_max-d_min)+d_min
    return data

### helper function: denormalize H
def denorm_H4(data,minmax_file,link_type='down'):
    fieldnames = ['link','min','max']
    with open(minmax_file) as csv_file:
        csv_reader = csv.DictReader(csv_file,fieldnames=fieldnames,delimiter=',')
        for row in csv_reader:
            if row['link']==link_type:
                    d_min = float(row['min'])
                    d_max = float(row['max'])
    data = (data+1)/2*(d_max-d_min)+d_min
    return data

# calculate NMSE
def calc_NMSE(x_hat,x_test,T=3):
    # x_test_real = np.reshape(x_test[:, :, 0, :, :], (len(x_test), -1))
    # x_test_imag = np.reshape(x_test[:, :, 1, :, :], (len(x_test), -1))
    # x_test_C = x_test_real + 1j*x_test_imag
    # x_hat_real = np.reshape(x_hat[:, :, 0, :, :], (len(x_hat), -1))
    # x_hat_imag = np.reshape(x_hat[:, :, 1, :, :], (len(x_hat), -1))
    # x_hat_C = x_hat_real + 1j*x_hat_imagI
    if T == 1:
        x_test_temp =  np.reshape(x_test, (len(x_test), -1))
        x_hat_temp =  np.reshape(x_hat, (len(x_hat), -1))
    else:
        x_test_temp =  np.reshape(x_test[:, :, :, :, :], (len(x_test), -1))
        x_hat_temp =  np.reshape(x_hat[:, :, :, :, :], (len(x_hat), -1))
    # print("x_test_temp.shape: {}".format(x_test_temp.shape))
    # print("x_hat_temp.shape: {}".format(x_hat_temp.shape))
    power = np.sum(abs(x_test_temp)**2, axis=1)
    mse = np.sum(abs(x_test_temp-x_hat_temp)**2, axis=1)
    # print("# of zeros to remove in power: {}".format(len(power)-len(power[np.nonzero(power)])))
    mse = mse[np.nonzero(power)] 
    power = power[np.nonzero(power)] 
    # print('Overall power: {}'.format(power))
    # print('Overall mse: {}'.format(mse))
    temp = mse/power
    # print('Overall norm squared err: {}'.format(temp))
    # print('min: {} - max: {} - mean: {}'.format(np.min(temp),np.max(temp),np.mean(temp)))
    print("Overall NMSE is {}".format(10*math.log10(np.mean(temp))))
    if T != 1:
        for t in range(T):
            # x_test_real = np.reshape(x_test[:, t, 0, :, :], (len(x_test), -1))
            # x_test_imag = np.reshape(x_test[:, t, 1, :, :], (len(x_test), -1))
            # x_test_C = x_test_real + 1j*x_test_imag
            # x_hat_real = np.reshape(x_hat[:, t, 0, :, :], (len(x_hat), -1))
            # x_hat_imag = np.reshape(x_hat[:, t, 1, :, :], (len(x_hat), -1))
            # x_hat_C = x_hat_real + 1j*x_hat_imag
            # x_hat_F = np.reshape(x_hat_C, T, (len(x_hat_C), img_height, img_width)
            # X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)
            # X_hat = X_hat[:, :, 0:125]
    
            # n1 = np.sqrt(np.sum(np.conj(X_test)*X_test, axis=1))
            # n1 = n1.astype('float64')
            # n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1))
            # n2 = n2.astype('float64')
            # aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))
            # rho = np.mean(aa/(n1*n2), axis=1)
            # X_hat = np.reshape(X_hat, (len(X_hat), -1))
            # X_test = np.reshape(X_test, (len(X_test), -1))
            x_test_temp =  np.reshape(x_test[:, t, :, :, :], (len(x_test[:, t, :, :, :]), -1))
            x_hat_temp =  np.reshape(x_hat[:, t, :, :, :], (len(x_hat[:, t, :, :, :]), -1))
            power = np.sum(abs(x_test_temp)**2, axis=1)
            mse = np.sum(abs(x_test_temp-x_hat_temp)**2, axis=1)
            # print("# of zeros to remove in power: {}".format(len(power)-len(power[np.nonzero(power)])))
            mse = mse[np.nonzero(power)] 
            power = power[np.nonzero(power)] 
            # separate predictions based on timeslot
            # power_d = np.sum(abs(X_hat)**2, axis=1)
            # power_list.append(power)
            # mse_list.append(mse)
    
            # print('power at t{}: {}'.format(t+1,power))
            # print('mse at t{}: {}'.format(t+1,mse))
            temp = mse/power
            # print('min: {} - max: {} - mean: {}'.format(np.min(temp),np.max(temp),np.mean(temp)))
            print("NMSE at t{} is {}".format(t+1, 10*math.log10(np.mean(temp))))
            # print("Correlation is ", np.mean(rho))

if __name__=="__main__":
    d = load_dataset(dataset_spec,T=T,batch_num=batch_num)
    tests = len(encoded_dims)
    for i in range(tests):
        if encoded_dims[i] <= 3:
            # hack for looking at single test. make this more flexible in future
            full_name = '{}_D{}_{}_{:1.1e}_bs{}'.format(network_name,encoded_dims[i],dates[i],lrs[i],batch_sizes[i]) 
        else:
            full_name = '{}_dim{}_{}'.format(network_name,encoded_dims[i],dates[i])
        model = model_with_weights(model_dir,full_name)
        print("-"*128)
        print("Calculate NMSE for {}".format(full_name))
        print("-"*128)
        x_hat = predict_with_model(model,d,aux_bool=aux_bool)
        x_test = d['test']
        # print("Before de-normalizing data...")
        # print('-> x_hat range is from {} to {}'.format(np.min(x_hat),np.max(x_hat)))
        # print('-> x_test range is from {} to {} '.format(np.min(x_test),np.max(x_test)))
        # calc_NMSE(x_hat,x_test,T=T)
        print("De-normalizing data...")
        if norm_range == "norm_H3":
            x_hat = denorm_H3(x_hat,minmax_file)
            x_test = denorm_H3(x_test,minmax_file)
        if norm_range == "norm_H4":
            x_hat = denorm_H4(x_hat,minmax_file)
            x_test = denorm_H4(x_test,minmax_file)
        print('-> x_hat range is from {} to {}'.format(np.min(x_hat),np.max(x_hat)))
        print('-> x_test range is from {} to {} '.format(np.min(x_test),np.max(x_test)))
        calc_NMSE(x_hat,x_test,T=T)
