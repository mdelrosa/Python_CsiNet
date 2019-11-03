import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time

from CsiNet import *
tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)

envir = 'indoor' #'indoor' or 'outdoor'
# fit params
epochs = 1000

# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim = 512  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
M_1 = 512 # aux layer compression dimension for temporal correlation

data_format = "channels_last"

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
batch_num = 10
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
x_train = subsample_data(x_train,1)
x_val = subsample_data(x_val,1)
x_test = subsample_data(x_test,1)
print('subsample x_train.shape: {}'.format(x_train.shape))

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

# get a dummy tensor for zeroed out aux input
aux_train = np.zeros((len(x_train),M_1))
aux_test = np.zeros((len(x_test),M_1))
print('final aux_train.shape: {}'.format(aux_train.shape))
print('final aux_test.shape: {}'.format(aux_test.shape))

x_train = np.reshape(x_train, get_data_shape(len(x_train), img_channels, img_height, img_width, data_format))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, get_data_shape(len(x_val), img_channels, img_height, img_width, data_format))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, get_data_shape(len(x_test), img_channels, img_height, img_width, data_format))  # adapt this if using `channels_first` image data format
print('final x_train.shape: {}'.format(x_train.shape))
print('final x_val.shape: {}'.format(x_val.shape))


encoded_dims = [512, 128, 64, 32]
aux = Input((M_1,))
for encoded_dim in encoded_dims:
        print('-------------------------------------') 
        print('Build and train CsiNet_aux for encoded_dim={}'.format(encoded_dim))
        print('-------------------------------------') 
        # build CsiNet-aux
        autoencoder, encoded = CsiNet(img_channels, img_height, img_width, encoded_dim,aux=aux) # CSINet with M_1 dimensional latent space
        autoencoder = Model(inputs=autoencoder.inputs,outputs=autoencoder.outputs[0]) # remove "encoded" in output for training
        autoencoder.compile(optimizer='adam', loss='mse')
        print(autoencoder.summary())

        class LossHistory(Callback):
                def on_train_begin(self, logs={}):
                        self.losses_train = []
                        self.losses_val = []

                def on_batch_end(self, batch, logs={}):
                        self.losses_train.append(logs.get('loss'))
                        
                def on_epoch_end(self, epoch, logs={}):
                        self.losses_val.append(logs.get('val_loss'))
                        

        history = LossHistory()
        file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+time.strftime('_%m_%d')
        path = 'result/TensorBoard_%s' %file

        data_train = [aux_train, x_train]
        data_test = [aux_test, x_test]

        autoencoder.fit(data_train, x_train,
                                        epochs=1000,
                                        batch_size=200,
                                        shuffle=True,
                                        validation_data=(data_test, x_test),
                                        callbacks=[history,
                                                           TensorBoard(log_dir = path)])

        filename = 'result/trainloss_%s.csv'%file
        loss_history = np.array(history.losses_train)
        np.savetxt(filename, loss_history, delimiter=",")

        filename = 'result/valloss_%s.csv'%file
        loss_history = np.array(history.losses_val)
        np.savetxt(filename, loss_history, delimiter=",")

        # save+serialize model to JSON
        model_json = autoencoder.to_json()
        outfile = "result/model_%s.json"%file
        with open(outfile, "w") as json_file:
                json_file.write(model_json)
        # serialize weights to HDF5
        outfile = "result/model_%s.h5"%file
        autoencoder.save_weights(outfile)

        # #Testing data
        # tStart = time.time()
        # x_hat = autoencoder.predict(data_train)
        # tEnd = time.time()
        # print ("It cost %f sec" % ((tEnd - tStart)/data_train[0].shape[0]))

# Calcaulating the NMSE and rho
# if envir == 'indoor':
#     mat = sio.loadmat('data/DATA_HtestFin_all.mat')
#     X_test = mat['HF_all']# array

# elif envir == 'outdoor':
#     mat = sio.loadmat('data/DATA_HtestFout_all.mat')
#     X_test = mat['HF_all']# array

# x_test = np.reshape(X_test, (len(X_test), img_height, 125))
# x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
# x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
# x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
# x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
# x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
# x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)
# x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
# X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)
# X_hat = X_hat[:, :, 0:125]
# 
# n1 = np.sqrt(np.sum(np.conj(X_test)*X_test, axis=1))
# n1 = n1.astype('float64')
# n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1))
# n2 = n2.astype('float64')
# aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))
# rho = np.mean(aa/(n1*n2), axis=1)
# X_hat = np.reshape(X_hat, (len(X_hat), -1))
# X_test = np.reshape(X_test, (len(X_test), -1))
# power = np.sum(abs(x_test_C)**2, axis=1)
# power_d = np.sum(abs(X_hat)**2, axis=1)
# mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)
# 
# print("In "+envir+" environment")
# print("When dimension is", encoded_dim)
# print("NMSE is ", 10*math.log10(np.mean(mse/power)))
# print("Correlation is ", np.mean(rho))
# filename = "result/decoded_%s.csv"%file
# x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
# np.savetxt(filename, x_hat1, delimiter=",")
# filename = "result/rho_%s.csv"%file
# np.savetxt(filename, rho, delimiter=",")
# 
# 
# import matplotlib.pyplot as plt
# '''abs'''
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
# 	# display origoutal
# 	ax = plt.subplot(2, n, i + 1 )
# 	x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
# 	plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
# 	plt.gray()
# 	ax.get_xaxis().set_visible(False)
# 	ax.get_yaxis().set_visible(False)
# 	ax.invert_yaxis()
# 	# display reconstruction
# 	ax = plt.subplot(2, n, i + 1 + n)
# 	decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5 
# 						  + 1j*(x_hat[i, 1, :, :]-0.5))
# 	plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
# 	plt.gray()
# 	ax.get_xaxis().set_visible(False)
# 	ax.get_yaxis().set_visible(False)
# 	ax.invert_yaxis()
# plt.show()
