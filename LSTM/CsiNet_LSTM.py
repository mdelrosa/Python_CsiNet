# CsiNet_LSTM.py

import tensorflow as tf
from keras.layers import Input, concatenate, Lambda, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, LSTM
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
from keras.backend import slice
import scipy.io as sio 
import numpy as np
import math
import time
from CsiNet import *
tf.reset_default_graph()

envir = 'indoor' #'indoor' or 'outdoor'
# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim = 512  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
T = 10

def CsiNet_LSTM(img_channels, img_height, img_width, T, M_1, M_2):

	# base CSINet models
	CsiNet_hi, encoded = CsiNet(img_channels, img_height, img_width, M_1) # CSINet with M_1 dimensional latent space
	aux = Input((M_1,))
	CsiNet_lo, _ = CsiNet(img_channels, img_height, img_width, M_2, aux=aux) # CSINet with M_2+M_1 dimensional latent space

	print("--- High Dimensional (M_1) Latent Space CsiNet ---")
	CsiNet_hi.summary()
	print("--- Lower Dimensional (M_2) Latent Space CsiNet ---")
	CsiNet_lo.summary()
	# TO-DO: load weights in hi/lo models

	# TO-DO: split large input tensor to use as inputs to 1:T CSINets
	x = Input((img_channels, img_height, img_width, T))
	CsiOut = []
	for i in range(T):
		CsiIn = Lambda( lambda x: x[:,:,:,:,i])(x)
		if i == 0:
			# use CsiNet_hi for t=1
			[OutLayer, EncodedLayer] = CsiNet_hi(CsiIn)
			print('#{} - EncodedLayer: {}'.format(i, EncodedLayer))
		else:
			# use CsiNet_lo for t in [2:T]
			# TO-DO: make sure M_1 codeword from CSINet_hi is an aux input to each CSINet_lo
			OutLayer = CsiNet_lo([EncodedLayer, CsiIn])
		print('#{} - OutLayer: {}'.format(i, OutLayer))
		CsiOut.append(OutLayer)
	
	# TO-DO: apply concatenated CSINet decoder outputs into unrolled LSTM
	LSTM_model = stacked_LSTM(img_channels, img_height, img_width, T)
	LSTM_model.compile(optimizer='adam', loss='mse')
	print(LSTM_model.summary())

	LSTM_in = concatenate(CsiOut)
	LSTM_out = LSTM_model(LSTM_in)

	full_model = Model(inputs=[x], outputs=[LSTM_out])
	full_model.compile(optimizer='adam', loss='mse')
	full_model.summary()
	# TO-DO: compile full model with large 4D tensor as input and LSTM 4D tensor as output
	return None # for now
	

def stacked_LSTM(img_channels, img_height, img_width, T, LSTM_depth=3):
	# assume entire time-series of CSI from 1:T is concatenated
	LSTM_dim = img_channels*img_height*img_width
	orig_shape = (img_height, img_width, img_channels, T)
	x = Input(shape=orig_shape)
	recurrent_out = Reshape((T,LSTM_dim))(x)
	for i in range(LSTM_depth):
		recurrent_out = LSTM(LSTM_dim, return_sequences=True)(recurrent_out)
	out = Reshape(orig_shape)(recurrent_out)
	LSTM_model = Model(input=[x], output=[out])
	return LSTM_model

# snippet testing CSINet_LSTM
T = 10
# codeword lengths: CR=1/4 -> 512, CR=1/16 -> 128, CR=1/32 -> 64, CR=1/64 -> 32
M_1 = 512
M_2 = 128
CsiNet_LSTM = CsiNet_LSTM(img_channels, img_height, img_width, T, M_1, M_2)

# split tensor with lambda layer and keras backend 'split' method
# x = Input((img_height, img_width, img_channels, T))
# i=3
# y = Lambda( lambda x: x[:,:,:,:,i:i+1])(x) # https://github.com/keras-team/keras/issues/890
# model = Model(inputs=x,outputs=y)
# model.compile(optimizer='adam', loss='mse')
# model.summary()

# # Data loading
# if envir == 'indoor':
#     mat = sio.loadmat('data/DATA_Htrainin.mat') 
#     x_train = mat['HT'] # array
#     mat = sio.loadmat('data/DATA_Hvalin.mat')
#     x_val = mat['HT'] # array
#     mat = sio.loadmat('data/DATA_Htestin.mat')
#     x_test = mat['HT'] # array

# elif envir == 'outdoor':
#     mat = sio.loadmat('data/DATA_Htrainout.mat') 
#     x_train = mat['HT'] # array
#     mat = sio.loadmat('data/DATA_Hvalout.mat')
#     x_val = mat['HT'] # array
#     mat = sio.loadmat('data/DATA_Htestout.mat')
#     x_test = mat['HT'] # array

# x_train = x_train.astype('float32')
# x_val = x_val.astype('float32')
# x_test = x_test.astype('float32')
# x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
# test split layer
# x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
# x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

# class LossHistory(Callback):
#     def on_train_begin(self, logs={}):
#         self.losses_train = []
#         self.losses_val = []

#     def on_batch_end(self, batch, logs={}):
#         self.losses_train.append(logs.get('loss'))
        
#     def on_epoch_end(self, epoch, logs={}):
#         self.losses_val.append(logs.get('val_loss'))
        

# history = LossHistory()
# file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+time.strftime('_%m_%d')
# path = 'result/TensorBoard_%s' %file

# autoencoder.fit(x_train, x_train,
#                 epochs=1000,
#                 batch_size=200,
#                 shuffle=True,
#                 validation_data=(x_val, x_val),
#                 callbacks=[history,
#                            TensorBoard(log_dir = path)])

# filename = 'result/trainloss_%s.csv'%file
# loss_history = np.array(history.losses_train)
# np.savetxt(filename, loss_history, delimiter=",")

# filename = 'result/valloss_%s.csv'%file
# loss_history = np.array(history.losses_val)
# np.savetxt(filename, loss_history, delimiter=",")

# #Testing data
# tStart = time.time()
# x_hat = autoencoder.predict(x_test)
# tEnd = time.time()
# print ("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))

# # Calcaulating the NMSE and rho
# if envir == 'indoor':
#     mat = sio.loadmat('data/DATA_HtestFin_all.mat')
#     X_test = mat['HF_all']# array

# elif envir == 'outdoor':
#     mat = sio.loadmat('data/DATA_HtestFout_all.mat')
#     X_test = mat['HF_all']# array

# X_test = np.reshape(X_test, (len(X_test), img_height, 125))
# x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
# x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
# x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
# x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
# x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
# x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)
# x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
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
# power = np.sum(abs(x_test_C)**2, axis=1)
# power_d = np.sum(abs(X_hat)**2, axis=1)
# mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)

# print("In "+envir+" environment")
# print("When dimension is", encoded_dim)
# print("NMSE is ", 10*math.log10(np.mean(mse/power)))
# print("Correlation is ", np.mean(rho))
# filename = "result/decoded_%s.csv"%file
# x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
# np.savetxt(filename, x_hat1, delimiter=",")
# filename = "result/rho_%s.csv"%file
# np.savetxt(filename, rho, delimiter=",")


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


# # save
# # serialize model to JSON
# model_json = autoencoder.to_json()
# outfile = "result/model_%s.json"%file
# with open(outfile, "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# outfile = "result/model_%s.h5"%file
# autoencoder.save_weights(outfile)