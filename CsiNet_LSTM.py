# CsiNet_LSTM.py

import tensorflow as tf
from tensorflow.keras.layers import concatenate, Lambda, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, LSTM
from tensorflow.keras import Input
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.utils import plot_model
import scipy.io as sio 
import numpy as np
import math
import time
from CsiNet import *
# tf.reset_default_graph()
# tf.enable_eager_execution()

envir = 'indoor' #'indoor' or 'outdoor'
# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim = 512  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
T = 3
pre_t1_bool = True

def get_file(envir,encoded_dim,train_date):
	file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+'_'+train_date
	return "result/model_%s.h5"%file

def CsiNet_LSTM(img_channels, img_height, img_width, T, M_1, M_2, data_format='channels_first', pre_t1_bool=True, pre_t2_bool=True):
	# base CSINet models
	if pre_t1_bool:
		date = "10_14"
		file = 'CsiNet_'+(envir)+'_dim'+str(M_1)+'_'+date
		outfile = "result/model_%s.json"%file
		json_file = open(outfile)
		loaded_model_json = json_file.read()
		json_file.close()
		CsiNet_hi = model_from_json(loaded_model_json)
		outfile = "result/model_%s.h5"%file
		print('CsiNet_hi: {}'.format(CsiNet_hi))
		encoded = CsiNet_hi.get_layer('CR2_dense').output
		inputs=CsiNet_hi.inputs
		outputs=CsiNet_hi.outputs[0]
		print('inputs: {} - outputs: {}'.format(inputs, outputs))
		CsiNet_hi_pass = Model(inputs=inputs,outputs=[outputs,encoded])
		CsiNet_hi_pass.load_weights(outfile)
		# CsiNet_hi, encoded = CsiNet_model
	else:
		CsiNet_hi, encoded = CsiNet(img_channels, img_height, img_width, M_1, data_format=data_format) # CSINet with M_1 dimensional latent space
	# plot_model(CsiNet_hi, to_file='CsiNet_hi.png')
	aux = Input((M_1,))
	CsiNet_lo, _ = CsiNet(img_channels, img_height, img_width, M_2, aux=aux, data_format=data_format) # CSINet with M_2+M_1 dimensional latent space
	# choose whether to load weights
	if pre_t2_bool:
		date = "10_14"
		CR2 = M_2 # compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
		file = 'CsiNet_'+(envir)+'_dim'+str(M_2)+'_'+date
		outfile = "result/model_%s.h5"%file # h5 file for weights
		CsiNet_lo.load_weights(outfile,by_name=True) # should only load encoder weights

		# rename encoder layers since they conflict
		for layer in CsiNet_lo.layers:
		    if 'CR2' in layer.name:
		    	layer._name = layer.name + str("_2")

	print("--- High Dimensional (M_1) Latent Space CsiNet ---")
	CsiNet_hi_pass.summary()
	print("--- Lower Dimensional (M_2) Latent Space CsiNet ---")
	CsiNet_lo.summary()
	# TO-DO: load weights in hi/lo models
	# load CR=1/4 for M1-generating CsiNet
	# weight_file = get_file(envir, M_1, '09_23')
	# CsiNet_hi.load_weights(weight_file)

	# TO-DO: split large input tensor to use as inputs to 1:T CSINets
	if(data_format == "channels_last"):
		x = Input((T, img_height, img_width, img_channels))
	elif(data_format == "channels_first"):
		x = Input((T, img_channels, img_height, img_width))
	else:
		print("Unexpected data_format param in CsiNet input.") # raise an exception eventually. For now, print a complaint
	# x = Input((T, img_channels, img_height, img_width))
	print('Pre-loop: type(x): {}'.format(type(x)))
	CsiOut = []
	CsiOut_temp = []
	for i in range(T):
		CsiIn = Lambda( lambda x: x[:,i,:,:,:])(x)
		print('#{} - type(CsiIn): {}'.format(i, type(CsiIn)))
		if i == 0:
			# use CsiNet_hi for t=1
			# OutLayer = CsiNet_hi(CsiIn)
			# EncodedLayer = CsiNet_hi.get_layer('CR2_dense').output
			OutLayer, EncodedLayer = CsiNet_hi_pass(CsiIn)
			print('#{} - EncodedLayer: {}'.format(i, EncodedLayer))
		else:
			# use CsiNet_lo for t in [2:T]
			# TO-DO: make sure M_1 codeword from CSINet_hi is an aux input to each CSINet_lo
			OutLayer = CsiNet_lo([EncodedLayer, CsiIn])
		print('#{} - OutLayer: {}'.format(i, OutLayer))
		# CsiOut.append(OutLayer)
		CsiOut.append(Reshape((1,img_height,img_width,img_channels))(OutLayer)) # # when uncommented, model compiles and fits. So issue is with LSTM
	
	# TO-DO: apply concatenated CSINet decoder outputs into unrolled LSTM
	LSTM_model = stacked_LSTM(img_channels, img_height, img_width, T, data_format=data_format)
	# LSTM_model.compile(optimizer='adam', loss='mse')
	print(LSTM_model.summary())

	LSTM_in = concatenate(CsiOut,axis=1)
	print('LSTM_in.shape: {}'.format(LSTM_in.shape))
	LSTM_out = LSTM_model(LSTM_in)

	# compile full model with large 4D tensor as input and LSTM 4D tensor as output
	full_model = Model(inputs=[x], outputs=[LSTM_out])
	# CsiOut_temp = concatenate(CsiOut_temp,axis=1) # when uncommented, model compiles and fits. So issue is with LSTM
	# full_model = Model(inputs=[x], outputs=CsiOut_temp) # when uncommented, model compiles and fits. So issue is with LSTM
	full_model.compile(optimizer='adam', loss='mse')
	full_model.summary()
	return full_model # for now

def stacked_LSTM(img_channels, img_height, img_width, T, LSTM_depth=3, plot_bool=False, data_format="channels_first"):
	# assume entire time-series of CSI from 1:T is concatenated
	LSTM_dim = img_channels*img_height*img_width
	if(data_format == "channels_last"):
		orig_shape = (T, img_height, img_width, img_channels)
	elif(data_format == "channels_first"):
		orig_shape = (T, img_channels, img_height, img_width)
	x = Input(shape=orig_shape)
	recurrent_out = Reshape((T,LSTM_dim))(x)
	for i in range(LSTM_depth):
		recurrent_out = LSTM(LSTM_dim, return_sequences=True)(recurrent_out)
	out = Reshape(orig_shape)(recurrent_out)
	LSTM_model = Model(inputs=[x], outputs=[out])
	if plot_bool:
		LSTM_model.summary()
		plot_model(LSTM_model, "LSTM_model.png")
	return LSTM_model

# # Data loading in batches
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
