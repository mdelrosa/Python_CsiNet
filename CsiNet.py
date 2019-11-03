import tensorflow as tf
from tensorflow.keras.layers import concatenate, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
# tf.reset_default_graph()

envir = 'indoor' #'indoor' or 'outdoor'
# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim = 512  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32

def CsiNet(img_channels, img_height, img_width, encoded_dim, encoder_in=None, residual_num=2, aux=None, encoded_in=None, data_format="channels_last",name=None):
	
	# Bulid the autoencoder model of CsiNet
	def residual_network(x, residual_num, encoded_dim, aux):
		def add_common_layers(y,enc_bool=False):
			if enc_bool:
				y = BatchNormalization(name='CR2_batch_normalization')(y)
				y = LeakyReLU(name='CR2_leaky_re_lu')(y)
			else:
				y = BatchNormalization()(y)
				y = LeakyReLU()(y)
			return y
		def residual_block_decoded(y):
			shortcut = y

			# according to CsiNet-LSTM paper Fig. 1, residual network has 2-filter conv2D layers before other conv2D layers
			y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format=data_format)(y)
			y = add_common_layers(y)

			y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format=data_format)(y)
			y = add_common_layers(y)
			
			y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format=data_format)(y)
			y = add_common_layers(y)
			
			y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format=data_format)(y)
			y = BatchNormalization()(y)

			y = add([shortcut, y])
			y = LeakyReLU()(y)

			return y
		
		# if encoder_in:
		x = Conv2D(2, (3, 3), padding='same', data_format=data_format, name='CR2_conv2d')(x)
		x = add_common_layers(x,enc_bool=True)
		
		x = Reshape((img_total,), name='CR2_reshape')(x)
		encoded = Dense(encoded_dim, activation='linear', name='CR2_dense')(x)
		# else:
		# 	x = Conv2D(2, (3, 3), padding='same', data_format=data_format)(x)
		# 	x = add_common_layers(x)
			
		# 	x = Reshape((img_total,))(x)
		# 	encoded = Dense(encoded_dim, activation='linear')(x)
		print("Aux check: {}".format(aux))
		tens_type = type(x)
		if type(aux) == tens_type:
			encoded = concatenate([aux,encoded])

		x = Dense(img_total, activation='linear')(encoded)
		# reshape based on data_format
		if(data_format == "channels_first"):
			x = Reshape((img_channels, img_height, img_width,))(x)
		elif(data_format == "channels_last"):
			x = Reshape((img_height, img_width, img_channels,))(x)

		for i in range(residual_num):
			x = residual_block_decoded(x)
		x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format=data_format)(x)

		return [x, encoded]

	if(data_format == "channels_last"):
		image_tensor = Input((img_height, img_width, img_channels))
	elif(data_format == "channels_first"):
		image_tensor = Input((img_channels, img_height, img_width))
	else:
		print("Unexpected tensor_shape param in CsiNet input.")
		# raise Exception
	# image_tensor = Input((img_channels, img_height, img_width))
	[network_output, encoded] = residual_network(image_tensor, residual_num, encoded_dim, aux)
	print('network_output: {} - encoded: {} -  aux: {}'.format(network_output, encoded, aux))
	tens_type = type(image_tensor)
	print('image_tensor.dtype: {}'.format(tens_type))
	print('type(aux): {}'.format(type(aux)))
	if type(aux) == tens_type:
		autoencoder = Model(inputs=[aux,image_tensor], outputs=[network_output,encoded])
	else:
		autoencoder = Model(inputs=[image_tensor], outputs=[network_output, encoded])
	if encoder_in:
		autoencoder.load_weights(by_name=True)
	return [autoencoder, encoded]
