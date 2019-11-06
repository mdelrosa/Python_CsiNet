# CsiNet_Temp.py

import tensorflow as tf
from tensorflow.keras.layers import concatenate, Lambda, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, LSTM
from tensorflow.keras import Input
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
from CsiNet import *

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
pre_t2_bool = True

def get_file(envir,encoded_dim,train_date):
        file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+'_'+train_date
        return "result/model_%s.h5"%file

def CsiNet_Temp(img_channels, img_height, img_width, T, M_1, M_2, data_format='channels_first', t1_trainable=True, pre_t1_bool=True, pre_t2_bool=True, aux_bool=True):
        # base CsiNet model at t=1: high CR
        CsiNet_hi, encoded = CsiNet(img_channels, img_height, img_width, M_1, data_format=data_format) # CSINet with M_1 dimensional latent space
        if pre_t1_bool:
                date = "10_14"
                file = 'CsiNet_'+(envir)+'_dim'+str(M_1)+'_'+date
                # outfile = "result/model_%s.json"%file
                # json_file = open(outfile)
                # loaded_model_json = json_file.read()
                # json_file.close()
                # CsiNet_hi = model_from_json(loaded_model_json)
                outfile = "result/model_%s.h5"%file
                CsiNet_hi.load_weights(outfile)
        CsiNet_hi.trainable = t1_trainable
        # base CsiNet model for t>=1 choose whether to load weights
        print("--- High Dimensional (M_1) Latent Space CsiNet ---")
        CsiNet_hi.summary()
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
                if i == 0:
                        # use CsiNet_hi for t=1
                        OutLayer, EncodedLayer = CsiNet_hi(CsiIn)
                else:
                        if aux_bool:
                            date = "10_30"
                            file = 'aux/model_CsiNet_'+(envir)+'_dim'+str(M_2)+'_'+date
                        else:
                            date = "10_14"
                            file = 'result/model_CsiNet_'+(envir)+'_dim'+str(M_2)+'_'+date
                        outfile = "%s.json"%file # h5 file for weights
                        json_file = open(outfile)
                        loaded_model_json = json_file.read()
                        json_file.close()
                        CsiNet_lo = model_from_json(loaded_model_json)
                        outfile = "%s.h5"%file
                        CsiNet_lo.load_weights(outfile)
                        CsiNet_lo._name = 'CsiNet_lo_t{}'.format(i+1)
                        if aux_bool:
                            OutLayer = CsiNet_lo([EncodedLayer,CsiIn])
                        else:
                            # use CsiNet_lo for t in [2:T]
                            OutLayer = CsiNet_lo(CsiIn)
                print('#{} - OutLayer: {}'.format(i, OutLayer))
                CsiOut.append(Reshape((1,img_height,img_width,img_channels))(OutLayer)) # when uncommented, model compiles and fits. So issue is with LSTM

        CsiNet_Temp_out = concatenate(CsiOut,axis=1)
        # print('LSTM_in.shape: {}'.format(LSTM_in.shape))
        # LSTM_out = LSTM_model(LSTM_in)

        # compile full model with large 4D tensor as input and LSTM 4D tensor as output
        full_model = Model(inputs=[x], outputs=[CsiNet_Temp_out])
        # CsiOut_temp = concatenate(CsiOut_temp,axis=1) # when uncommented, model compiles and fits. So issue is with LSTM
        # full_model = Model(inputs=[x], outputs=CsiOut_temp) # when uncommented, model compiles and fits. So issue is with LSTM
        full_model.compile(optimizer='adam', loss='mse')
        full_model.summary()
        return full_model # for now
