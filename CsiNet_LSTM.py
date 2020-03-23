# CsiNet_LSTM.py

import tensorflow as tf
from tensorflow.keras.layers import concatenate, Lambda, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, LSTM, CuDNNLSTM, ConvLSTM2D
from tensorflow.keras import Input
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers 
from tensorflow.keras.optimizers import Adam
import scipy.io as sio 
import numpy as np
import math
import time
# from CsiNet import *
from CsiNet_v2 import *
from unpack_json import *

# tf.reset_default_graph()
# tf.enable_eager_execution()

# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim = 512  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32

def get_file(envir,encoded_dim,train_date):
        file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+'_'+train_date
        return "result/model_%s.h5"%file

def make_CsiNet(aux_bool,M_1, img_channels, img_height, img_width, encoded_dim, data_format, lo_bool=False):
        if aux_bool:
            aux = Input((M_1,))
        else:
            aux = None
        # build CsiNet
        out_activation = 'tanh'
        autoencoder, encoded = CsiNet(img_channels, img_height, img_width, encoded_dim, aux=aux, data_format=data_format, out_activation=out_activation) # CSINet with M_1 dimensional latent space
        # autoencoder = Model(inputs=autoencoder.inputs,outputs=autoencoder.outputs[0])
        prediction = autoencoder.outputs[0]
        encoded = autoencoder.outputs[1]
        if (lo_bool):
            model = Model(inputs=autoencoder.inputs,outputs=prediction)
        else:
            model = Model(inputs=autoencoder.inputs,outputs=[encoded,prediction])
        # return [autoencoder, encoded]
        # optimizer = Adam()
        # model.compile(optimizer=optimizer, loss='mse') 
        return model

def CsiNet_LSTM(img_channels, img_height, img_width, T, M_1, M_2, LSTM_depth=3,data_format='channels_first',t1_trainable=False,t2_trainable=True,pre_t1_bool=True,pre_t2_bool=True,aux_bool=True, share_bool=True, pass_through_bool=False, lstm_latent_bool=False, pre_lstm_bool=True, conv_lstm_bool=False, pretrained_bool=True):
        # base CSINet models
        aux=Input((M_1,))
        if(data_format == "channels_last"):
                x = Input((T, img_height, img_width, img_channels))
        elif(data_format == "channels_first"):
                x = Input((T, img_channels, img_height, img_width))
        else:
                print("Unexpected data_format param in CsiNet input.") # raise an exception eventually. For now, print a complaint
        CsiNet_hi = make_CsiNet(aux_bool, M_1, img_channels, img_height, img_width, M_1, data_format)
        if (pretrained_bool):
                # config_hi = 'config/indoor0001/v2/csinet_cr512.json'
                config_hi = 'config/outdoor300/v2/csinet_cr512.json'
                dim, date, model_dir = unpack_compact_json(config_hi)
                network_name = get_network_name(config_hi)
                CsiNet_hi = load_weights_into_model(network_name,model_dir,CsiNet_hi)
        CsiNet_hi._name = "CsiNet_hi"
        CsiNet_hi.trainable = t1_trainable
        print("--- High Dimensional (M_1) Latent Space CsiNet ---")
        CsiNet_hi.summary()
        print('CsiNet_hi.inputs: {}'.format(CsiNet_hi.inputs))
        print('CsiNet_hi.outputs: {}'.format(CsiNet_hi.outputs))
        # TO-DO: split large input tensor to use as inputs to 1:T CSINets
        CsiOut = []
        CsiOut_temp = []
        for i in range(T):
                CsiIn = Lambda( lambda x: x[:,i,:,:,:])(x)
                print('#{}: CsiIn: {}'.format(i,CsiIn))
                if i == 0:
                        # use CsiNet_hi for t=1
                        # OutLayer, EncodedLayer = CsiNet_hi([aux,CsiIn])
                        EncodedLayer, OutLayer = CsiNet_hi([aux,CsiIn])
                        print('EncodedLayer: {}'.format(EncodedLayer))
                else:
                        # choose whether or not to share parameters between low-dimensional timeslots
                        if (i==1 or not share_bool):
                                CsiNet_lo = make_CsiNet(aux_bool, M_1, img_channels, img_height, img_width, M_2, data_format, lo_bool=True)
                                if (pretrained_bool):
                                        # config_lo = 'config/indoor0001/v2/csinet_cr{}.json'.format(M_2)
                                        config_lo = 'config/outdoor300/v2/csinet_cr{}.json'.format(M_2)
                                        dim, date, model_dir = unpack_compact_json(config_lo)
                                        network_name = get_network_name(config_lo)
                                        CsiNet_lo = load_weights_into_model(network_name,model_dir,CsiNet_lo)
                                CsiNet_lo.trainable = t2_trainable
                                CsiNet_lo._name = "CsiNet_lo_{}".format(i)
                                print('CsiNet_lo.inputs: {}'.format(CsiNet_lo.inputs))
                                print('CsiNet_lo.outputs: {}'.format(CsiNet_lo.outputs))
                                if i==1:
                                        print("--- Low Dimensional (M_2) Latent Space CsiNet ---")
                                        CsiNet_lo.summary()
                        if aux_bool:
                                OutLayer = CsiNet_lo([EncodedLayer,CsiIn])
                        else:
                                # use CsiNet_lo for t in [2:T]
                                OutLayer = CsiNet_lo(CsiIn)
                print('#{} - OutLayer: {}'.format(i, OutLayer))
                if data_format == "channels_last":
                        CsiOut.append(Reshape((1,img_height,img_width,img_channels))(OutLayer)) 
                if data_format == "channels_first":
                        CsiOut.append(Reshape((1,img_channels,img_height,img_width))(OutLayer)) 
                # for the moment, we don't handle separate case of loading convLSTM
        # lstm_config = 'config/indoor0001/lstm_depth3_opt.json'
        if conv_lstm_bool:
            print('---> Convolutional recurrent activations.')
            LSTM_model = stacked_convLSTM(img_channels, img_height, img_width, T, lstm_latent_bool, LSTM_depth=LSTM_depth,data_format=data_format)
        else:
            print('---> Non-convolutional recurrent activations.')
            LSTM_model = stacked_LSTM(img_channels, img_height, img_width, T, lstm_latent_bool, LSTM_depth=LSTM_depth,data_format=data_format)
            # comment back in to load in weights
            # if (pretrained_bool):
            #     lstm_config = 'config/outdoor300/lstm_depth3_opt.json'
            #     dim, date, model_dir = unpack_compact_json(lstm_config)
            #     network_name = get_network_name(lstm_config)
            #     LSTM_model = load_weights_into_model(network_name,model_dir,LSTM_model) # option to load weights; try random initialization for the network
        print(LSTM_model.summary())

        LSTM_in = concatenate(CsiOut,axis=1)
        print('LSTM_in.shape: {}'.format(LSTM_in.shape))
        LSTM_out = LSTM_model(LSTM_in)

        # compile full model with large 4D tensor as input and LSTM 4D tensor as output
        if pass_through_bool:
            full_model = Model(inputs=[aux,x], outputs=[LSTM_in])
        else:
            full_model = Model(inputs=[aux,x], outputs=[LSTM_out])
        full_model.compile(optimizer='adam', loss='mse')
        full_model.summary()
        return full_model

def CsiNet_ConvLSTM(img_channels, img_height, img_width, T, M_1, M_2, LSTM_depth=2,data_format='channels_first',t1_trainable=False,t2_trainable=True,pre_t1_bool=True,pre_t2_bool=True,aux_bool=True, lstm_latent_bool=False, pre_lstm_bool=False, conv_lstm_bool=True):
        # convolutional lstm layers for encoder/decoder 
        # base CSINet models
        aux=Input((M_1,))
        # base CsiNet model at t=1: high CR
        CsiNet_hi, encoded = CsiNet(img_channels, img_height, img_width, M_1, aux=aux,data_format=data_format) # CSINet with M_1 dimensional latent space
        if pre_t1_bool:
                date = "11_09"
                file = 'CsiNet_'+(envir)+'_dim'+str(M_1)+'_'+date
                outfile = "aux/model_%s.h5"%file
                CsiNet_hi.load_weights(outfile)
        CsiNet_hi.trainable = t1_trainable
        aux_lo = Input((M_1,))

        print("--- High Dimensional (M_1) Latent Space CsiNet ---")
        CsiNet_hi.summary()

        # TO-DO: split large input tensor to use as inputs to 1:T CSINets
        if(data_format == "channels_last"):
                data_shape = (T, img_height, img_width, img_channels)
                data_shape_small = (1,img_height, img_width, img_channels)
        elif(data_format == "channels_first"):
                data_shape = (T, img_channels, img_height, img_width)
                data_shape_small = (1,img_channels, img_height, img_width)
        else:
                print("Unexpected data_format param in CsiNet input.") # raise an exception eventually. For now, print a complaint
        x = Input(data_shape)
        print('Pre-loop: type(x): {}'.format(type(x)))
        CsiOut = []
        CsiOut_temp = []
        y = ConvLSTM2D(filters=2, input_shape=data_shape, kernel_size=(3, 3), padding='same', return_sequences=True,stateful=False,recurrent_activation='sigmoid',data_format=data_format)(x)
        y = add_common_layers(y)
        encoded_list = []
        decoded_list = []
        for i in range(T):
                EncodedIn = Lambda( lambda x: Reshape((img_total,))(x[:,i,:,:,:]))(y)
                M = M_1 if i==0 else M_2
                encoded = Dense(M, activation='linear', name='t{}_dense'.format(i+1))(EncodedIn)
                if aux_bool and i > 0:
                    encoded = concatenate([encoded_list[0],encoded])
                decoded = Dense(img_total, activation='linear',name='t{}_decode'.format(i+1))(encoded)
                decoder = Reshape(data_shape_small)(decoded)
                encoded_list.append(encoded)
                decoded_list.append(decoder)
        print('decoded_list: {}'.format(decoded_list))
        LSTM_in = concatenate(decoded_list,axis=1)
        LSTM_model = stacked_convLSTM(img_channels, img_height, img_width, T, lstm_latent_bool, LSTM_depth=LSTM_depth,data_format=data_format)
        print(LSTM_model.summary())
        LSTM_out = LSTM_model(LSTM_in)
        # compile full model with large 4D tensor as input and LSTM 4D tensor as output
        full_model = Model(inputs=[x], outputs=[LSTM_out])
        full_model.summary()
        return full_model # for now

def CsiNet_LSTM_feedback(img_channels, img_height, img_width, T, M_1, M_2, LSTM_depth=3,data_format='channels_first',t1_trainable=False,t2_trainable=True,pre_t1_bool=True,pre_t2_bool=True,aux_bool=True, lstm_latent_bool=False):
        # lstm for feedback layer 
        # base CSINet models
        aux=Input((M_1,))
        # base CsiNet model at t=1: high CR
        CsiNet_hi, encoded = CsiNet(img_channels, img_height, img_width, M_1, aux=aux,data_format=data_format) # CSINet with M_1 dimensional latent space
        if pre_t1_bool:
                date = "11_09"
                file = 'CsiNet_'+(envir)+'_dim'+str(M_1)+'_'+date
                outfile = "aux/model_%s.h5"%file
                CsiNet_hi.load_weights(outfile)
        # CsiNet_hi.trainable = t1_trainable # PUT THIS BACK IN
        aux_lo = Input((M_1,))

        print("--- High Dimensional (M_1) Latent Space CsiNet ---")
        CsiNet_hi.summary()

        CsiNet_hi_enc, CsiNet_hi_dec = split_CsiNet(CsiNet_hi, M_1)
        
        print("--- Encoder from CsiNet_hi (M_1) ---")
        CsiNet_hi_enc.summary()
        CsiNet_hi_dec.summary()

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
                if i == 0:
                        # use CsiNet_hi for t=1
                        OutLayer, EncodedLayer = CsiNet_hi([aux,CsiIn])
                        print('EncodedLayer: {}'.format(EncodedLayer))
                else:
                        if aux_bool:
                            dim, date, model_dir = unpack_compact_json('config/csinet_aux_test_cr{}.json'.format(M_2))
                            file = 'CsiNet_'+(envir)+'_dim'+str(dim)+'_'+date
                            CsiNet_lo = model_with_weights_h5(file,envir,dim,date,model_dir,weights_bool=pre_t2_bool) # option to load weights; try random initialization for the network
                        else:
                            date = "10_30"
                            file = 'aux/model_CsiNet_'+(envir)+'_dim'+str(M_2)+'_'+date
                        CsiNet_lo.trainable = t2_trainable
                        CsiNet_lo._name = "CsiNet_lo_t{}".format(i+1)
                        if i==1:
                            print("--- Low Dimensional (M_2) Latent Space CsiNet ---")
                            CsiNet_lo.summary()
                        if aux_bool:
                            OutLayer = CsiNet_lo([EncodedLayer,CsiIn])
                        else:
                            # use CsiNet_lo for t in [2:T]
                            OutLayer = CsiNet_lo(CsiIn)
                print('#{} - OutLayer: {}'.format(i, OutLayer))
                if data_format == "channels_last":
                        CsiOut.append(Reshape((1,img_height,img_width,img_channels))(OutLayer)) 
                if data_format == "channels_first":
                        CsiOut.append(Reshape((1,img_channels,img_height,img_width))(OutLayer)) 
        if conv_lstm_bool:
            print('---> Convolutional recurrent activations.')
            LSTM_model = stacked_convLSTM(img_channels, img_height, img_width, T, lstm_latent_bool, LSTM_depth=LSTM_depth,data_format=data_format)
        else:
            print('---> Non-convolutional recurrent activations.')
            LSTM_model = stacked_LSTM(img_channels, img_height, img_width, T, lstm_latent_bool, LSTM_depth=LSTM_depth,data_format=data_format)
        print(LSTM_model.summary())

        LSTM_in = concatenate(CsiOut,axis=1)
        print('LSTM_in.shape: {}'.format(LSTM_in.shape))
        LSTM_out = LSTM_model(LSTM_in)

        # compile full model with large 4D tensor as input and LSTM 4D tensor as output
        pass_through = False
        if pass_through:
            full_model = Model(inputs=[aux,x], outputs=[LSTM_in])
        else:
            full_model = Model(inputs=[aux,x], outputs=[LSTM_out])
        # CsiOut_temp = concatenate(CsiOut_temp,axis=1) # when uncommented, model compiles and fits. So issue is with LSTM
        # full_model = Model(inputs=[x], outputs=CsiOut_temp) # when uncommented, model compiles and fits. So issue is with LSTM
        full_model.compile(optimizer='adam', loss='mse')
        full_model.summary()
        return full_model # for now

def split_CsiNet(model,CR):
        # split model into encoder and decoder
        layers = []
        for layer in model.layers:
                # print("layer.name: {} - type(layer): {}".format(layer.name, type(layer)))
                # layers.append(layer)
                # if 'dense' in layer.name:
                #     print('Dense layer "{}" has output shape {}'.format(layer.name,layer.output_shape))
                if layer.output_shape == (None,CR):
                    print('Feedback layer "{}"'.format(layer.name))
                    feedback_layer_output = layer.output # take feedback layer as output of decoder
                elif 'dense' in layer.name:
                    enc_input = layer.input # get concatenate layer's dimension to generate new inp for encoder
        dec_input = model.input
        # enc_input = Input((enc_in_dim))
        dec_model = Model(inputs=[dec_input],outputs=[feedback_layer_output])
        enc_model = Model(inputs=[enc_input],outputs=[model.output])  

def stacked_LSTM(img_channels, img_height, img_width, T, lstm_latent_bool, LSTM_depth=3, plot_bool=False, data_format="channels_first",kernel_initializer=initializers.glorot_uniform(seed=100), recurrent_initializer=initializers.orthogonal(seed=100)):
        # assume entire time-series of CSI from 1:T is concatenated
        LSTM_dim = img_channels*img_height*img_width
        if(data_format == "channels_last"):
                orig_shape = (T, img_height, img_width, img_channels)
        elif(data_format == "channels_first"):
                orig_shape = (T, img_channels, img_height, img_width)
        x = Input(shape=orig_shape)
        LSTM_tup = (T,LSTM_dim)
        recurrent_out = Reshape(LSTM_tup)(x)
        for i in range(LSTM_depth):
            # By default, orthogonal/glorot_uniform initializers for recurrent/kernel
                # recurrent_out = LSTM(LSTM_dim, return_sequences=True, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, stateful=False)(recurrent_out)
                recurrent_out = CuDNNLSTM(LSTM_dim, return_sequences=True, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, stateful=False)(recurrent_out) # CuDNNLSTM does not support recurrent activations; switch back to vanilla LSTM in meantime
                print("Dim of LSTM #{} - {}".format(i+1,recurrent_out.shape))
        out = Reshape(orig_shape)(recurrent_out)
        LSTM_model = Model(inputs=[x], outputs=[out])
        return LSTM_model

def stacked_convLSTM(img_channels, img_height, img_width, T, lstm_latent_bool, LSTM_depth=3, plot_bool=False, data_format="channels_first"):
        # assume entire time-series of CSI from 1:T is concatenated
        LSTM_dim = img_channels*img_height*img_width
        if(data_format == "channels_last"):
                orig_shape = (T, img_height, img_width, img_channels)
        elif(data_format == "channels_first"):
                orig_shape = (T, img_channels, img_height, img_width)
        x = Input(shape=orig_shape)
        recurrent_out = x 
        for i in range(LSTM_depth):
                recurrent_out = residual_block_LSTM(recurrent_out, orig_shape, data_format, return_sequences=True, stateful=False)
                print("Dim of LSTM #{} - {}".format(i+1,recurrent_out.shape))
        LSTM_model = Model(inputs=[x], outputs=[recurrent_out])
        if plot_bool:
                LSTM_model.summary()
                plot_model(LSTM_model, "LSTM_model.png")
        return LSTM_model
    
def residual_block_LSTM(y,orig_shape,data_format,return_sequences=True,stateful=False):
        shortcut = y
        # according to CsiNet-LSTM paper Fig. 1, residual network has 2-filter conv2D layers before other conv2D layers
        print('Init y.shape: {}'.format(y.shape))
        y = ConvLSTM2D(filters=2, input_shape=orig_shape, kernel_size=(3, 3), padding='same', return_sequences=return_sequences,stateful=stateful,data_format=data_format)(y)
        y = add_common_layers(y)
        print('After y.shape: {}'.format(y.shape))

        y = ConvLSTM2D(filters=8, input_shape=orig_shape, kernel_size=(3, 3), padding='same', return_sequences=return_sequences,stateful=stateful,data_format=data_format)(y)
        y = add_common_layers(y)
        
        y = ConvLSTM2D(filters=16, input_shape=orig_shape, kernel_size=(3, 3), padding='same', return_sequences=return_sequences,stateful=stateful,data_format=data_format)(y)
        y = add_common_layers(y)
        
        y = ConvLSTM2D(filters=2, input_shape=orig_shape, kernel_size=(3, 3), padding='same', return_sequences=return_sequences,stateful=stateful,data_format=data_format)(y)
        y = BatchNormalization()(y)

        y = add([shortcut, y])
        y = LeakyReLU()(y)

        return y

def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

def stacked_LSTM_exp(img_channels, img_height, img_width, T, lstm_latent_bool, LSTM_depth=3, plot_bool=False, data_format="channels_first"):
        # assume entire time-series of CSI from 1:T is concatenated
        # experimental LSTM; trying different configs
        LSTM_dim = img_channels*img_height*img_width
        if(data_format == "channels_last"):
                orig_shape = (T, img_height, img_width, img_channels)
        elif(data_format == "channels_first"):
                orig_shape = (T, img_channels, img_height, img_width)
        x = Input(shape=orig_shape)
        LSTM_tup = (T,LSTM_dim)
        recurrent_out = Reshape(LSTM_tup)(x)
        for i in range(LSTM_depth):
            if i < LSTM_depth-1 and lstm_latent_bool:
                # recurrent_out = LSTM(LSTM_dim, activation='sigmoid',unroll=True,return_sequences=True)(recurrent_out)
                recurrent_out = LSTM(LSTM_dim, return_sequences=True, stateful=False, recurrent_activation='leaky_relu')(recurrent_out)
            else:
                recurrent_out = LSTM(LSTM_dim, return_sequences=True, stateful=False, recurrent_activation='sigmoid')(recurrent_out)
                print("Dim of LSTM #{} - {}".format(i+1,recurrent_out.shape))
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
