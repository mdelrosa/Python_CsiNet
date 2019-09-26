# training_testing_data_generation_batch.py
# load batches of data, normalize entire dataset, export back into batches?
# -> mat file probably won't support all data...
# ---> save as multiple mat files?
# ---> save as single csv?
import scipy.io as sio
import numpy as np
import os # for checking process memory
import psutil # for checking process memory
import sys

def load_batches(batch_num, base_name='indoor53_1user_10slots'):
    # iterate through batches, add to base values
    for i in range(1,batch_num+1):
        try:
            print("Loading batch #{}...".format(i))
            mat = sio.loadmat('{}_part{}.mat'.format(base_name, i))
            # mat_up = sio.loadmat('{}_part{}.mat'.format(base_name, i))
            print('-> mat["H_down"].shape: {}'.format(mat["H_down"].shape))
            print('-> mat["H_up"].shape: {}'.format(mat["H_up"].shape))
            if i == 1:
                Hur_down = mat["H_down"]
                Hur_up = mat["H_up"]
            else:
                Hur_down = np.vstack((Hur_down, mat["H_down"]))
                Hur_up = np.vstack((Hur_up, mat["H_up"]))
            print('-> Hur_down.shape: {}'.format(Hur_down.shape))
            print('-> Hur_up.shape: {}'.format(Hur_up.shape))
        except IOError as e:
            batch_num = batch_num - 1
            print(e.errno)

    return [Hur_down, Hur_up, batch_num]

def normalize_dataset(Hur_down, Hur_up, batch_num, train_ratio=0.7):
    check_process_memory(stage='start')
    # generate normalized dataset
    # 2D delay domain for OrgNet
    # n_ant = int(np.sqrt(Hur_down.shape[1]/2))
    n_ant = int(Hur_down.shape[2])
    print("Number of antennas: {}".format(n_ant))
    # org_up = Hur_up
    # org_down = Hur_down
    num_samples = Hur_up[:,1].shape[0]
    num_taps = Hur_up[:,1].shape[2]
    split_ind = round(train_ratio*num_samples)
    print('Batches contain {} and {} samples for training and validation.'.format(split_ind, num_samples-split_ind))

    print('> Save H3 uplink batches')
    H_up_n1 = norm_H3(Hur_up);
    save_dataset(H_up_n1[0:split_ind,:], batch_num, data_str='Data100_Htrainin_up_FDD', key_str='HU_train', n_ant=n_ant) # training batch
    save_dataset(H_up_n1[split_ind+1:num_samples,:], batch_num, data_str='Data100_Hvalin_up_FDD', key_str='HU_val', n_ant=n_ant) # validation batch
    check_process_memory(stage='pre-H3 Up')
    del H_up_n1 # clean up uplink normalized data
    check_process_memory(stage='post-H3 Up')
    
    print('> Save H3 downlink batches')
    H_down_n1 = norm_H3(Hur_down);
    save_dataset(H_down_n1[0:split_ind,:], batch_num, data_str='Data100_Htrainin_down_FDD', key_str='HD_train',n_ant=n_ant) # training batch
    save_dataset(H_down_n1[split_ind+1:num_samples,:], batch_num, data_str='Data100_Hvalin_down_FDD', key_str='HD_val', n_ant=n_ant) # validation batch
    check_process_memory(stage='pre-H3 Down')
    del H_down_n1 # clean up uplink normalized data
    check_process_memory(stage='post-H3 Down')
    
    print('> Save H2 uplink batches')
    H_up_n2 = norm_H2(Hur_up);
    save_dataset(H_up_n2[0:split_ind,:], batch_num, data_str='Data100_Htrainin_up_FDD2',key_str='HU_train', n_ant=n_ant) # training batch
    save_dataset(H_up_n2[split_ind+1:num_samples,:], batch_num, data_str='Data100_Hvalin_up_FDD2',key_str='HU_val', n_ant=n_ant) # validation batch
    check_process_memory(stage='pre-H2 Up')
    del H_up_n2
    check_process_memory(stage='post-H2 Up')
    
    print('> Save H2 downlink batches')
    H_down_n2 = norm_H2(Hur_down);
    save_dataset(H_down_n2[0:split_ind,:], batch_num, data_str='Data100_Htrainin_down_FDD2',key_str='HD_train', n_ant=n_ant) # training batch
    save_dataset(H_down_n2[split_ind+1:num_samples,:], batch_num, data_str='Data100_Hvalin_down_FDD2',key_str='HD_val', n_ant=n_ant) # validation batch
    check_process_memory(stage='pre-H2 Down')
    del H_down_n2
    check_process_memory(stage='post-H2 Down')

    # CSINet does not use mag data. Ignore for now.
    # calc magnitude
    # half_ind = round(num_taps/2)
    # HD_1 = Hur_down[:,0:half_ind]
    # HD_2 = Hur_down[:,half_ind:num_samples]
    # HU_1 = Hur_up[:,0:half_ind]
    # HU_2 = Hur_up[:,half_ind:num_samples]
    # HD_mag_tmp = HD_1**2 + HD_2**2
    # HD_mag = np.sqrt(HD_mag_tmp)
    # HU_mag_tmp = HU_1**2 + HU_2**2
    # HU_mag = np.sqrt(HU_mag_tmp)

    # H_down_n3 = norm_H2(HD_mag)
    # H_up_n3 = norm_H2(HU_mag)

    # print('> Save mag uplink/downlink batches')
    # data_dict_train = {'HD_train': H_down_n3[0:split_ind,:],'HU_train': H_up_n3[0:split_ind,:]}
    # save_dataset_dict(data_dict_train, batch_num, round(train_ratio*num_samples), data_str='Data100_Htrainin_mag', n_ant=n_ant) # training batch
    # check_process_memory(stage='pre-mag Up/Down Training')
    # del data_dict_train
    # check_process_memory(stage='post-mag Up/Down Training')
    # data_dict_val = {'HD_val': H_down_n3[split_ind:num_samples,:],'HU_val': H_up_n3[split_ind:num_samples,:]}
    # save_dataset_dict(data_dict_val, batch_num, round((1-train_ratio)*num_samples), data_str='Data100_Hvalin_mag', n_ant=n_ant) # training batch
    # check_process_memory(stage='pre-mag Up/Down Validation')
    # del data_dict_val
    # check_process_memory(stage='post-mag Up/Down Validation')

def check_process_memory(stage=None):
    process = psutil.Process(os.getpid())
    stage_mem = process.memory_info().rss / 1e6 # memory in GB
    if stage is None:
        print('-> {} MB'.format(stage_mem))
    else:
        print('-> {} - {} MB'.format(stage, stage_mem))
    return stage_mem
        
def norm_H3(data):
    # Normalize data to [0,1]. Min-Max Feature scaling
    data_min = np.min(data)
    normH = data - data_min
    normH_max = np.max(normH)
    normH = (data - data_min) / normH_max;
    # print('norm_H3 -- data_min: {} - normH_max: {}'.format(data_min, normH_max))
    return normH

def norm_H2(data):
    # Normalize data to [0.5,1]. Min-Max Feature scaling
    data = np.absolute(data);
    data_min = np.min(data)
    normH = data - np.min(data);
    normH_max = np.max(normH)
    normH = ( (normH / normH_max) + 1 ) / 2;
    # print('norm_H2 -- data_min: {} - normH_max: {}'.format(data_min, normH_max))
    return normH

def save_dataset(data, batch_num, data_str='data', key_str='data', batch_split=None, n_ant=None):
    # iteratively save batches of data due to file limits; takes single np.array data input
    print('-> Saving data in batches...')
    if batch_split == None:
        batch_split = data.shape[0] / batch_num
    for i in range(1, batch_num+1):
        start = int((i-1)*batch_split+1)
        end = int(i*batch_split)
        data_temp = data[start:end,:]
        if n_ant == None:
            sio.savemat('{}_{}.mat'.format(data_str, i), {key_str: data_temp})
        else:
            sio.savemat('{}_{}ant_{}.mat'.format(data_str, n_ant,i), {key_str: data_temp})
        print('--> Batch #{} saved.'.format(i))
        
def save_dataset_dict(data_dict, batch_num, num_samples, data_str='data', batch_split=None, n_ant=None):
    # iteratively save batches of data due to file limits; takes dictionary input
    print('-> Saving data in batches...')
    batch_num = 10
    if batch_split == None:
        batch_split = num_samples / batch_num
    for i in range(1, batch_num+1):
        start = int((i-1)*batch_split+1)
        end = int(i*batch_split)
        dict_temp = {}
        for key, val in data_dict.items():
            # data_temp = data[start:end,:]
            dict_temp[key] = data_dict[key][start:end,:]
        if n_ant == None:
            sio.savemat('{}_{}.mat'.format(data_str, i), dict_temp)
        else:
            sio.savemat('{}_{}ant_{}.mat'.format(data_str, n_ant,i), dict_temp)
        print('--> Batch #{} saved.'.format(i))

def main():
    batch_num = int(sys.argv[1])
    str_base = sys.argv[2]
    [Hur_down, Hur_up, batch_num] = load_batches(batch_num,str_base) # accounts for missing batches in data
    normalize_dataset(Hur_down, Hur_up, batch_num)

if __name__=="__main__":
    main()