# -*- coding: utf-8 -*-
import numpy as np
import time
import os
import h5py


def read_hdf5(pathname,filenames,keyname):
    flag = True
    for dirname in filenames:
        path = pathname + dirname
        f = h5py.File(path, 'r')
        data_part = f[keyname]
        if flag:
            data = np.tile(data_part,(1,1))
            flag = False
        else:
            data = np.vstack((data, data_part))
        f.close()
    return data.astype('float32')


def data_scale_transform(data,frames):
    '''
    normal martix, copy context 
    :param data:
    :param frames: 
    :return: tr_x(N,frames,129or256)
    '''
    if frames == 1:
        dimenson_of_frames = len(data[0])
        return data.reshape((-1,1,dimenson_of_frames))
    else:
        num_of_frames = len(data)
        dimenson_of_frames = len(data[0])
        tr_x = np.zeros((num_of_frames,frames,dimenson_of_frames))
        for i in range(0,num_of_frames):
            for j in range(0,frames):
                x_num = i + j - (frames -1 ) // 2
                if x_num < 0 or x_num >= num_of_frames:
                    pass
                else:
                    tr_x[i, j, ] = data[x_num,]
        return tr_x


def load_train_data(context_num=1,shape_num=1):
    '''
    read h5
    :param context_num:  
    :param shape_num:  
    :return:    train_x(N,context_num,129or256)  train_y(N,129or256)
    '''
    x_t1 = time.time()
    
    train_nb_speech_path = "./dataset/train_8k/"
    train_nb_speech_names = os.listdir(train_nb_speech_path)
    train_x = read_hdf5(train_nb_speech_path,train_nb_speech_names,"trainX")
    # train_x[:, 256] = train_x[:, 0]
    
    train_x_mean = train_x.mean(axis=0) # 以列为单位进行运算
    train_x_std = train_x.std(axis=0)   
    train_x_normal = (train_x - train_x_mean)/train_x_std
    # ================
# =============================================================================
#     valida_nb_speech_path = "data/valida_8k/"
#     valida_nb_speech_names = os.listdir(valida_nb_speech_path)
#     valida_x = read_hdf5(valida_nb_speech_path,valida_nb_speech_names,"validaX")
#     
#     valida_x_normal = (valida_x - train_x_mean)/train_x_std
# =============================================================================
    
    x_t2 = time.time()
    
    # ===============
    train_wb_speech_path = "./dataset/train_16k/"
    train_wb_speech_names = os.listdir(train_wb_speech_path)
    train_y = read_hdf5(train_wb_speech_path,train_wb_speech_names,"trainY")
    
    train_y_mean = train_y.mean(axis=0)
    train_y_std = train_y.std(axis=0)   
    train_y_normal = (train_y - train_y_mean)/train_y_std 
    # ===============
# =============================================================================
#     valida_wb_speech_path = "data/valida_16k/"
#     valida_wb_speech_names = os.listdir(valida_wb_speech_path)
#     valida_y = read_hdf5(valida_wb_speech_path,valida_wb_speech_names,"validaY")
#     
#     valida_y_normal = (valida_y - train_y_mean)/train_y_std
# =============================================================================
    
    x_t3 = time.time()

    print("prepare NB data time:", x_t2 - x_t1)
    print("prepare WB data time", x_t3 - x_t2)
    print(train_x_normal.shape, train_y_normal.shape)
    
    params_file = 'params.h5'
    if not os.path.exists(params_file):
        file = h5py.File(params_file, 'w')
        file.create_dataset('x_mean', data=train_x_mean)
        file.create_dataset('x_std', data=train_x_std)
        file.create_dataset('y_mean', data=train_y_mean)
        file.create_dataset('y_std', data=train_y_std)
        file.close()
     
    return train_x_normal, train_y_normal


def load_test_data(test_x,X_mean,X_std,context_num):       
    test_x_processing = (test_x-X_mean)/X_std
    test_x_processed = data_scale_transform(test_x_processing,context_num)
    return test_x_processed
