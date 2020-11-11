# ======================================================================
# 提取音频文件特征：分帧加窗
# 窄带音频：winsize = 256
# 宽带音频：winsize = 512
# ======================================================================

import os
import h5py
import numpy as np
import wave
import scipy.io.wavfile as sciwav
from preprocess.Extraction_of_speech_features import waveread, enframe


def save_data2hdf5(path, data, keyname):
    file = h5py.File(path, 'w')
    file.create_dataset(keyname, data=data)
    file.close()


def write_hdf5(filepath, hdf5path, keyname, winsize, stepsize, is_nb=False, num=1):
    """
    filepath : 输入文件夹
    hdf5path : 输出文件夹
    keyname  : 关键字标记
    num      : 复制次数
    """
    dirnames = os.listdir(filepath)
    i = 0
    j = 0

    flag = True
    for dirname in dirnames:
        i = i + num
        print('file index:', i)
        # ==============================================
        # 提取特征：分帧加窗
        # ==============================================
        speechpath = filepath + dirname
        signal, channels, sampwidth, framerate = waveread(speechpath)
        frames, wins = enframe(signal, winsize, stepsize, winfunc=np.hamming)

        if flag:
            data = np.tile(frames, (num, 1))
            flag = False
        else:
            data = np.vstack((data, np.tile(frames, (num, 1))))

        if i % 200 == 0:
            j = j + 1
            if j < 10:
                hdf5_path = hdf5path + "00%s.h5" % j
            elif j < 100:
                hdf5_path = hdf5path + "0%s.h5" % j
            else:
                hdf5_path = hdf5path + "%s.h5" % j
            save_data2hdf5(hdf5_path, data, keyname)
            data = []
            flag = True

    if data != []:
        j = j + 1
        if j < 10:
            hdf5_path = hdf5path + "00%s.h5" % j
        elif j < 100:
            hdf5_path = hdf5path + "0%s.h5" % j
        else:
            hdf5_path = hdf5path + "%s.h5" % j
        save_data2hdf5(hdf5_path, data, keyname)

    return None


def extract_input_data(filepath, h5path, winsize, stepsize, keyname):
    print('========= extract nb input data ============')
    if not os.path.exists(h5path):
        os.mkdir(h5path)
    write_hdf5(filepath, h5path, keyname, winsize, stepsize, num=1)
    print('================ finish ... =====================')
    return None


def extract_target_data(filepath, h5path, winsize, stepsize, keyname):
    print('========= extract wb output(target) data ============')
    if not os.path.exists(h5path):
        os.mkdir(h5path)
    write_hdf5(filepath, h5path, keyname, winsize, stepsize, num=1)
    print('================ finish ... =====================')
    return None


if __name__ == "__main__":
    keyname = 'trainX'
    filepath = "../../TIMIT/train_8k/"
    hdf5path = "../dataset/train_8k/"
    winsize = 256
    stepsize = winsize // 2
    extract_input_data(filepath, hdf5path, winsize , stepsize, keyname)

    kn_wb = 'trainY'
    filepath_wb = "../../TIMIT/train_16k/"
    hdf5path_wb = "../dataset/train_16k/"
    winsize_wb = 512
    stepsize_wb = 256
    extract_target_data(filepath_wb, hdf5path_wb, winsize_wb, stepsize_wb, kn_wb)
