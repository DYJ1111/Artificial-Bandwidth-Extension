# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:52:57 2018

@author: Administrator

合成语音
"""

from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import wave
import struct
from pandas import DataFrame
import math
import h5py
from scipy import signal
import keras
# from preprocess.Extraction_of_speech_features import waveread, enframe, spectrum_magnitude
# from model.tf_model import dft_matrix, after_fft_layer, before_fft_layer
# import tensorflow as tf


file = h5py.File('params.h5', 'r')
X_mean = file['x_mean']
X_std = file['x_std']
Y_mean = file['y_mean']
Y_std = file['y_std']


def restore_speech(magnitude, phase, inc, frames, frame_length):
    '''
    :param magnitude:  幅度
    :param phase: 相位
    :param inc:
    :param hamwin: 用于除去汉明窗的增益
    '''
    NFFT = frame_length
    # 反傅里叶
    spectral = np.multiply(np.sqrt(np.exp(magnitude)), np.exp(np.multiply(complex(0, 1), phase)))
    # (N,512)
    xi_all = np.real(np.fft.irfft(spectral, NFFT))  # 宽带信号
    # ==========================================================================
    # 定义滤波器
    fs = 16000
    filterOrder = 12
    cutOffFreq = 4000
    b_lpf, a_lpf = signal.butter(filterOrder, cutOffFreq / (fs / 2), 'low')
    b_hpf, a_hpf = signal.butter(filterOrder, cutOffFreq / (fs / 2), 'high')
    # tukeywin：r defaults to 0.5.
    # If you input r ≤ 0, you obtain a rectwin window.
    # If you input r ≥ 1, you obtain a hann window.
    win_WB = signal.tukey(NFFT, 0.25)

    # 上采样：即内插0.......frames_up数据来自窄带frames
    frames_up = np.zeros((frames.shape[0], 2 * frames.shape[1]))
    for i in range(0, frames_up.shape[0]):
        for j in range(0, frames_up.shape[1]):
            if (j % 2 == 0):
                frames_up[i][j] = 2 * frames[i][j // 2]
            else:
                continue
    # 滤波
    xi_hpf = signal.filtfilt(b_hpf, a_hpf, xi_all)  # 高通滤波器，宽带信号高通滤波
    # xi_lpf = signal.filtfilt(b_lpf,a_lpf,real)
    xi_up2 = signal.filtfilt(b_lpf, a_lpf, frames_up)  # 低通滤波器，窄带信号上采样后低通滤波
    # 估计的xi_all的高频+原始frames_up的低频
    real_part = (xi_hpf + xi_up2) * win_WB
    # real_part = (xi_hpf + xi_up2)
    # ==========================================================================
    # 合成语音
    speech = np.zeros((1, int(real_part.size / 2)))
    for i in range(1, int(real_part.size / frame_length)):
        speech[0, int((i - 1) * inc):int(inc * 2 + (i - 1) * inc)] = \
            speech[0, int((i - 1) * inc):int(inc * 2 + (i - 1) * inc)] + \
            real_part[i - 1, :]
    # ==========================================================================
    # 消除窗的增益
    ham = np.hamming(frame_length)
    # 窗为正态形状，在两边会有少部分需要平滑的区域，此处计算需要平滑的范围
    hamwin = np.zeros((1, int(real_part.size / 2)))

    for i in range(1, int(real_part.size / frame_length)):
        hamwin[0, int((i - 1) * frame_length / 2):int(frame_length + (i - 1) * frame_length / 2)] = \
            hamwin[0, int((i - 1) * frame_length / 2):int(frame_length + (i - 1) * frame_length / 2)] + \
            ham
    h = hamwin[0]
    speech[0, :] = [0 if h[i] == 0 else speech[0, i] / h[i] for i in range(0, speech.size)]

    return speech[0]


def waviowrite(speech, write_wavfile_name, sampwidth, framerate, nchannels):
    # wavio.write(write_wavfile_name, speech, framerate, sampwidth=2)  # 该方法写的wav文件有偏移
    outwave = wave.open(write_wavfile_name, 'wb')
    data_size = len(speech)
    nframes = data_size
    comptype = "NONE"
    compname = "not compressed"
    outwave.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
    for v in speech:
        outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))  # outData:16位，-32767~32767，注意不要溢出
    outwave.close()
    return None


def wave_restruct(phase, frames, magnitude, filename):
    frame_length = 512  # 窗长
    frame_step = frame_length / 2  # 窗的移动距离
    sampwidth = 2
    fs = 16000
    framerate = int(fs)
    nchannels = 1

    N_length = phase.shape[1]
    # 通过窄带音频相位反转得到宽带音频相位
    # (N,256) ---> (N,512)
    # phase_predict_ = np.hstack((phase[:,0:N_length],phase[:,0:1]))
    # phase_predict = np.hstack((phase_predict_,-np.fliplr(phase[:,1:N_length])))
    # (N,129) ---> (N,257)
    phase_predict = np.hstack((phase[:, 0:N_length], -np.fliplr(phase[:, 1:N_length])))

    # ==========================================================================
    # 还原语音
    speech = restore_speech(magnitude, phase_predict, frame_step, frames, frame_length)
    # ==========================================================================
    # 写数据到.wav文件

    waviowrite(speech, filename, sampwidth, framerate, nchannels)


# if __name__ == "__main__":
#     with keras.utils.custom_object_scope({"dft_matrix": dft_matrix, "after_fft_layer": after_fft_layer, "before_fft_layer": before_fft_layer}):
#         model = keras.models.load_model('model_dnn_test.h5')
#     data, nch, samp, fr = waveread('D:/Python_Workspace/ABE/TIMIT/test_8k_wav/test_files0001.wav')
#     frames, win = enframe(data)
#     _, phase = spectrum_magnitude(frames)
#     frames_normal = (frames - X_mean) / X_std
#     pre = model.predict(frames_normal)
#     pre = pre * Y_std + Y_mean
#
#     N_length = phase.shape[1]
#     phase_predict = np.hstack((phase[:, 0:N_length], -np.fliplr(phase[:, 1:N_length])))
#     speech = restore_speech(pre, phase_predict, 256, frames, 512)
#     waviowrite(speech, 'out.wav', 2, 16000, 1)

