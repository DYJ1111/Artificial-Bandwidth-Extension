import wave
import numpy as np
import math
# import matplotlib.pyplot as plt
import scipy.signal as signal
import struct
import h5py


def waveread(filename):
    '''提取语音信号域基本特征'''
    f = wave.open(filename, 'rb')
    params = f.getparams()
    # 声道数, 量化位数（byte单位）, 采样频率, 采样点数
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)
    waveData = np.fromstring(strData, dtype=np.int16)
    waveData = waveData * 1.0 / (pow(2, 15))
    waveData = np.reshape(waveData, [nframes, nchannels]).T
    return waveData[0], nchannels, sampwidth, framerate


def enframe(signal, frame_length, frame_step, winfunc=lambda x: np.ones((x,))):
    '''将音频信号转化为帧。
    	参数含义：
    	signal:原始音频型号
    	frame_length:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    	frame_step:相邻帧的间隔（同上定义）
    	winfunc:lambda函数，用于生成一个向量
        '''
    # 分帧
    signal_length = len(signal)  # 信号总长度
    frame_length = int(round(frame_length))  # 以帧帧时间长度
    frame_step = int(round(frame_step))  # 相邻帧之间的步长
    if signal_length <= frame_length:  # 若信号长度小于一个帧的长度，则帧数定义为1
        frames_num = 1
    else:  # 否则，计算帧的总长度
        frames_num = 1 + int(math.ceil((1.0 * signal_length - frame_length) / frame_step))
    pad_length = int((frames_num - 1) * frame_step + frame_length)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((signal, zeros))  # 填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, frame_length), (frames_num, 1)) + np.tile(
        np.arange(0, frames_num * frame_step, frame_step),
        (frame_length, 1)).T  # 相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    # 加窗
    win = np.tile(winfunc(frame_length), (frames_num, 1))  # window窗函数，这里默认取1
    ham = winfunc(frame_length)
    hamwin = np.zeros((1, int(math.floor(pad_length / frame_length) * frame_length)))
    for i in range(1, frames_num):
        hamwin[0, int((i - 1) * frame_length / 2):int(frame_length + (i - 1) * frame_length / 2)] = \
            hamwin[0, int((i - 1) * frame_length / 2):int(frame_length + (i - 1) * frame_length / 2)] + \
            ham
    return frames * win, hamwin[0]  # 返回帧信号矩阵


def spectrum_magnitude(frames):
    '''计算每一帧经过FF变幻以后的频谱的幅度
    参数说明：
    frames:即audio2frame函数中的返回值矩阵，帧矩阵
    '''
    complex_spectrum = np.fft.fft(frames)  # 对frames进行FFT变换
    magnitude = np.log(np.square(np.abs(complex_spectrum)))
    phase = np.angle(complex_spectrum)
    return magnitude, phase  # 返回频谱的幅度值，相位


def complex_real_imag(frames):
    complex_spectrum = np.fft.rfft(frames)
    complex_real = complex_spectrum.real
    complex_imag = complex_spectrum.imag

    return complex_real, complex_imag


def SMM_calculation(speech_frames, noise_frames):
    s_complex_spectrum = np.fft.rfft(speech_frames)
    n_complex_spectrum = np.fft.rfft(noise_frames)
    s_energy = np.abs(s_complex_spectrum)
    n_energy = np.abs(n_complex_spectrum)
    # IRM = ( S^2 / (S^2 + N^2) ) ^ 1/2
    SMM = np.true_divide(s_energy, n_energy)
    return SMM


def IRM_calculation(speech_frames, noise_frames):
    s_complex_spectrum = np.fft.rfft(speech_frames)
    n_complex_spectrum = np.fft.rfft(noise_frames)
    s_energy = np.square(np.abs(s_complex_spectrum))
    n_energy = np.square(np.abs(n_complex_spectrum))
    # IRM = ( S^2 / (S^2 + N^2) ) ^ 1/2
    IRM = np.sqrt(np.true_divide(s_energy, np.add(s_energy, n_energy)))
    return IRM


def restore_speech(magnitude, phase, inc):
    '''
    :param magnitude:  幅度
    :param phase: 相位
    :param inc:
    :param hamwin: 用于除去汉明窗的增益
    '''
    # =====幅度相位=====
    spectral = np.multiply(np.sqrt(np.exp(magnitude)), np.exp(np.multiply(complex(0, 1), phase)))
    # =====幅度相位=====
    # =====实部虚部组合复数=====
    # spectral = np.add(np.multiply(complex(1, 0), np.exp(magnitude)), np.multiply(complex(0, 1), np.exp(phase)))
    # =====实部虚部组合复数=====
    real_part = np.real(np.fft.ifft(spectral))

    # ===
    ham = np.hamming(int(inc * 2))
    # 窗为正态形状，在两边会有少部分需要平滑的区域，此处计算需要平滑的范围   
    hamwin = np.zeros((1, int(real_part.size / 2)))
    frame_length = int(inc * 2)
    for i in range(1, int(real_part.size / frame_length)):
        hamwin[0, int((i - 1) * frame_length / 2):int(frame_length + (i - 1) * frame_length / 2)] = \
            hamwin[0, int((i - 1) * frame_length / 2):int(frame_length + (i - 1) * frame_length / 2)] + \
            ham
    hamwin = hamwin[0]
    # ===

    speech = np.zeros((1, len(hamwin)))
    for i in range(1, int(speech.size / inc)):
        speech[0, int((i - 1) * inc):int(inc * 2 + (i - 1) * inc)] = \
            speech[0, int((i - 1) * inc):int(inc * 2 + (i - 1) * inc)] + \
            real_part[i - 1, :]
    speech[0, :] = [0 if hamwin[i] == 0 else speech[0, i] / hamwin[i] for i in range(0, speech.size)]
    return speech[0]


def upsample(frames):
    # 上采样：即内插0.......frames_up数据来自窄带frames
    frames_up = np.zeros((frames.shape[0], 2 * frames.shape[1]))
    for i in range(0, frames_up.shape[0]):
        for j in range(0, frames_up.shape[1]):
            if (j % 2 == 0):
                frames_up[i][j] = 2 * frames[i][j // 2]
            else:
                continue
    return frames_up


def restore_wideband_speech(magnitude, phase, inc, frames_up, frame_length):
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

    # frames_up = upsample(frames)
    # 滤波
    xi_hpf = signal.filtfilt(b_hpf, a_hpf, xi_all)  # 高通滤波器，宽带信号高通滤波
    # xi_lpf = signal.filtfilt(b_lpf,a_lpf,real)
    xi_up2 = signal.filtfilt(b_lpf, a_lpf, frames_up)  # 低通滤波器，窄带信号上采样后低通滤波
    # 估计的xi_all的[低频为0，高频]+原始frames_up的[低频,高频为0]
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


def wavewrite(speech, filepath, sampwidth, framerate, nchannels):  # 还原语音wav写入
    # wavio.write(filepath, speech, framerate, sampwidth=2)  # url:https://github.com/WarrenWeckesser/wavio
    outwave = wave.open(filepath, 'wb')
    data_size = len(speech)
    nframes = data_size
    comptype = "NONE"
    compname = "not compressed"
    outwave.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
    for v in speech:
        outwave.writeframes(struct.pack('h', int(v * 64000 / 2)))
    outwave.close()
    return None


def save_data(filepath, data):
    np.savetxt(filepath, data.astype(np.float32), delimiter=',')
    return None

# =============================================================================
# if __name__ == "__main__":
#     filename = "TIMIT/test_8k_wav/test_files0001.wav"
#     noisename = "noise.wav"
#     cleanname = "speech.wav"
#     wavedata, nchannels, sampwidth, framerate = waveread(filename)
#     winsize = 256
#     inc = winsize / 2
#     frames, hamwin = enframe(wavedata, winsize, inc, signal.hamming)
#     mag, phase = spectrum_magnitude(frames)
# =============================================================================
# =============================================================================
#     real, imag = complex_real_imag(frames)
#     file = h5py.File('com.h5', 'w')
#     file.create_dataset('real', data=real[:, 129:])
#     file.create_dataset('imag', data=imag[:, 129:])
#     file.close()
# =============================================================================
# zeros_array = np.zeros((1, imag.shape[1]))
# print(zeros_array.shape)
# imag = np.concatenate([zeros_array, imag, zeros_array], axis=0)
# print(imag.shape)

# speech = restore_speech(real, imag, inc)
# wavewrite(speech, 'complex.wav', sampwidth, framerate, nchannels)
# ===========
# complex_spectrum = np.fft.rfft(frames)  # 对frames进行FFT变换
# magnitude = np.abs(complex_spectrum)
# mix_mag,mix_phase = spectrum_magnitude(frames)
#
# c_wavedata, c_nchannels, c_sampwidth, c_framerate = waveread(cleanname)
# c_frames, hamwin = enframe(c_wavedata, winsize, inc, signal.hamming)
# c_m,c_phase = spectrum_magnitude(c_frames)
# c_spectrum = np.fft.rfft(frames)
# c_mag = np.abs(c_spectrum)
#
# n_wavedata, n_nchannels, n_sampwidth, n_framerate = waveread(noisename)
# n_frames, hamwin = enframe(n_wavedata, winsize, inc, signal.hamming)
# IRM = IRM_calculation(c_frames, n_frames)
# SMM = SMM_calculation(c_frames, frames)
#
#
# S_t = IRM * magnitude
#
# # logmag = np.log(S_t)
# # speech = retore_speech(logmag,mix_phase,inc,hamwin)
# mag = np.log(np.square(S_t))
# speech = restore_speech(mag,c_phase,inc,hamwin)
# wavewrite(speech,"1.wav",sampwidth,framerate,nchannels)
# #
# # s_wavedata, nchannels, sampwidth, framerate = waveread(cleanname)
# # s_frames, hamwin = enframe(s_wavedata, winsize, inc, signal.hamming)
# # s_mag, s_phase = spectrum_magnitude(s_frames)
# # print(s_mag)
