from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
import keras
from keras import layers, Input, models, Model
from keras.layers import Lambda
from keras.models import load_model
from preprocess.load_data import load_train_data
from preprocess.Extraction_of_speech_features import spectrum_magnitude
from keras.utils import plot_model
import numpy as np
import scipy
from preprocess.Extraction_of_speech_features import waveread, enframe, spectrum_magnitude
# from model.eval import restore_speech, waviowrite
import h5py


PRELU = layers.PReLU()

def weights_mag(inputs):
    time, freq = inputs
    w1 = tf.constant(0.8, dtype=tf.float32)
    w2 = tf.constant(0.2, dtype=tf.float32)
    return tf.add(tf.multiply(time, w1), tf.multiply(freq, w2))


def before_time_to_freq(x): # [256]
    # (num, dim) = inputs.shape
    # W = fft_matrix(dim)
    # fm = tf.constant(W, tf.float32)
    # tensor_w256 = tf.convert_to_tensor(w256, dtype=tf.complex64)
    # x = tf.cast(x, tf.complex64)
    # print(x.shape)
    # x = tf.reshape(x, (x.shape[1], x.shape[0]))
    # print(x.shape, tensor_w256)
    # x = tf.multiply(tensor_w256, x)
    # x_mag = tf.cast(x, tf.float32)
    # # x = x.real()
    # return x_mag
    real, imag = dft_matrix(256)
    real = tf.convert_to_tensor(real, tf.float32)
    imag = tf.convert_to_tensor(imag, tf.float32)
    rxs = tf.matmul(real, tf.transpose(x))
    ixs = tf.matmul(imag, tf.transpose(x))
    rxs = tf.transpose(rxs)
    ixs = tf.transpose(ixs)
    complex_spec = tf.complex(rxs, ixs)
    mag = tf.log(tf.square(tf.abs(complex_spec)))
    phase = tf.angle(complex_spec)
    return mag


def after_time_to_freq(x): # [512]
    # tensor_w512 = tf.convert_to_tensor(w512, dtype=tf.complex64)
    # x = tf.cast(x, tf.complex64)
    # x = tf.reshape(x, (x.shape[1], x.shape[0]))
    # x = tf.multiply(tensor_w512, x)
    # x_mag = tf.cast(x, tf.float32)
    # # x = x.real()
    # return x_mag
    real, imag = dft_matrix(512)
    real = tf.convert_to_tensor(real, tf.float32)
    imag = tf.convert_to_tensor(imag, tf.float32)
    rxs = tf.matmul(real, tf.transpose(x))
    ixs = tf.matmul(imag, tf.transpose(x))
    rxs = tf.transpose(rxs)
    ixs = tf.transpose(ixs)
    complex_spec = tf.complex(rxs, ixs)
    mag = tf.log(tf.square(tf.abs(complex_spec)))
    phase = tf.angle(complex_spec)
    return mag


# time_fft_layer = Lambda(tf_fft)
# freq_fft_layer = Lambda(tf_fft)
after_fft_layer = Lambda(after_time_to_freq)
before_fft_layer = Lambda(before_time_to_freq)
weights_layer = Lambda(weights_mag)


def dft_matrix(NFFT):
    """
    DFT.dot(x) 等价于 FFT(x)
    :param NFFT:
    :return: dft matrix, real of it, imag of it
    """
    D = scipy.linalg.dft(NFFT)
    real = D.real
    imag = D.imag
    return real, imag

class DFT(tf.keras.layers.Layer):
    def __init__(self, NFFT):
        super(DFT, self).__init__()
        self.NFFT = NFFT
        self.real, self.imag = dft_matrix(self.NFFT)
        self.m = None
        self.p = None

    def call(self, inputs, **kwargs):
        self.m, self.p = self.dft(inputs)
        return self.m

    def dft(self, inputs):
        """
        :param inputs:
        :param NFFT:
        :return: mag, phase
        """
        real = tf.convert_to_tensor(self.real, dtype=tf.float32)
        imag = tf.convert_to_tensor(self.imag, dtype=tf.float32)
        rxs = tf.matmul(real, tf.transpose(inputs))
        ixs = tf.matmul(imag, tf.transpose(inputs))
        print(rxs, ixs)
        # rxs, ixs = [], []
        # for temp in inputs[:]:
        #     temp = tf.reshape(temp, (self.NFFT, -1))
        #     rxx, ixx = tf.matmul(real, temp), tf.matmul(imag, temp)
        #     rxs.append(rxx)
        #     ixs.append(ixx)
        # rxs = tf.reshape(rxs, (-1, self.NFFT))
        # ixs = tf.reshape(ixs, (-1, self.NFFT))
        rxs = tf.transpose(rxs)
        ixs = tf.transpose(ixs)
        print(rxs, ixs)
        complex_spec = tf.complex(rxs, ixs)
        mag = tf.log(tf.square(tf.abs(complex_spec)))
        phase = tf.angle(complex_spec)
        return mag, phase


def fcnet(name):
    with tf.name_scope(name):
        inputs = Input(shape=(256,))
        x = layers.Dense(1024, activation='selu')(inputs)
        x = layers.Dense(1024, activation='selu')(x)
        outputs = layers.Dense(512, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
    return model


def net():
    inputs = Input(shape=(256,))

    # time
    time_outputs = fcnet('time_net')(inputs)  # (None, 256)-->(None, 512)
    time_outputs_mag = after_fft_layer(time_outputs)

    # dft_layer1 = DFT(512)
    # time_outputs_mag = dft_layer1(time_outputs)  # (None, 512)-->(None, 512)



    # freq
    # dft_layer2 = DFT(256)
    # freq_inputs_mag = dft_layer2(inputs)  # (None, 256)-->(None, 256)
    freq_inputs_mag = before_fft_layer(inputs)
    freq_outputs_mag = fcnet('freq_net')(freq_inputs_mag)  # (None, 256)-->(None, 512)


    outputs = weights_layer([time_outputs_mag, freq_outputs_mag])  # 加权后的mag
    model = Model(inputs=inputs, outputs=outputs)
    optimze = keras.optimizers.Adam(lr=0.001, decay=1e-05)
    model.compile(optimizer=optimze, loss='mse', metrics=['mae'])

    return model

def decay():
    pass

if __name__ == '__main__':
    # ================ net ======================================================
    trainx, trainy = load_train_data()
    # print(trainx.shape, trainy.shape)
    trainy, _ = spectrum_magnitude(trainy)
    model = net()
#    plot_model(model, to_file='tf_model.png', show_shapes=True)
    model.summary()
    model.fit(trainx, trainy, batch_size=64, epochs=1000, verbose=2, validation_split=0.2, callbacks=[
        keras.callbacks.ModelCheckpoint('./model_dnn_test.h5', monitor='val_loss', verbose=2, save_best_only=True, mode='auto')])

    # ================ eval ====================================================
    # file = h5py.File('params.h5', 'r')
    # X_mean = file['x_mean']
    # X_std = file['x_std']
    # Y_mean = file['y_mean']
    # Y_std = file['y_std']
    # with keras.utils.custom_object_scope({"dft_matrix": dft_matrix, "after_fft_layer": after_fft_layer, "before_fft_layer": before_fft_layer, "tf": tf}):
    #     model = load_model('model_dnn_test.h5')
    # model.summary()
    # data, nch, samp, fr = waveread('D:/Python_Workspace/ABE/TIMIT/test_8k_wav/test_files0001.wav')
    # frames, win = enframe(data, 256, 128, winfunc=np.hamming)
    # frames_normal = (frames - X_mean) / X_std
    # pre = model.predict(frames_normal)
    # print(model.eval(frames_normal))
    # pre = pre * Y_std + Y_mean
    # # 先提供一个wb的phase
    # data2, nch2, samp2, fr2 = waveread('D:/Python_Workspace/ABE/TIMIT/test_16k_wav/test_files0001.wav')
    # frames2, win2 = enframe(data2, 512, 256, winfunc=np.hamming)
    # _, phase = spectrum_magnitude(frames2)
    # print('reconstruct wave')
    # speech = restore_speech(pre, phase, 256, frames, 512)
    # waviowrite(speech, 'out.wav', 2, 16000, 1)
    # print('--done--')
    # ============ DFT.dot(x) == FFT(x) =========================================
    # x = np.array([i for i in range(1, 6)])
    # print(x.shape)
    # r, i = dft_matrix(5)
    # print(r.dot(x), i.dot(x))
    # print(scipy.fftpack.fft(x))
    # ============ TF-DFT =======================================================
    # x = tf.convert_to_tensor(x, dtype=tf.float32)
    # x_ = tf.reshape(x, (5, -1))
    # print(x.shape)
    # real, imag = dft_matrix(5)
    # real = tf.convert_to_tensor(real, dtype=tf.float32)
    # imag = tf.convert_to_tensor(imag, dtype=tf.float32)
    # rxx, ixx = tf.matmul(real, x_), tf.matmul(imag, x_)
    # with tf.Session() as sess:
    #     rx, ix = sess.run([rxx, ixx])
    # print(rx, ix)
