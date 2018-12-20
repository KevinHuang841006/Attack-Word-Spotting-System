import os
import sys

import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from random import shuffle


def compute_mfcc(audio, **kwargs):
    """
    Compute the MFCC for a given audio waveform. This is
    identical to how DeepSpeech does it, but does it all in
    TensorFlow so that we can differentiate through it.
    """

    batch_size, size = audio.get_shape().as_list()
    audio = tf.cast(audio, tf.float32)
    # 1. Pre-emphasizer, a high-pass filter
    audio = tf.concat((audio[:, :1], audio[:, 1:] - 0.97*audio[:, :-1], np.zeros((batch_size,1000),dtype=np.float32)), 1)
    # 2. windowing into frames of 320 samples, overlapping
    windowed = tf.stack([audio[:, i:i+400] for i in range(0,size-320,160)],1)
    # 3. Take the FFT to convert to frequency space
    ffted = tf.spectral.rfft(windowed, [512])
    ffted = 1.0 / 512 * tf.square(tf.abs(ffted))
    # 4. Compute the Mel windowing of the FFT
    energy = tf.reduce_sum(ffted,axis=2)+1e-30
    filters = np.load("filterbanks.npy").T
    feat = tf.matmul(ffted, np.array([filters]*batch_size,dtype=np.float32))+1e-30
    # 5. Take the DCT again, because why not
    feat = tf.log(feat)
    feat = tf.spectral.dct(feat, type=2, norm='ortho')[:,:,:26]
    # 6. Amplify high frequencies for some reason
    _,nframes,ncoeff = feat.get_shape().as_list()
    n = np.arange(ncoeff)
    lift = 1 + (22/2.)*np.sin(np.pi*n/22)
    feat = lift*feat
    width = feat.get_shape().as_list()[1]

    # 7. And now stick the energy next to the features
    feat = tf.concat((tf.reshape(tf.log(energy),(-1,width,1)), feat[:, :, 1:]), axis=2)
    
    return feat

def load_labels(filename):
    return [line.rstrip() for line in tf.gfile.FastGFile(filename)]

def load_audiofile(filename):
    _, ddd = wav.read(filename)
    return ddd
    #with open(filename, 'rb') as fh:
        #return fh.read()
    
if __name__ == '__main__':    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    
    
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    
    
    # make modle
    input =  tf.placeholder(tf.float32, shape=( 30 , 16000) )
    label_y =  tf.placeholder(tf.float32, shape=( 30 , 10) )
    mfcc = compute_mfcc(input)
    mfcc = tf.reshape(mfcc,shape=(30,1,98,26))
    conv1 =  tf.layers.conv2d(inputs=mfcc ,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    conv2 =  tf.layers.conv2d(inputs=conv1 ,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    conv3 =  tf.layers.conv2d(inputs=conv2 ,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    conv4 =  tf.layers.conv2d(inputs=conv3 ,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    print(conv4)
    fc1 = tf.contrib.layers.flatten(conv4)
    #fc1 = tf.layers.dense(fc1, 64, activation = tf.nn.relu)
    fc1 = tf.layers.dense(fc1, 10)
    predict = tf.nn.softmax(fc1)
    loss = tf.losses.softmax_cross_entropy(logits=predict, onehot_labels=label_y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc1, labels=label_y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
    
    # load data, label
    datas = []
    labels = []
    
    labels_path="ckpts/action_labels.txt"
    labels_dict = load_labels(labels_path)
    for label in labels_dict:
        print(label)
        data_dir = "data/"+label
        wav_files_list =\
        [f for f in os.listdir(data_dir) if f.endswith(".wav")]
        l = labels_dict.index(label)
        ll = [0,0,0,0,0,0,0,0,0,0]
        ll[l] = 1
        #print(wav_files_list[0])
        for input_file in wav_files_list:
            #print(data_dir+'/'+input_file)
            d = load_audiofile(data_dir+'/'+input_file)
            if len(d)==16000:
                #print(len(d)," ",d[1000])
                datas.append(d)
                labels.append(ll)
    # load_data();
    #encode_label = OneHotEncoder(sparse=False)
    #labels = encode_label.fit_transform(labels)
    
    datas = np.array(datas)
    labels = np.array(labels)
    #max_abs_sacler = preprocessing.MaxAbsScaler()
    #max_abs_sacler.fit(datas)
    #datas = max_abs_sacler.transform(datas)
    datas = datas / 32767
    for i in datas[0]:
        if i > 1 or i < -1 :
            print("preprocessing err~~~~")
    
    print(datas.shape)
    print(labels.shape)
    #print(datas[0][0])
    #from random import shuffle

    ind_list = [i for i in range( len(datas) )]
    shuffle(ind_list)
    datas  = datas[ind_list, :]
    labels = labels[ind_list,:]
    
    #datas, labels = shuffle(datas, labels)
    #datas = np.array(datas)
    #labels = np.array(labels)
    print(datas[0:10])
    print(labels[0:10])
    
    # seperate train and test
    trainX = datas[0:19200]
    trainY = labels[0:19200]
    testX = datas[19200:]
    testY = labels[19200:]
    
    
    # initialize_all_variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Add ops to save and restore all the variables.
    
    saver = tf.train.Saver()
    
    
    # train
    for j in range(100):
        save_path_name = "ckpts/" + str(j) + "_batch30_model.ckpt"
        print("iter: ",j," path name: ",save_path_name)
        for i in range(int(len(trainX)/30)):
            sess.run(train_step, feed_dict={input: trainX[i*30:(i+1)*30], label_y: trainY[i*30:30*(i+1)]})
        save_path = saver.save(sess, save_path_name)
    """
    ans = tf.argmax(input=predict, axis=1)
    tot = 0
    for i in range(len(testX)):
        a = sess.run(ans, feed_dict={input: testX[i:i+1]})
        if a[0] == test[i]:
            tot = tot + 1
    print("ans: ",tot/3312)        
    """
