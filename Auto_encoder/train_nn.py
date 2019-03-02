import os
import sys

import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from random import shuffle

import NN




def load_labels(filename):
    return [line.rstrip() for line in tf.gfile.FastGFile(filename)]

def load_audiofile(filename):
    _, ddd = wav.read(filename)
    return ddd
    #with open(filename, 'rb') as fh:
        #return fh.read()
    
if __name__ == '__main__':    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    
    
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    
    
    
    
    # load data, label
    # datas = np.zeros((3000,16000))
    datas = []
    labels = []
    # labels = np.zeros((3000,16000))
    
    labels_path="ckpts/action_labels.txt"
    labels_dict = load_labels(labels_path)
    for label in labels_dict:
        if label == 'yes':
            print(label)
            
            data_dir = "adv_datas/yes/"+label
            wav_files_list =\
            [f for f in os.listdir(data_dir) if f.endswith(".wav")]
            
            data_dir1 = "adv_datas/yes/no"
            wav_files_list1 =\
            [f for f in os.listdir(data_dir1) if f.endswith(".wav")]
            
            
            for input_file in wav_files_list:
                #print(data_dir+'/'+input_file)
                for input_file1 in wav_files_list1:
                    if input_file==input_file1:
                        d = load_audiofile(data_dir+'/'+input_file)
                        d1 = load_audiofile(data_dir1+'/'+input_file1)
                        if len(d)==16000:
                            #print(len(d)," ",d[1000])
                            datas.append(d)
                            labels.append(d1)
            break
    
    
    datas = np.array(datas)
    labels = np.array(labels)
    datas = datas / 32767
    labels = labels / 32767
    """
    for i in datas[0]:
        if i > 1 or i < -1 :
            print("preprocessing err~~~~")
    """
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
    #trainX = datas[0:1500]
    #trainY = labels[0:1500]
    #testX = datas[1500:]
    #testY = labels[1500:]
    
    
    # initialize_all_variables
    nn = NN.my_NN("nn", session = sess ,data=datas,ans=labels)
    nn.build_model()
    init = tf.global_variables_initializer()
    sess.run(init)
    #for n in tf.get_default_graph().as_graph_def().node:
    #    print(n.name)
    # Add ops to save and restore all the variables.
    input_x = tf.placeholder(tf.float32, [None, 16000], name="input_x")
    stfts = tf.contrib.signal.stft(input_x, frame_length=1000, 
                                 frame_step=250,fft_length=1000)
    #nn.train()
    inverse_stft = tf.contrib.signal.inverse_stft(
                    stfts, frame_length=1000, frame_step=250,
                    window_fn=tf.contrib.signal.inverse_stft_window_fn(250))
    
    result = sess.run(inverse_stft, feed_dict={input_x:datas[0:10]})
    result = np.array(result)
    print(result.shape)
    for i in range(10):
        err = 0
        for j in range(16000):
            err = max(abs(datas[i][j]-result[i][j]),err)
        print("err: ",err)
