import sys
import simpleaudio as sa
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QVBoxLayout, QFormLayout, QLabel
from PyQt5.QtMultimedia import QSound
from PyQt5 import QtGui

import os
import math

import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from random import shuffle
from scipy.signal import butter, lfilter
#from subprocess import call
import record as recorder

"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
"""
sess = tf.Session()

#saver = tf.train.Saver()
saver = tf.train.import_meta_graph('ckpts/model3.meta')
saver.restore(sess, 'ckpts/model3')

logit = tf.get_default_graph().get_tensor_by_name("dense/BiasAdd:0")
predict = tf.get_default_graph().get_tensor_by_name("Softmax:0")
place_hold = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
label_y =  tf.placeholder(tf.float32, shape=( 30 , 10) )
loss = tf.losses.softmax_cross_entropy(logits=predict, onehot_labels=label_y)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
grad = tf.gradients(loss, place_hold)
grad = tf.sign(grad)
grad = grad * 0.0001
adv = grad + place_hold
adv = tf.clip_by_value(adv,-1,1,name="adv_name")

def load_labels(filename):
    return [line.rstrip() for line in tf.gfile.FastGFile(filename)]

labels_dict = []
labels_path="ckpts/action_labels.txt"
labels_dict = load_labels(labels_path)

class MainWindow(QWidget):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi()
        self.show()

    def setupUi(self):
        self.setWindowTitle("DEMO~~~~")

        self.button_hello = QPushButton()
        self.button_hello.setText("Start~")
        self.button_hello.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        
        self.button_cancel = QPushButton()
        self.button_cancel.setText("cancel")
        self.button_cancel.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        
        self.line_hello = QLineEdit()
        self.line_hello.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        
        ## predict label & text box
        self.label1 = QLabel()
        self.label1.setText("yes ")
        self.label1.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.label2 = QLabel("no ")
        self.label2.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.label3 = QLabel("up ")
        self.label3.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.label4 = QLabel("down ")
        self.label4.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.label5 = QLabel("left ")
        self.label5.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.label6 = QLabel("right ")
        self.label6.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.label7 = QLabel("on ")
        self.label7.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.label8 = QLabel("off ")
        self.label8.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.label9 = QLabel("stop ")
        self.label9.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.label10 = QLabel("go ")
        self.label10.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.labelfinal = QLabel("Result ")
        self.labelfinal.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.text1 = QLineEdit()
        self.text1.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.text2 = QLineEdit()
        self.text2.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.text3 = QLineEdit()
        self.text3.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.text4 = QLineEdit()
        self.text4.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.text5 = QLineEdit()
        self.text5.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.text6 = QLineEdit()
        self.text6.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.text7 = QLineEdit()
        self.text7.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.text8 = QLineEdit()
        self.text8.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.text9 = QLineEdit()
        self.text9.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.text10 = QLineEdit()
        self.text10.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        self.textfinal = QLineEdit()
        self.textfinal.setFont(QtGui.QFont("Times",20,QtGui.QFont.Bold))
        
        form_layout = QFormLayout()
        form_layout.addRow(self.button_hello, self.line_hello)
        ## predict label
        form_layout.addRow(self.label1, self.text1)
        form_layout.addRow(self.label2, self.text2)
        form_layout.addRow(self.label3, self.text3)
        form_layout.addRow(self.label4, self.text4)
        form_layout.addRow(self.label5, self.text5)
        form_layout.addRow(self.label6, self.text6)
        form_layout.addRow(self.label7, self.text7)
        form_layout.addRow(self.label8, self.text8)
        form_layout.addRow(self.label9, self.text9)
        form_layout.addRow(self.label10, self.text10)
        form_layout.addRow(self.labelfinal, self.textfinal)
        
        form_layout.addRow(self.button_cancel)

        h_layout = QVBoxLayout()
        h_layout.addLayout(form_layout)

        self.setLayout(h_layout)

        self.button_hello.clicked.connect(self.hello)
        self.button_cancel.clicked.connect(self.cancel)

    def hello(self):
        #self.line_hello.setText("hello 我被觸發了")
        #file_name = self.line_hello.text()
        recorder.Record_audio()
        #sound = QSound(file_name)
        #sound.play()
        #call(["aplay", file_name])
        #wave = sa.WaveObject.from_wave_file("output.wav")
        #play = wave.play()
        
        datas = []
        rate, data = wav.read("output.wav")
        data = list(data)
        while len(data)!=16000:
            data.append(0)
        data = np.array(data)
        print("rate: ",rate)
        print("data shape: ",data.shape)
        for qqq in range(30):
            datas.append(data)
        datas = np.array(datas)
        datas = datas / 32767
        
        pred = sess.run(predict, feed_dict={place_hold:datas} )
        pred = np.array(pred)
        
        
        cur_target = np.argmax(pred)
        print("cur: ",cur_target," name: ",labels_dict[cur_target])
        self.line_hello.setText( labels_dict[cur_target] )
        
        
        input_labels = np.zeros((30,10))
        for k1 in range(30):
            input_labels[k1][ cur_target ] = 1
        
        adv_data_t = sess.run(adv, feed_dict={place_hold: datas, label_y: input_labels })
        adv_data_t = adv_data_t[0]
        
        for cow in range(10):
            adv_data_t = sess.run(adv, feed_dict={place_hold: adv_data_t, label_y: input_labels })
            adv_data_t = adv_data_t[0]
        
        pred = sess.run(predict, feed_dict={place_hold:adv_data_t} )
        pred = np.array(pred)
        
        self.text1.setText( str( round(pred[0][0], 6) ) )
        self.text2.setText( str( round(pred[0][1], 6) ) )
        self.text3.setText( str( round(pred[0][2], 6) ) )
        self.text4.setText( str( round(pred[0][3], 6) ) )
        self.text5.setText( str( round(pred[0][4], 6) ) )
        self.text6.setText( str( round(pred[0][5], 6) ) )
        self.text7.setText( str( round(pred[0][6], 6) ) )
        self.text8.setText( str( round(pred[0][7], 6) ) )
        self.text9.setText( str( round(pred[0][8], 6) ) )
        self.text10.setText( str( round(pred[0][9], 6) ) )
        self.textfinal.setText( labels_dict[ np.argmax(pred[0]) ] )
        #print(file_name)
        adv_data_t = np.clip(adv_data_t, -1, 1)
        adv_data_t = adv_data_t*32767
        adv_data_t = np.array(adv_data_t, dtype=np.int16)
        print(adv_data_t.shape)
        wav.write("iter_fgsm.wav" ,16000 , adv_data_t[0])
        wave = sa.WaveObject.from_wave_file("iter_fgsm.wav")
        play = wave.play()
        
    def cancel(self):
        QApplication.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    sys.exit(app.exec_())
