"""

Model architecture :

input audio : 16000
rfft        : 8001
dense       : 16002   
dense       : 400
dense       : 16002
irfft       : 16000

"""
import numpy as np
import tensorflow as tf


class my_NN():
    def __init__(self, name, in_size = 16000, out_size = 16000, session  = None, lr = 0.01, batch_size = 30, data = None, ans = None, ckpt_d = None):
        self.model_name = name
        self.input_x = tf.placeholder(tf.float32, [None, in_size], name="input_x")
        self.output_y = tf.placeholder(tf.float32, [None, out_size], name="output_y")
        self.keep_rate = tf.placeholder(tf.float32, [], name="keep_rate")
        self.sess = session
        self.lr = lr
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_d
        self.data = data
        self.ans = ans
        
        append_size = self.batch_size - len(data)%self.batch_size
        self.num_batches = int(len(data)/self.batch_size)
        """
        if(append_size!=0):
            for i in range(append_size):
                np.append(data,data[i])
                np.append(ans,ans[i])
        """
        
    def build_model(self):
        rfft = tf.spectral.rfft(self.input_x)
        #input_rfft = tf.stack( [ tf.real(rfft), tf.imag(rfft) ], 2 )
        input_rfft_r = tf.real(rfft)
        input_rfft_i = tf.imag(rfft)
        
        #print("ft shape: ", str(input_rfft))
        #input_rfft = tf.reshape( input_rfft , shape=(tf.shape(input_rfft)[0], 8001*2 ))
        hidden1 = tf.layers.dense( input_rfft_r , 8001, activation=tf.nn.elu, name="h_1_1")
        hidden2 = tf.layers.dense( input_rfft_i , 8001, activation=tf.nn.elu, name="h_1_2")
        #print("hidden: ",str(hidden))
        hidden1 = tf.layers.dense(hidden1, 200, activation=tf.nn.tanh, name="h_2_1")
        hidden2 = tf.layers.dense(hidden2, 200, activation=tf.nn.tanh, name="h_2_2")
        # Add dropout
        #with tf.variable_scope("dropout"):
        #    hidden = tf.layers.dropout(hidden, 1 - self.keep_rate)
        #hidden = tf.layers.dense(hidden, 16002, activation=tf.nn.sigmoid)
        hidden1 = tf.layers.dense(hidden1, 8001, name="h_3_1")
        hidden2 = tf.layers.dense(hidden2, 8001, name="h_3_2")
        hidden1 = tf.nn.sigmoid(hidden1, name="s1")
        hidden2 = tf.nn.sigmoid(hidden2, name="s2")
        #print("last hidden: ",str(hidden))
        #split0, split1 = tf.split(hidden, num_or_size_splits=2, axis=1)
        #print("split0: ",str(split0))
        #print("split1: ",str(split1))
        real = tf.multiply(tf.real( rfft ), hidden1)
        imag = tf.multiply(tf.imag( rfft ), hidden2)
        self.predictions = tf.spectral.irfft(tf.complex(real, imag))
        #self.predictions = tf.nn.tanh( network ,name="conf_score")
        print("predictions: ", str(self.predictions) )
        #self.loss = tf.nn.l2_loss(self.predictions - self.input_x)
        #self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def train(self, num_epoch = 3000):
        saver = tf.train.Saver()
        for i in range(num_epoch):
            loss = []
            for j in range(self.num_batches):
                l, _ = self.sess.run([self.loss, self.train_op], feed_dict = {self.input_x:self.data[j*self.batch_size:(j+1)*self.batch_size],
                                                                              self.output_y:self.ans[j*self.batch_size:(j+1)*self.batch_size]})
                loss.append(l)
            loss_this_epoch = np.sum(loss)
            print('epoch : {}, loss = {}'.format( i, loss_this_epoch/len(self.data)))
            if i%200==0:
                saver.save( self.sess, "fft_checkpoint/fft_model" )
            
    
    def save(self):
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, self.ckpt_dir + 'evaluate_model')
        
    def restore(self):
        #self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name +'/'))
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        #ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        self.saver.restore(self.sess, "fft_checkpoint/fft_model" )

        
#trainX = []
#trainY = []

#

#sess = tf.Session()
#nn = my_NN("nn", session = sess,data=trainX,ans=trainY)
#nn.build_model()


