import numpy as np
import tensorflow as tf

class my_NN():
    def __init__(self, name, in_size = 16000, out_size = 16000, session  = None, lr = 0.001, batch_size = 30, data = None, ans = None):
        self.model_name = name
        self.input_x = tf.placeholder(tf.float32, [None, in_size], name="input_x")
        self.output_y = tf.placeholder(tf.float32, [None, out_size], name="output_y")
        self.keep_rate = tf.placeholder(tf.float32, [], name="keep_rate")
        self.sess = session
        self.lr = lr
        self.batch_size = batch_size
        self.ckpt_dir = 'NN/'
        self.data = data
        self.ans = ans
        
        append_size = self.batch_size - len(data)%self.batch_size
        self.num_batches = int(len(data)/self.batch_size)
        if(append_size!=0):
            for i in range(append_size):
                np.append(data,data[i])
                np.append(ans,ans[i])

    def build_model(self):
        hidden = tf.layers.dense(self.input_x, 16000)
        hidden = tf.layers.dense(hidden, 400)
        # Add dropout
        with tf.variable_scope("dropout"):
            hidden = tf.layers.dropout(hidden, 1 - self.keep_rate)
        hidden = tf.layers.dense(hidden, 16000)
        self.predictions = tf.nn.tanh(hidden)
        
        self.loss = tf.nn.l2_loss(self.predictions - self.output_y)
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    def train(self, num_epoch = 50):
        
        for i in range(num_epoch):
            loss = []
            for j in range(self.num_batches):
                l, _ = self.sess.run([self.loss, self.train_op], feed_dict = {self.input_x:self.data[j*self.batch_size:(j+1)*self.batch_size],
                                                                              self.output_y:self.ans[j*self.batch_size:(j+1)*self.batch_size]})
                loss.append(l)
            loss_this_epoch = np.sum(loss)
            print('epoch : {}, loss = {}'.format( i, loss_this_epoch))

            if i%20==0:
                self.save()
    
    def save(self):
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, self.ckpt_dir + '/evaluate_model')
        
    def restore(self):
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name +'/'))
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        self.saver.restore(self.sess, self.ckpt_dir + '/evaluate_model')

        
