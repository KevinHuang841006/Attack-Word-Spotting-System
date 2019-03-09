import argparse 
import tensorflow as tf
import numpy as np
import os
import NN
import scipy.io.wavfile as wav

## Load Data
datas = []
labels = []
c = []

def load_labels(filename):
    return [line.rstrip() for line in tf.gfile.FastGFile(filename)]   

def load_audiofile(filename):
    rrr, ddd = wav.read(filename)
    return rrr, ddd    
    
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
                    r, d = load_audiofile(data_dir+'/'+input_file)
                    r1, d1 = load_audiofile(data_dir1+'/'+input_file1)
                    #print(len(d))
                    if len(d)==16000:
                        #print(len(d)," ",d[1000])
                        datas.append(d)
                        label_to_atk = "left"
                        l = labels_dict.index(label_to_atk)
                        ll = [0,0,0,0,0,0,0,0,0,0]
                        ll[1] = 1
                        labels.append(ll)
                        c.append([1000])
        break

c = np.array(c)
datas = np.array(datas)
labels = np.array(labels)
datas = datas / 32767
#labels = labels / 32767
print("num of data--------------")
print(len(datas))
print("-------------------------")

## Load NN
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="ckpts/frozen_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()
    #加载已经将参数固化后的图
    #graph = load_graph(args.frozen_model_filename)
    nn = NN.my_NN("nn", session = sess,data=datas,ans=labels)
    nn.build_model()
    new_input = nn.predictions + nn.input_x
    
    sess.run(tf.initialize_all_variables())
    
    first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    saver = tf.train.Saver()
    new_saver = tf.train.import_meta_graph('ckpts/cnn_model.meta',input_map={'Placeholder:0':  new_input})
    new_saver.restore(sess, 'ckpts/cnn_model')
    
    ## input placeholder
    #output = tf.get_default_graph().get_tensor_by_name('fc1/BiasAdd:0')
    
    
    ## loss1 : norm2 loss
    loss1 = tf.reduce_sum(tf.square(nn.predictions),  1 )
    
    output = tf.get_default_graph().get_tensor_by_name('fc1/BiasAdd:0')
    placeholder_y = tf.placeholder(tf.float32, shape=( None , 10), name='placeholder_y')
    
    Const = tf.placeholder(tf.float32, shape=( None , 1), name='constant')
    
    real = tf.reduce_sum( placeholder_y * output, 1)
    other = tf.reduce_max( (1 - placeholder_y) * output - placeholder_y * 10000, 1)
    
    ## loss2 : Base on confidence score
    loss2 = tf.maximum( np.asarray(0., dtype=np.dtype('float32')) , other - real )
    loss = loss2 * Const + loss1
    loss = tf.reduce_sum(loss, name="tot_loss")
    
    
    #optimizer = tf.train.AdamOptimizer(0.01)
    trains = tf.train.AdamOptimizer(0.01).minimize( loss, var_list=first_train_vars, name="trains" )
    
    print("----------")
    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)
    print("----------")
    
    sess.run(tf.global_variables_initializer())
    #init_m = tf.variables_initializer(var_list=first_train_vars )
    #sess.run(init_m)
    
    new_saver.restore(sess, 'ckpts/cnn_model')
    
    for iter in range(30):
        save_path_name = "new_ckpts/" + str(iter+1) + "_batch30_model.ckpt"
        tot_l = 0
        t_l = 0
        tt=0
        for i in range(int(len(datas)/30)):
            _, l1, l2, l3  =  sess.run([trains,loss1,loss2,loss], feed_dict={nn.input_x: datas[i*30:(i+1)*30], placeholder_y: labels[i*30:30*(i+1)], Const:c[i*30:30*(i+1)]})
            tot_l = tot_l + l1
            t_l =t_l + l2
            tt = tt + l3
        print("iter: ",iter,"  loss1: ",tt/(len(datas)))
        
    
    
    
    
    
