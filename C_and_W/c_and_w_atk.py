import os
import sys

import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from random import shuffle
from scipy.signal import butter, lfilter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess2 = tf.Session(config=config)

# important~~~  when initialize_restore_graph
tf.reset_default_graph()

graph_def = tf.GraphDef()

def ZERO():
    return np.asarray(0., dtype=np.float32)


    
def load_labels(filename):
    return [line.rstrip() for line in tf.gfile.FastGFile(filename)]    

def load_audiofile(filename):
    rrr, ddd = wav.read(filename)
    return rrr, ddd
labels_dict = []
labels_path="ckpts/action_labels.txt"
labels_dict = load_labels(labels_path)
rates = []
datas = []
labels = []

#for i in range(10):
#    datas.append([])

def load_data():
    
    o2=0
    for label in labels_dict:
    
        rates1 = []
        datas1 = []
        labels1 = []
        
        
        
        print(label," ",labels_dict[o2])
        o2 = o2 + 1
        data_dir = "data/"+label
        wav_files_list =\
        [f for f in os.listdir(data_dir) if f.endswith(".wav")]
        l = labels_dict.index(label)
        ll = [0,0,0,0,0,0,0,0,0,0]
        ll[1] = 1
        #print(wav_files_list[0])
        in_file_count = 0
        for input_file in wav_files_list:
            
            if in_file_count == 201:
                break
            r, d = load_audiofile(data_dir+'/'+input_file)
            if len(d)==16000:
                in_file_count = in_file_count + 1
                #print(len(d)," ",d[1000])
                rates1.append(r)
                datas1.append(d)
                labels1.append(ll)
        datas1 = np.array(datas1)
        print(datas1.shape)
        datas.append(datas1)
        
load_data()
#def load_graph(filename):
datas = np.array(datas)
print("data shape: ",datas.shape)
datas = datas / 32767
#return new_input

filename = "ckpts/train2.pb"
## GPU Environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
#sess = tf.Session(config = config)


with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    #with tf.gfile.FastGFile( filename , 'rb') as f:
        
    #    graph_def.ParseFromString(f.read())
    #tf.import_graph_def(graph_def, name='')
    
    
    ## global varieble for tensorflow to optimaize    
    modifier = tf.Variable(np.zeros( (30, 16000) ,  dtype=np.float32))  
    new_input = tf.placeholder(tf.float32, shape=( 30 , 16000), name='new_input_placeholder')
    audio1 = tf.add(modifier , new_input)
    print(audio1)
    audio12 = tf.clip_by_value(audio1,-1,1,name="adv_name")
    
    
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    #save_path_name = "ckpts/new1.ckpt"
    #save_path = saver.save(sess, save_path_name)
    
    new_saver = tf.train.import_meta_graph('ckpts/model3.meta',input_map={'Placeholder:0':  audio12})
    new_saver.restore(sess, 'ckpts/model3')
    
    #saver.restore(sess, save_path_name)
    
    save_path_name = "ckpts/new.ckpt"
    save_path = saver.save(sess, save_path_name)
    #init_op = sess.graph.get_operation_by_name('init')
    #sess.run(init_op)
    #print(init_op)
    
    #tf.import_graph_def(graph_def, name='', input_map={"Reshape_1": mfcc2})
    ## LOAD predict tensor
    output = tf.get_default_graph().get_tensor_by_name("dense/BiasAdd:0")
    print(output)
    ## compute the probability of the label class versus the maximum other
    #tlab = tf.get_default_graph().get_tensor_by_name("Placeholder_1_1:0")
    tlab = tf.placeholder(tf.float32, shape=( 30 , 10) )
    real = tf.reduce_sum( tlab * output, 1)
    other = tf.reduce_max( (1 - tlab) * output - tlab * 10000, 1)
    print(tlab)
    loss1 = tf.maximum(ZERO(), other - real )
    
    ## loss 2
    loss2 = tf.reduce_sum(tf.square(audio1 - new_input),  1 )
    #loss2 = l2dist * 5000
    
    tot_loss = loss1 + loss2 * 5000;
    
    optimizer = tf.train.AdamOptimizer(0.01)
    trains = optimizer.minimize( tot_loss, var_list=[modifier])
    
    predict = tf.nn.softmax(output)
    
    init = tf.variables_initializer(var_list=[modifier]+tf.global_variables() )
    #init = tf.variables_initializer(var_list=[modifier] )
    
    sess.run(tf.global_variables_initializer())
    init_m = tf.variables_initializer(var_list=[modifier] )
    sess.run(init_m)
    """
    saver = tf.train.Saver()
    save_path_name = "ckpts/" + str(99) + "_batch30_model.ckpt"
    saver.restore(sess, save_path_name)
    """
    saver.restore(sess, save_path_name)
    #init_op2 = tf.global_variables_initializer()
    #sess.run(init_op2)
    #sess.run(init_op)
    new_saver.restore(sess, 'ckpts/model3')
    
    """
    print("********************************")
    print("*   train original model acc:   *")
    print("********************************")
    print("")
    print("")
    anss = 0
    all_data = 0
    for dst in labels_dict:
        
        #sess.run(init)
        print("destination file: ",dst)
        labels_t = []
        l1 = labels_dict.index(dst)
        ll1 = [0,0,0,0,0,0,0,0,0,0]
        ll1[l1] = 1
            
        for i in range(int(len(datas[l1])/30)):
            #sess.run(init)
            if i > 3:
                break
            sess.run(init_m)
            result = sess.run(predict, feed_dict={new_input: datas[l1][i*30:i*30+30]})
            all_data = all_data + 30
            for k in result:
                if np.argmax(k) == l1:
                    anss = anss + 1
    print(anss/all_data)
    """
    
    ## some measure-ment matric
    count_metric = np.zeros((10, 10))
    
    total_data = 0
    total_ans = 0
    
    print(labels_dict[0])
    for src in labels_dict:
        src1 = labels_dict.index(src)
        print("source file: ",src)
        for dst in labels_dict:
            if src != dst:
                sess.run(init_m)
                print("destination file: ",dst)
                labels_t = []
                l1 = labels_dict.index(dst)
                ll1 = [0,0,0,0,0,0,0,0,0,0]
                ll1[l1] = 1
                for kk in range(200):
                    labels_t.append(ll1)
                
                tmp_ans = 0
                
                for i in range(int(len(datas[src1])/30)):
                    #sess.run(init)
                    if i > 3:
                        break
                    ## count total adv
                    total_data = total_data + 30
                    
                    _, adv = sess.run( [trains, audio12 ] , feed_dict={new_input : datas[src1][i*30:i*30+30], tlab : labels_t[i*30:i*30+30] }  )
                    adv = np.array(adv)
                    #print(adv.shape)
                    result = sess.run(predict, feed_dict={new_input: adv})
                    result = result
                    anss = 0
                    for k in result:
                        if np.argmax(k) == l1:
                            anss = anss + 1
                    check_fail = 0
                    
                    
                    
                    check_np1 = np.zeros(30)
                    while anss<25:
                        
                        # check whether write to file
                        check_np = np.zeros(30)
                        
                        check_fail = check_fail + 1
                        if check_fail % 200==0:
                            print("check fail: ",check_fail)
                        if check_fail == 200:
                            break
                        _, adv = sess.run( [trains, audio12 ] , feed_dict={new_input : datas[src1][i*30:i*30+30], tlab : labels_t[i*30:i*30+30] }  )
                        adv = np.array(adv)
                        #print(adv.shape)
                        result = sess.run(predict, feed_dict={new_input: adv})
                        result = result
                        anss = 0
                        l_np = 0
                        for k in result:
                            if np.argmax(k) == l1:
                                check_np[l_np]=1
                                anss = anss + 1
                            l_np = l_np + 1
                        check_np1 = check_np
                    #print("ans: ",anss)
                    if check_fail == 200:
                        print("Fail！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
                        break;
                    ## count success adv
                    total_ans = total_ans + anss
                    tmp_ans = tmp_ans + anss
                    
                    adv = adv * 32767
                    adv = np.array(adv, dtype=np.int16)
                    for j in range( 30 ):
                        if check_np1[j]==1:
                            out_name = "adv_datas/" + src + "/" + dst + "/" +str(i*30 + j)+".wav"
                            wav.write(out_name ,16000 , adv[j])
                    """
                    for j in range( 30 ):
                        out_name = "adv_data_C_and_W/orig_yes/" +str(i*30 + j)+".wav"
                        adv2 = datas[src1][j] * 32767
                        adv2 = np.array(adv2, dtype=np.int16)
                        wav.write(out_name ,16000 , adv2)
                    """
                print("     tmp acc: ",tmp_ans/120)
                count_metric[src1][l1] = tmp_ans
    print("acc: ",total_ans/total_data)
    np.save("C_and_W.npy", count_metric)
                        
    
    
