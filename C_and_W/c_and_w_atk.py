import os
import sys
import math

import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from random import shuffle
from scipy.signal import butter, lfilter

tf.reset_default_graph()


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config2 = tf.ConfigProto() 
config2.gpu_options.per_process_gpu_memory_fraction = 0.3


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
datas = np.array(datas)
print("data shape: ",datas.shape)
datas = datas / 32767


#min_arr = np.zeros( (len(datas),len(datas[0]) , 1) , dtype=np.float32 )
#max_arr = np.zeros( (len(datas),len(datas[0]) , 1) , dtype=np.float32 )
#const_arr = np.zeros( (len(datas),len(datas[0]) , 1) , dtype=np.float32 )
#print("const_arr size: ", const_arr.shape)

#for jjj in range(len(datas)):
#    for iii in range( len(datas[0]) ):
#        max_arr[ jjj ][iii][0] = 100000
#        const_arr[ jjj ][iii][0] = 5000

filename = "ckpts/train2.pb"
## GPU Environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3


modifier_tmp = np.zeros((30, 16000))
modifier_tmp2 = np.zeros((30, 16000))

with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    
    Const = tf.placeholder(tf.float32, shape=( 30 , 1), name='constant')
    ## global varieble for tensorflow to optimaize    
    modifier = tf.Variable(np.zeros( (30, 16000) ,  dtype=np.float32))
    modifier_place_holder = tf.placeholder(tf.float32, shape=( 30 , 16000), name='modify')
    assign_op = modifier.assign( modifier_place_holder )
    #assign_op2 = modifier.assign(modifier_tmp2)
    new_input = tf.placeholder(tf.float32, shape=( 30 , 16000), name='new_input_placeholder')
    audio1 = tf.add(modifier , new_input)
    print(audio1)
    audio12 = tf.clip_by_value(audio1,-1,1,name="adv_name")
    
    
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    new_saver = tf.train.import_meta_graph('ckpts/model3.meta',input_map={'Placeholder:0':  audio12})
    new_saver.restore(sess, 'ckpts/model3')
    
    
    
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
    
    ## make the target confidence score more higher
    zeross = np.zeros(30,dtype=np.float32)
    for k1 in range(30):
        zeross[k1]=0
    
    loss1 = tf.maximum( zeross , other - real )
    
    ## loss 2
    loss2 = tf.reduce_sum(tf.square(audio1 - new_input),  1 )
    #loss2 = l2dist * 5000
    
    tot_loss = loss1 * Const + loss2;
    
    optimizer = tf.train.AdamOptimizer(0.01)
    trains = optimizer.minimize( tot_loss, var_list=[modifier])
    predict = tf.nn.softmax(output)
    predict2 = tf.get_default_graph().get_tensor_by_name("Softmax:0")
    
    init = tf.variables_initializer(var_list=[modifier]+tf.global_variables() )
    
    sess.run(tf.global_variables_initializer())
    init_m = tf.variables_initializer(var_list=[modifier] )
    sess.run(init_m)
    
    
    #saver.restore(sess, save_path_name)
    new_saver.restore(sess, 'ckpts/model3')
    
    
    
    ### New Graph
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config2 = tf.ConfigProto() 
    config2.gpu_options.per_process_gpu_memory_fraction = 0.3

    sess2 = tf.Session(config=config2)
    #sess2.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('ckpts/model4.meta')
    saver2.restore(sess2, 'ckpts/model4')
    
    placeholder_2 = tf.get_default_graph().get_tensor_by_name("Placeholder_3:0")
    predict2 = tf.get_default_graph().get_tensor_by_name("Softmax_2:0")
    """
    
    
    
    ## some measure-ment matric
    count_metric = np.zeros((10, 10))
    tot_norm = 0
    
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
                
                
                
                tot_err_arr = np.zeros( (120) , dtype=np.float32 )
                """
                anss = 0
                tot = 0
                for i in range(int(len(datas[src1])/30)):
                    sess.run(assign_op, feed_dict={ modifier_place_holder : modifier_tmp2 })
                    result = sess.run(predict2, feed_dict={new_input: datas[src1][i*30:i*30+30] })
                    for k in result:
                        tot = tot + 1
                        if np.argmax(k) == src1:
                            anss = anss + 1
                print(src," ",anss," ",tot)
                """
                
                for i in range(int(len(datas[src1])/30)):
                    #sess.run(init)
                    sess.run(assign_op, feed_dict={ modifier_place_holder : modifier_tmp2 })
                    if i > 3:
                        break
                    ## count total adv
                    total_data = total_data + 30
                    
                    
                    ## add constant
                    min_arr = np.zeros( (30 , 1) , dtype=np.float32 )
                    max_arr = np.zeros( (30 , 1) , dtype=np.float32 )
                    const_arr = np.zeros( (30 , 1) , dtype=np.float32 )
                    for iji in range(30):
                        max_arr[iji][0] = 1000000
                        const_arr[iji][0] = 5000
                    
                    # norm2 perturbation
                    norm2_arr = np.zeros(30,dtype=np.float32)
                    
                    check_np1 = np.zeros(30)
                    for binary in range(5):
                        
                        
                        
                        #modifier
                        _, adv = sess.run( [trains, audio12 ] , feed_dict={new_input : datas[src1][i*30:i*30+30], tlab : labels_t[i*30:i*30+30], Const : const_arr }  )
                        modifier_tmp = sess.run(modifier)
                        adv = np.array(adv)
                        #print(adv.shape)
                        #assign_op = modifier.assign(modifier_tmp)
                        sess.run(assign_op, feed_dict={ modifier_place_holder : modifier_tmp2 })
                        #result = sess2.run(predict2, feed_dict={placeholder_2: adv})
                        result = sess.run(predict2, feed_dict={new_input: adv})
                        sess.run(assign_op, feed_dict={ modifier_place_holder : modifier_tmp})
                        
                        anss = 0
                        for k in result:
                            if np.argmax(k) == l1:
                                anss = anss + 1
                        check_fail = 0
                        
                        check_np1 = np.zeros(30)
                        #while anss<20:
                        for iter in range(800):
                            # check whether write to file
                            check_np = np.zeros(30)
                            
                            check_fail = check_fail + 1
                            if check_fail % 200==0:
                                print("check fail: ",check_fail,"  binary: ",binary ,"  modifier: ", np.sum(modifier_tmp) )
                            if check_fail == 800:
                                break
                            _, adv = sess.run( [trains, audio12 ] , feed_dict={new_input : datas[src1][i*30:i*30+30], tlab : labels_t[i*30:i*30+30], Const : const_arr }  )
                            modifier_tmp = sess.run( modifier )
                            adv = np.array(adv)
                            #print(adv.shape)
                            #assign_op = modifier.assign(modifier_tmp)
                            sess.run(assign_op, feed_dict={ modifier_place_holder : modifier_tmp2 })
                            #result = sess2.run(predict2, feed_dict={placeholder_2: adv})
                            result = sess.run(predict2, feed_dict={new_input: adv})
                            sess.run(assign_op, feed_dict={ modifier_place_holder : modifier_tmp })
                            
                            #result = result
                            anss = 0
                            #l_np = 0
                            adv = adv * 32767
                            adv = np.array(adv, dtype=np.int16)
                            
                            real_data = datas[src1][i*30:i*30+30] * 32767
                            real_data = np.array(real_data, dtype=np.int16)
                            
                            for k in range(len(result)):
                                if np.argmax(result[k]) == l1:
                                    #if check_np[]
                                    check_np[ k ]=1
                                    anss = anss + 1
                                    
                                    out_name = "adv_datas/" + src + "/" + dst + "/" +str(i*30 + k)+".wav"
                                    wav.write(out_name ,16000 , adv[k])
                                    
                                    tmp_norm = 0
                                    #for kk in range(16000):
                                    #    tmp_norm = tmp_norm + ( adv[k][kk] - real_data[k][kk] )*( adv[k][kk] - real_data[k][kk] )
                                    #norm2_arr[k] = np.linalg.norm( adv[k] - real_data[k], ord=2)
                                    adv_max = np.max(np.abs(adv[k]))
                                    real_max = np.max(np.abs(real_data[k]))
                                    
                                    norm2_arr[k] = abs(adv_max - real_max)
                                    
                                    
                                    
                            if anss>20:
                                break
                                    
                                    
                                    
                            check_np1 = check_np
                        
                        ## count success adv
                        total_ans = total_ans + anss
                        #tmp_ans = tmp_ans + anss
                        
                        #adv = adv * 32767
                        #adv = np.array(adv, dtype=np.int16)
                        for j in range( 30 ):
                            if check_np1[j]==1:
                                #out_name = "adv_datas/" + src + "/" + dst + "/" +str(i*30 + j)+".wav"
                                #wav.write(out_name ,16000 , adv[j])
                                
                                const_arr[ j ][0] = (min_arr[ j ][0] + const_arr[ j ][0])/2
                                max_arr[ j ][0] = const_arr[ j ][0]
                                
                                tot_err_arr[ i * 30 + j ] = 1
                            else:
                                const_arr[ j ][0] = (max_arr[ j ][0] + const_arr[ j ][0])/2
                                min_arr[ j ][0] = const_arr[ j ][0]
                    
                    
                    tot_norm = tot_norm + np.sum(norm2_arr)
                    
                    
                    
                for q1q in range(120):
                    if tot_err_arr[q1q] == 1 :
                        tmp_ans = tmp_ans + 1
                print("     tmp acc: ",tmp_ans/120)
                count_metric[src1][l1] = tmp_ans
                
                
                
                
    print("acc: ",total_ans/total_data)
    np.save("C_and_W.npy", count_metric)
    print("norm: ", tot_norm/total_data)
    print("tot_norm: ",tot_norm)
    print("tot_data: ",total_data)
