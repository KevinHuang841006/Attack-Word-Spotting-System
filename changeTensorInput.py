"""
python3 question.py --output_dir=output --labels_file=ckpts/conv_actions_labels.txt --graph_file=ckpts/DS_CNN_S.pb

重點在
tf.import_graph_def 
in load_graph()

example:
    更換每個以Reshape當input的tensor 
    tf.import_graph_def(graph_def, name='', input_map={"Reshape_1:0": new_input})
"""

import numpy as np
import tensorflow as tf
from speech_commands import label_wav
import os, sys
import csv
flags = tf.flags
flags.DEFINE_string('output_dir', '', 'output data directory')
flags.DEFINE_string('labels_file', '', 'Labels file.')
flags.DEFINE_string('graph_file', '', '')
flags.DEFINE_string('output_file', 'eval_output.csv', 'CSV file of evaluation results')
FLAGS = flags.FLAGS

def load_graph(filename):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # tf.import_graph_def(graph_def, name='')
        new_input = tf.placeholder(tf.float32, shape=(None, 49, 10, 1), name='new_input_placeholder')
        tf.import_graph_def(graph_def, name='', input_map={"Reshape_1:0": new_input})
    return new_input

def load_labels(filename):
    return [line.rstrip() for line in tf.gfile.FastGFile(filename)]


def load_audiofile(filename):
    with open(filename, 'rb') as fh:
        return fh.read()

if __name__ == '__main__':
    output_dir = FLAGS.output_dir
    labels_file = FLAGS.labels_file
    graph_file = FLAGS.graph_file
    output_file = FLAGS.output_file
    labels = load_labels(labels_file)
    n_labels = len(labels)
    result_mat = np.zeros((n_labels, n_labels))

    new_input_tensor = load_graph(graph_file)

    ## Header of output file
    output_fh = open(output_file, 'w')
    fieldnames = ['filename', 'original', 'target', 'predicted']
    for label in labels:
        fieldnames.append(label)
    csv_writer = csv.DictWriter(output_fh, fieldnames=fieldnames)
    print(fieldnames)
    csv_writer.writeheader()

    ## GPU Environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.03
    sess = tf.Session(config = config)

    print('\nPrint all variables from graph')
    for n in tf.get_default_graph().as_graph_def().node:
        if 'conv_1/Conv2D' in n.name:
            print(n.name)
            print("input: ", n.input)
            print()

    print('This is the original input to conv, "Reshape_1"')
    tensor = (tf.get_default_graph().get_tensor_by_name('Reshape_1:0'))
    print(tensor)
    print('This is the new input placeholder (with the same shape as "Reshape_1"')
    print(new_input_tensor)
