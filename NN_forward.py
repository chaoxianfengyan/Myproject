import tensorflow as tf
import numpy as np

def layer_weight_init(input_quantity, output_quantity):
    layer_weight = tf.Variable(tf.random_normal([input_quantity, output_quantity],stddev = 0.1))
    return layer_weight
    
def bias_init(output_quantity):
    bias = tf.constant(0.1,shape = [output_quantity])
    return bias
    
def forward(input, layer_weight, active, bias):
    
    if active == 'relu':
        result = tf.nn.relu(tf.matmul(input,layer_weight)+bias)
    elif active == 'sigmoid':
        result = tf.nn.sigmoid(tf.matmul(input,layer_weight)+bias)
    else:
        result = tf.nn.softmax(tf.matmul(input,layer_weight)+bias)
    return result
    
def NN_forward(x, layer, layer_size, active):
    '''这一函数建立多层全连接神经网络，最后一层没有进行激活操作
       x:网络的输入数据
       layer:网络的层数
       layer_size:网络每层神经元的个数
       active:网络的激活函数，有'relu','sigmoid','softmax'三个选项'''
     input = x
     layers = layer_size
     for i in range(layer-1):
         #print(i)
         weights = layer_weight_init(layers[i], layers[i+1])
         biases = bias_init(layers[i+1])
         if i+1==layer:
             NN_forward = tf.matmul(input,layer_weight)+bias
         else:
             NN_forward = forward(input,weights,active, biases)
         input = NN_forward
     return input