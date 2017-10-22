import tensorflow as tf
import numpy as np

def filter_init(conv_size, original_deep, conv_deep):
    filter = tf.Variable(tf.random_normal([conv_size, conv_size, original_deep, conv_deep],stddev = 0.1))
    return filter
    
def bias_init(conv_deep):
    bias = tf.constant(0.0,shape = [conv_deep])
    return bias
    
def conv(input, weights, strides, padding, active, bias):
    if padding == "SAME":
        conv = tf.nn.conv2d(input, weights,strides = [1, strides, strides, 1], padding = 'SAME')
    else:
        conv = tf.nn.conv2d(input, weights,strides = [1, strides, strides, 1], padding = 'VALID')   
    if active == 'relu':
        result = tf.nn.relu(tf.nn.bias_add(conv, bias))
    elif active == 'sigmoid':
        result = tf.nn.sigmoid(tf.nn.bias_add(conv, bias))
    else:
        result = tf.nn.softmax(tf.nn.bias_add(conv, bias))
    return result
        
def pooling(result, ksize, strides, padding, pool):
    if padding == "SAME" and pool == 'MAX':
        pool = tf.nn.max_pool(result, ksize = [1,ksize,ksize,1], strides = [1, strides, strides,1],padding = 'SAME')
    elif padding == "SAME" and pool == 'AVG':
        pool = tf.nn.avg_pool(result, ksize = [1,ksize,ksize,1], strides = [1, strides, strides,1],padding = 'SAME')
    elif padding == "None" and pool == 'MAX':
        pool = tf.nn.max_pool(result, ksize = [1,ksize,ksize,1], strides = [1, strides, strides,1],padding = 'VALID')
    else:
        pool = tf.nn.avg_pool(result, ksize = [1,ksize,ksize,1], strides = [1, strides, strides,1],padding = 'VALID')
    return pool
    
def CNN_conv_pool(x, layer, filter_size, channels, filter_deep, filter_strides, filter_padding, filter_active, pool_size, pool_strides, pool_padding, pool_type):
    '''这一函数建立一个多层卷积神经网络，卷积层与池化层一一对应，一个卷积层必有一个池化层与之匹配
    各个输入参数如下所示：
    x表示网络第一层的输入数据；
    layer表示网络的层数，表示有几个卷积层，池化层对
    filter_size表示卷积层中过滤器（卷积核）的尺寸，用列表表示每个卷积层的过滤器尺寸，如[5，5]；
    channels表示输入数据的通道数，对于RGB三色图像来说，该数值为3，输入必须为列表形式，如[3]
    filter_deep表示过滤器的深度，用列表表示每个卷积层的过滤器深度，如[32，64]
    filter_strides表示过滤器移动的步长，如1,2
    filter_padding表示输入层是否需要全0填充，需要输入为'SAME',不需要输入为'VALID'
    filter_active表示激活函数，有三种输入，分别为'sigmoid','relu','softmax'
    pool_size表示池化层的尺寸，用列表表示每个池化层的尺寸，如[2，2]
    pool_strides表示池化层移动的步数，用列表表示每个池化层移动的步数，如[1，1]
    pool_padding 表示池化层的输入是否需要全0填充，需要输入为'SAME',不需要输入为'VALID'
    pool_tyoe表示进行池化操作的类型，使用最大池化层填入'MAX'，使用平均池化层填入AVG'''
    input = x
    filter = filter_size
    deep = channels + filter_deep
    filter_stride = filter_strides
    pool_shape = pool_size
    print(pool_shape)
    pool_stride = pool_strides
    print(pool_stride)
    for i in range(layer):
        con_weight = filter_init(filter[i], deep[i],deep[i+1])
        #print(i)
        con_biases = bias_init(deep[i+1])
        #print(i)
        con_forward = conv(input, con_weight, filter_stride[i], filter_padding, filter_active, con_biases)
       # print(i)
        #print(pool_shape[i])
        #print(type(pool_shape[i]))
        #print(pool_stride[i])
        #print(type(pool_stride[i]))
        pool_result = pooling(con_forward, pool_shape[i], pool_stride[i], 'SAME', pool_type)
        input = pool_result
    return input