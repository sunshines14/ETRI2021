import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from keras.models import Model


def conv2d_layer(inputs, filter_num, kernel_size):
    x = inputs
    x = BatchNormalization(center=True, scale=True)(x)
    x = Conv2D(filter_num, kernel_size=kernel_size, padding='same', activation=tf.nn.relu)(x)
    return x

def My_freq_split1(x):
    from keras import backend as K
    return x[:,:,0:20,:]

def My_freq_split2(x):
    from keras import backend as K
    return x[:,:,20:40,:]

def model_etri(NUM_CLASSES, input_shape=(40,None,1)):   
    
    inputs = Input(shape=input_shape)
    
    Splits1 = Lambda(My_freq_split1)(inputs)
    Splits2 = Lambda(My_freq_split2)(inputs)

    #split1
    Splits1 = conv2d_layer(inputs=Splits1,filter_num=18,kernel_size=(3,3))
    Splits1 = conv2d_layer(inputs=Splits1,filter_num=36,kernel_size=(3,3))
    Splits1 = AveragePooling2D(pool_size=(3,3),strides=[2,1],padding='same')(Splits1)
    
    Splits1 = conv2d_layer(inputs=Splits1,filter_num=72,kernel_size=(3,3))
    Splits1 = conv2d_layer(inputs=Splits1,filter_num=144,kernel_size=(3,3))
    Splits1 = AveragePooling2D(pool_size=(3,3),strides=[2,1],padding='same')(Splits1)
    
    Splits1 = conv2d_layer(inputs=Splits1,filter_num=288,kernel_size=(3,3))
    Splits1 = AveragePooling2D(pool_size=(3,3),strides=[1,1],padding='same')(Splits1)
    
    
    #split2
    Splits2 = conv2d_layer(inputs=Splits2,filter_num=18,kernel_size=(3,3))
    Splits2 = conv2d_layer(inputs=Splits2,filter_num=36,kernel_size=(3,3))
    Splits2 = AveragePooling2D(pool_size=(3,3),strides=[2,1],padding='same')(Splits2)
    
    Splits2 = conv2d_layer(inputs=Splits2,filter_num=72,kernel_size=(3,3))
    Splits2 = conv2d_layer(inputs=Splits2,filter_num=144,kernel_size=(3,3))
    Splits2 = AveragePooling2D(pool_size=(3,3),strides=[2,1],padding='same')(Splits2)
    
    Splits2 = conv2d_layer(inputs=Splits2,filter_num=288,kernel_size=(3,3))
    Splits2 = AveragePooling2D(pool_size=(3,3),strides=[1,1],padding='same')(Splits2)

    
    OutPath = concatenate([Splits1,Splits2],axis=2)
    
    OutPath = conv2d_layer(inputs=OutPath,filter_num=NUM_CLASSES,kernel_size=(3,3))
    OutPath = BatchNormalization(center=False, scale=False)(OutPath)
    OutPath = GlobalAveragePooling2D()(OutPath)
    OutPath = Activation('softmax')(OutPath)
    
    model = Model(inputs=inputs, outputs=OutPath)
    
    return model