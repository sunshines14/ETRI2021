import keras
from keras import backend as K
from tensorflow.keras.layers import *
from keras.regularizers import l2
from keras.models import Model
from networks.attention import se_channel_attention, coordinate_attention


def res(inputs, num_filters, kernel_size, strides, learn_bn, wd, use_relu):
    x = inputs
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)             
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=True)(x)
    return x

def pad_depth(inputs, desired_channels):
    from keras import backend as K
    y = K.zeros_like(inputs)
    return y

def split1(x):
    from keras import backend as K
    return x[:,0:20,:,:]

def split2(x):
    from keras import backend as K
    return x[:,20:40,:,:]

def model_resnet(num_classes, input_shape, num_filters, kernel_size, wd, num_stacks, use_split):
    num_res_blocks = 2
    inputs = Input(shape = input_shape)
    
    if use_split:
        Split1 = Lambda(split1)(inputs)
        Split2 = Lambda(split2)(inputs)
        ResidualPath1 = res(inputs = Split1,
                            num_filters=num_filters,
                            kernel_size=kernel_size,
                            strides=(1,1),
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
        ResidualPath2 = res(inputs = Split2,
                            num_filters=num_filters,
                            kernel_size=kernel_size,
                            strides=(1,1),
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
        for stack in range(num_stacks):
            for res_block in range(num_res_blocks):
                if stack > 0 and res_block == 0:
                    strides = (1,1)
                ConvPath1 = res(inputs=ResidualPath1,
                                num_filters=num_filters,
                                kernel_size=kernel_size,
                                strides=(1,1),
                                learn_bn=True,
                                wd=wd,
                                use_relu=True)
                ConvPath2 = res(inputs=ResidualPath2,
                                num_filters=num_filters,
                                kernel_size=kernel_size,
                                strides=(1,1),
                                learn_bn=True,
                                wd=wd,
                                use_relu=True)
                ConvPath1 = res(inputs=ConvPath1,
                                num_filters=num_filters,
                                kernel_size=kernel_size,
                                strides=(1,1),
                                learn_bn=True,
                                wd=wd,
                                use_relu=True)
                ConvPath1 = res(inputs=ConvPath2,
                                num_filters=num_filters,
                                kernel_size=kernel_size,
                                strides=(1,1),
                                learn_bn=True,
                                wd=wd,
                                use_relu=True)
                if stack > 0 and res_block == 0:  
                    ResidualPath1 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(ResidualPath1)
                    ResidualPath2 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(ResidualPath2)
                    desired_channels = ConvPath1.shape.as_list()[-1]
                    Padding1=Lambda(pad_depth,arguments={'desired_channels':desired_channels})(ResidualPath1)
                    ResidualPath1 = Concatenate(axis=-1)([ResidualPath1,Padding1])
                    Padding2=Lambda(pad_depth,arguments={'desired_channels':desired_channels})(ResidualPath2)
                    ResidualPath2 = Concatenate(axis=-1)([ResidualPath2,Padding2])
                ResidualPath1 = add([ConvPath1,ResidualPath1])
                ResidualPath2 = add([ConvPath2,ResidualPath2])
            num_filters *= 2
        ResidualPath = Concatenate(axis=1)([ResidualPath1,ResidualPath2])
        
    else:
        ResidualPath = res(inputs = inputs,
                           num_filters=num_filters,
                           kernel_size=kernel_size,
                           strides=(1,1),
                           learn_bn=True,
                           wd=wd,
                           use_relu=True)
        for stack in range(num_stacks):
            for res_block in range(num_res_blocks):
                if stack > 0 and res_block == 0:
                    strides = (1,1)
                ConvPath = res(inputs=ResidualPath,
                               num_filters=num_filters,
                               kernel_size=kernel_size,
                               strides=(1,1),
                               learn_bn=True,
                               wd=wd,
                               use_relu=True)
                ConvPath = res(inputs=ConvPath,
                               num_filters=num_filters,
                               kernel_size=kernel_size,
                               strides=(1,1),
                               learn_bn=True,
                               wd=wd,
                               use_relu=True)
                if stack > 0 and res_block == 0:  
                    ResidualPath = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(ResidualPath)
                    desired_channels = ConvPath.shape.as_list()[-1]
                    Padding = Lambda(pad_depth,arguments={'desired_channels':desired_channels})(ResidualPath)
                    ResidualPath = Concatenate(axis=-1)([ResidualPath,Padding])
                ResidualPath = add([ConvPath,ResidualPath])  
            num_filters *= 2
            
    OutputPath = res(inputs=ResidualPath,
                     num_filters=num_classes,
                     kernel_size=(1,1),
                     strides=(1,1),
                     learn_bn=False,
                     wd=wd,
                     use_relu=False)
    OutputPath = BatchNormalization(center=False, scale=False)(OutputPath)
    #OutputPath = coordinate_attention(OutputPath, reduction_ratio=8)
    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Activation('softmax')(OutputPath)
    
    model = Model(inputs=inputs, outputs=OutputPath)
    return model