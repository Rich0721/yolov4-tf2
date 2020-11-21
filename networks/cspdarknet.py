import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, LeakyReLU, Add, Concatenate, MaxPooling2D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model


def conv(x, filters, kernel_size, downsampling=False, activation='leaky', batch_norm=True, use_bias=False):

    def mish(x):
        return x * tf.tanh(tf.math.softplus(x))
    
    if downsampling:
        #x = ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
        padding = 'same'
        strides = 2
    else:
        padding = 'same'
        strides = 1
    
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, kernel_initializer=RandomNormal(mean=0, stddev=0.01))(x)
    
    if batch_norm:
        x = BatchNormalization()(x)
    
    if activation == 'mish':
        x = mish(x)
    elif activation == 'leaky':
        x = LeakyReLU(alpha=0.1)(x)
    

    return x


def residual_block(x, filter1, filter2, activation='leaky'):

    y = conv(x, filter1, kernel_size=1, activation=activation)
    y = conv(y, filter2, kernel_size=3, activation=activation)
    
    return Add()([x, y])


def csp_block(x, residual_out, repeat, residual_bottleneck=False):
    """Cross Stage Partial Network(CSPNet)"""

    filter1 = residual_out // 2 if residual_bottleneck else residual_out
    filter2 = residual_out
    
    route = x
    route = conv(route, residual_out, 1, activation='mish')
    x = conv(x, residual_out, 1, activation="mish")
    for i in range(repeat):
        
        x = residual_block(x, filter1, filter2, activation='mish')
        
    x = conv(x, residual_out, 1, activation='mish')

    x = Concatenate()([x, route])
    return x


def cspdarknet53(inputs):
    x = conv(inputs, 32, 3)
    x = conv(x, 64, 3, downsampling=True)
    
    x = csp_block(x, residual_out=64, repeat=1, residual_bottleneck=True)
    x = conv(x, 64, 1, activation='mish')
    x = conv(x, 128, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=64, repeat=2)
    x = conv(x, 128, 1, activation='mish')
    x = conv(x, 256, 1, activation='mish', downsampling=True)
    
    x = csp_block(x, residual_out=128, repeat=8)
    x = conv(x, 256, 1, activation='mish')
    route0 = x
    x = conv(x, 512, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=256, repeat=8)
    x = conv(x, 512, 1, activation='mish')
    route1 = x
    x = conv(x, 1024, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=512, repeat=4)

    x = conv(x, 1024, 1, activation='mish')
    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = conv(x, 512, 1)

    maxpool_13 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    maxpool_9 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool_5 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)

    x = Concatenate()([maxpool_13, maxpool_9, maxpool_5, x])
    
    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    route2 = conv(x, 512, 1)
    return Model(inputs, [route0, route1, route2])

