from functools import wraps, reduce
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, Layer, Concatenate
from tensorflow.keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from tensorflow.keras import backend as K

class Mish(Layer):

    '''
    Mish = x * tanh(ln(1 + e^x))
    Softplus = ln(1 + e^x)
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True
    
    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))
    
    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

def compose(*funcs):

    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_kwargs = {'kernel_regularizer':l2(5e-4)}
    darknet_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_kwargs)


def DarknetConv2D_BN_Mish(*args, **kwargs):

    no_bias_kwargs = {'use_bias':False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


def DarknetConv2D_BN_Leaky(*args, **kwargs):

    no_bias_kwargs = {'use_bias':False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))



def resdual_block(inputs, filters, num_blocks=1, all_narrow=True):

    preconv1 = ZeroPadding2D(((1,0),(0,1)))(inputs)
    preconv1 = DarknetConv2D_BN_Mish(filters, (3, 3), strides=(2, 2))(preconv1)
    shortconv = DarknetConv2D_BN_Mish(filters//2 if all_narrow else filters, (1, 1))(preconv1)
    mainconv = DarknetConv2D_BN_Mish(filters//2 if all_narrow else filters, (1, 1))(preconv1)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Mish(filters//2, (1, 1)),
            DarknetConv2D_BN_Mish(filters//2 if all_narrow else filters, (3, 3)))(mainconv)
        mainconv = Add()([mainconv, y])
    
    postconv = DarknetConv2D_BN_Mish(filters//2 if all_narrow else filters, (1, 1))(mainconv)
    x = Concatenate()([postconv, shortconv])
    x = DarknetConv2D_BN_Mish(filters, (1, 1))(x)
    return x


def CSPdarknet53(inputs):

    x = DarknetConv2D_BN_Mish(32, (3, 3))(inputs)
    x = resdual_block(x, 64,  num_blocks=1, all_narrow=False)
    x = resdual_block(x, 128, num_blocks=2)
    x = resdual_block(x, 256, num_blocks=8)
    feat1 = x
    x = resdual_block(x, 512, num_blocks=8)
    feat2 = x
    x = resdual_block(x, 1024, num_blocks=4)
    feat3 = x
    return feat1, feat2, feat3
