import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers.convolutional import Convolution1D
import numpy as np
import sys

class Neural_Tensor_layer(Layer):
    def __init__(self,output_dim,input_dim=None, **kwargs):
        self.output_dim=output_dim
        self.input_dim=input_dim
        if self.input_dim:
            kwargs['input_shape']=(self.input_dim,)
        super(Neural_Tensor_layer,self).__init__(**kwargs)

    def call(self,inputs,mask=None):
        e1=inputs[0]
        e2=inputs[1]
        batch_size=K.shape(e1)[0]
        k=self.output_dim

        feed_forward=K.dot(K.concatenate([e1,e2]),self.V)

        bilinear_tensor_products = [ K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1) ]

        for i in range(k)[1:]:	
            btp=K.sum((e2*K.dot(e1,self.W[i]))+self.b,axis=1)
            bilinear_tensor_products.append(btp)

        result=K.tanh(K.reshape(K.concatenate(bilinear_tensor_products,axis=0),(batch_size,k))+feed_forward)

        return result
    
    def build(self,input_shape):
        mean=0.0
        std=1.0
        k=self.output_dim
        d=self.input_dim
        ##truncnorm generate continuous random numbers in given range
        W_val=stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k,d,d))
        V_val=stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(2*d,k))
        self.W=K.variable(W_val)
        self.V=K.variable(V_val)
        self.b=K.zeros((self.input_dim,))
        self.trainable_weights=[self.W,self.V,self.b]    

    def compute_output_shape(self, input_shape):
        batch_size=input_shape[0][0]
        return(batch_size,self.output_dim)
    
class Temporal_Mean_Pooling(Layer): # conversion from (samples,timesteps,features) to (samples,features)
    def __init__(self, **kwargs):
        super(Temporal_Mean_Pooling,self).__init__(**kwargs)
        # masked values in x (number_of_samples,time)
        self.supports_masking=True
        # Specifies number of dimensions to each layer
        self.input_spec=InputSpec(ndim=3)
        
    def call(self,x,mask=None):
        if mask is None:
            mask=K.mean(K.ones_like(x),axis=-1)

        mask=K.cast(mask,K.floatx())
        #dimension size single vec/number of samples
        return K.sum(x,axis=-2)/K.sum(mask,axis=-1,keepdims=True)        

    def compute_mask(self,input,mask):
        return None
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[2])
    
class Attention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights = [self.att_v, self.att_W]

    def call(self, x, mask=None):
        y = K.dot(x, self.att_W)
        if not self.activation:
            weights = K.theano.tensor.tensordot(self.att_v, y, axes=[0, 2])
        elif self.activation == 'tanh':
            weights = K.theano.tensor.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
        weights = K.softmax(weights)
        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            out = out.sum(axis=1)
        elif self.op == 'attmean':
            out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MeanOverTime(Layer):
    def __init__(self, mask_zero=True, **kwargs):
        self.mask_zero = mask_zero
        self.supports_masking = True
        super(MeanOverTime, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.mask_zero:
            mask = K.cast(mask, K.floatx())
            return K.cast(K.sum(x, axis=1) / K.sum(mask, axis=1, keepdims= True), K.floatx())
        else:
            return K.mean(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'mask_zero': self.mask_zero}
        base_config = super(MeanOverTime, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Conv1DWithMasking(Convolution1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Conv1DWithMasking, self).__init__(**kwargs)

    def compute_mask(self, x, mask):
        return mask