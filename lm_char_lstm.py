'''
Build a simple neural language model using GRU units
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

#from char_data_iterator import TextIterator

profile = False
weight_aux = 0.0005


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'latent_lstm': ('param_init_lstm', 'latent_lstm_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# orthogonal initialization for weights
# see Saxe et al. ICLR'14
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

import numpy as np
import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes
_data_cache = dict()
def get_cPTB(which_set):
     if which_set not in _data_cache:
         try:
             data = np.load('/data/lisa/data/PennTreebankCorpus/char_level_penntree.npz')
         except:
             data = np.load('char_level_penntree.npz')
         #data = np.load(path)
         # put the entire thing on GPU in one-hot (takes
         # len(self.vocab) * len(self.data) * sizeof(floatX) bytes
         # which is about 1G for the training set and less for the
         # other sets)
         #cudandarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.CudaNdarray
         # (doing it in numpy first because cudandarray doesn't accept
         # lists of indices)
         #one_hot_data = np.eye(len(data["vocab"]), dtype=theano.config.floatX)[data[which_set]]
         #one_hot_data = data[which_set].astype('float32')
         #_data_cache[which_set] = cudandarray(one_hot_data)
     return data[which_set] #_data_cache[which_set]

class PTB(fuel.datasets.Dataset):
     provides_sources = ('features',)
     example_iteration_scheme = None

     def __init__(self, which_set, length, augment=True):
         self.which_set = which_set
         self.length = length
         self.augment = augment
         self.data = get_cPTB(which_set)
         self.num_examples = int(len(self.data) / self.length)
         if self.augment:
             # -1 so we have one self.length worth of room for augmentation
             self.num_examples -= 1
         super(PTB, self).__init__()

     def open(self):
         offset = 0
         if self.augment:
             # choose an offset to get some data augmentation by
             # not always chopping the examples at the same point.
             offset = np.random.randint(self.length)
         # none of this should copy
         data = self.data[offset:]
         # reshape to nonoverlapping examples
         data = (data[:self.num_examples * self.length]
                 .reshape((self.num_examples, self.length))) #self.data.shape[1])))
         # return the data so we will get it as the "state" argument to get_data
         return data

     def get_data(self, state, request):
         if isinstance(request, (tuple, list)):
             request = np.array(request, dtype=np.int64)
             return (state.take(request, 0),)
         return (state[request],)


def get_stream(which_set, batch_size, length, num_examples=None, augment=True):

     dataset = PTB(which_set, length=length, augment=augment)
     if num_examples is None or num_examples > dataset.num_examples:
         num_examples = dataset.num_examples
     streams = fuel.streams.DataStream.default_stream(
         dataset,
         iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
     return streams

# batch preparation, returns padded batch and mask
def prepare_data(seqs_x, maxlen=30, n_words=30000, minlen=0):
     # x: a list of sentences
     lengths_x = [len(s) for s in seqs_x]

     # filter according to mexlen
     if maxlen is not None:
         new_seqs_x = []
         new_lengths_x = []
         for l_x, s_x in zip(lengths_x, seqs_x):
             if True:#l_x < maxlen:
                 new_seqs_x.append(s_x[:maxlen])
                 new_lengths_x.append(min(l_x,maxlen))
         lengths_x = new_lengths_x
         seqs_x = new_seqs_x

         if len(lengths_x) < 1:
             return None, None


     n_samples = len(seqs_x)
 #    maxlen_x = numpy.max(lengths_x) + 1


     x = numpy.zeros((maxlen, n_samples)).astype('int64')
     x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
     for idx, s_x in enumerate(seqs_x):
         x[:lengths_x[idx], idx] = s_x
         x_mask[:lengths_x[idx]+1, idx] = 1.

     return x, x_mask

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])

def param_init_lstm(options,
                     params,
                     prefix='lstm',
                     nin=None,
                     dim=None):
     if nin is None:
         nin = options['dim_proj']

     if dim is None:
         dim = options['dim_proj']

     W = numpy.concatenate([norm_weight(nin,dim),
                            norm_weight(nin,dim),
                            norm_weight(nin,dim),
                            norm_weight(nin,dim)],
                            axis=1)

     params[_p(prefix,'W')] = W
     U = numpy.concatenate([ortho_weight(dim),
                            ortho_weight(dim),
                            ortho_weight(dim),
                            ortho_weight(dim)],
                            axis=1)

     params[_p(prefix,'U')] = U
     params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

     return params

def lstm_layer(tparams, state_below,
                options,
                prefix='lstm',
                mask=None, one_step=False,
                init_state=None,
                init_memory=None,
                nsteps=None,
                **kwargs):

     if nsteps is None:
         nsteps = state_below.shape[0]

     if state_below.ndim == 3:
         n_samples = state_below.shape[1]
     else:
         n_samples = 1

     param = lambda name: tparams[_p(prefix, name)]
     dim = param('U').shape[0]

     if mask is None:
         mask = tensor.alloc(1., state_below.shape[0], 1)

     # initial/previous state
     if init_state is None:
         if not options['learn_h0']:
             init_state = tensor.alloc(0., n_samples, dim)
         else:
             init_state0 = theano.shared(numpy.zeros((options['dim'])),
                                  name=_p(prefix, "h0"))
             init_state = tensor.alloc(init_state0, n_samples, dim)
             tparams[_p(prefix, 'h0')] = init_state0

     U = param('U')
     b = param('b')
     W = param('W')
     non_seqs = [U, b, W]

     # initial/previous memory
     if init_memory is None:
         init_memory = tensor.alloc(0., n_samples, dim)

     def _slice(_x, n, dim):
         if _x.ndim == 3:
             return _x[:, :, n*dim:(n+1)*dim]
         return _x[:, n*dim:(n+1)*dim]

     def _step(mask, sbelow, sbefore, cell_before, *args):
         preact = tensor.dot(sbefore, param('U'))
         preact += sbelow
         preact += param('b')

         i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
         f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
         o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
         c = tensor.tanh(_slice(preact, 3, dim))

         c = f * cell_before + i * c
         c = mask * c + (1. - mask) * cell_before
         h = o * tensor.tanh(c)
         h = mask * h + (1. - mask) * sbefore

         return h, c

     lstm_state_below = tensor.dot(state_below, param('W')) + param('b')
     if state_below.ndim == 3:
         lstm_state_below = lstm_state_below.reshape((state_below.shape[0],
                                                      state_below.shape[1],
                                                      -1))
     if one_step:
         mask = mask.dimshuffle(0, 'x')
         h, c = _step(mask, lstm_state_below, init_state, init_memory)
         rval = [h, c]
     else:
         if mask.ndim == 3 and mask.ndim == state_below.ndim:
             mask = mask.reshape((mask.shape[0], \
                                  mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
         elif mask.ndim == 2:
             mask = mask.dimshuffle(0, 1, 'x')

         rval, updates = theano.scan(_step,
                                     sequences=[mask, lstm_state_below],
                                     outputs_info = [init_state,
                                                     init_memory],
                                     name=_p(prefix, '_layers'),
                                     non_sequences=non_seqs,
                                     strict=True,
                                     n_steps=nsteps)
     return [rval, updates]


def latent_lstm_layer(tparams, state_below,
                options,
                prefix='lstm',
                back_states = None,
                gaussian_s = None,
                latent_tparams = None,
                mask=None, one_step=False,
                init_state=None,
                init_memory=None,
                nsteps=None,
                **kwargs):

     if nsteps is None:
         nsteps = state_below.shape[0]

     if state_below.ndim == 3:
         n_samples = state_below.shape[1]
     else:
         n_samples = 1

     param = lambda name: tparams[_p(prefix, name)]
     dim = param('U').shape[0]

     if mask is None:
         mask = tensor.alloc(1., state_below.shape[0], 1)

     # initial/previous state
     if init_state is None:
         if not options['learn_h0']:
             init_state = tensor.alloc(0., n_samples, dim)
         else:
             init_state0 = theano.shared(numpy.zeros((options['dim'])),
                                  name=_p(prefix, "h0"))
             init_state = tensor.alloc(init_state0, n_samples, dim)
             tparams[_p(prefix, 'h0')] = init_state0

     U = param('U')
     b = param('b')
     W = param('W')
     non_seqs = [U, b, W,  tparams[_p('z_cond', 'W')],
                   latent_tparams[_p('trans_1', 'W')],
                   latent_tparams[_p('trans_1', 'b')],
                   latent_tparams[_p('z_mu', 'W')],
                   latent_tparams[_p('z_mu', 'b')],
                   latent_tparams[_p('z_sigma', 'W')],
                   latent_tparams[_p('z_sigma', 'b')],
                   latent_tparams[_p('inf', 'W')],
                   latent_tparams[_p('inf', 'b')],
                   latent_tparams[_p('inf_mu', 'W')],
                   latent_tparams[_p('inf_mu', 'b')],
                   latent_tparams[_p('inf_sigma', 'W')],
                   latent_tparams[_p('inf_sigma', 'b')],
                   latent_tparams[_p('gen_mu', 'W')],
                   latent_tparams[_p('gen_mu', 'b')],
                   latent_tparams[_p('gen_sigma', 'W')],
                   latent_tparams[_p('gen_sigma', 'b')]]

     # initial/previous memory
     if init_memory is None:
         init_memory = tensor.alloc(0., n_samples, dim)

     def _slice(_x, n, dim):
         if _x.ndim == 3:
             return _x[:, :, n*dim:(n+1)*dim]
         return _x[:, n*dim:(n+1)*dim]

     def _step(mask, sbelow, d_, g_s, sbefore, cell_before,
               U, b, W, W_cond, trans_1_w, trans_1_b,
               z_mu_w, z_mu_b,
               z_sigma_w, z_sigma_b,
               inf_w, inf_b,
               inf_mu_w, inf_mu_b,
               inf_s_w, inf_s_b,
               gen_mu_w, gen_mu_b,
               gen_s_w, gen_s_b):


         p_z = tensor.nnet.softplus(tensor.dot(sbefore, trans_1_w) + trans_1_b)
         z_mu = tensor.dot(p_z, z_mu_w) + z_mu_b
         z_sigma = tensor.dot(p_z, z_sigma_w) + z_sigma_b

         if d_ is not None:
             encoder_hidden = tensor.nnet.softplus(tensor.dot(concatenate([sbefore, d_], axis=1), inf_w) + inf_b)
             encoder_mu = tensor.dot(encoder_hidden, inf_mu_w) + inf_mu_b
             encoder_sigma = tensor.dot(encoder_hidden, inf_s_w) + inf_s_b
             tild_z_t = encoder_mu +  g_s * tensor.exp(0.5 * encoder_sigma)
             kld = gaussian_kld(encoder_mu, encoder_sigma, z_mu, z_sigma)
             kld = tensor.sum(kld, axis=-1)
             decoder_mu = tensor.dot(tild_z_t, gen_mu_w) + gen_mu_b
             decoder_sigma = tensor.dot(tild_z_t, gen_s_w) + gen_s_b
             decoder_mu = tensor.tanh(decoder_mu)
             recon_cost = (tensor.exp(0.5 * decoder_sigma) + tensor.sqr(d_ - decoder_mu)/(2 * tensor.sqr(tensor.exp(0.5 * decoder_sigma))))
             recon_cost = tensor.sum(recon_cost, axis=-1)
         else:
             tild_z_t = z_mu + g_s * tensor.exp(0.5 * z_sigma)
             kld = tensor.sum(tild_z_t, axis=-1) * 0.
             recon_cost = tensor.sum(tild_z_t, axis=-1) * 0.

         z = tild_z_t


         preact = tensor.dot(sbefore, param('U')) +  tensor.dot(z, W_cond)
         preact += sbelow
         preact += param('b')

         i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
         f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
         o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
         c = tensor.tanh(_slice(preact, 3, dim))

         c = f * cell_before + i * c
         c = mask * c + (1. - mask) * cell_before
         h = o * tensor.tanh(c)
         h = mask * h + (1. - mask) * sbefore

         return h, c, z, kld, recon_cost

     lstm_state_below = tensor.dot(state_below, param('W')) + param('b')
     if state_below.ndim == 3:
         lstm_state_below = lstm_state_below.reshape((state_below.shape[0],
                                                      state_below.shape[1],
                                                      -1))
     if one_step:
         mask = mask.dimshuffle(0, 'x')
         h, c = _step(mask, lstm_state_below, init_state, init_memory)
         rval = [h, c]
     else:
         if mask.ndim == 3 and mask.ndim == state_below.ndim:
             mask = mask.reshape((mask.shape[0], \
                                  mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
         elif mask.ndim == 2:
             mask = mask.dimshuffle(0, 1, 'x')

         rval, updates = theano.scan(_step,
                                     sequences=[mask, lstm_state_below, back_states, gaussian_s],
                                     outputs_info = [init_state,
                                                     init_memory, None, None, None],
                                     name=_p(prefix, '_layers'),
                                     non_sequences=non_seqs,
                                     strict=True,
                                     n_steps=nsteps)
     return [rval, updates]


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):

    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux


    return params

def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    return _x[:, n*dim:(n+1)*dim]


def gaussian_kld(mu_left, logvar_left, mu_right, logvar_right):
    gauss_klds = 0.5 * (logvar_right - logvar_left + (tensor.exp(logvar_left) / tensor.exp(logvar_right)) + ((mu_left - mu_right)**2.0 / tensor.exp(logvar_right)) - 1.0)
    return gauss_klds

def gru_layer(tparams, state_below, options, prefix='gru',
              mask=None, one_step=False, init_state=None, **kwargs):
    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = state_below.shape[0]

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_,  h_,          U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    # set initial state to all zeros
    if init_state is None:
        init_state = tensor.unbroadcast(tensor.alloc(0., n_samples, dim), 0)

    if one_step:  # sampling
        rval = _step(*(seqs+[init_state]+shared_vars))
    else:  # training
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state],
                                    non_sequences=shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    rval = [rval, updates]
    return rval

#srng = theano.tensor.shared_randomstreams.RandomStreams(42)
srng = theano.sandbox.rng_mrg.MRG_RandomStreams(42)
def latent_gru_layer(tparams, state_below, options, prefix='gru', back_states = None,
                     gaussian_s = None, mask=None, latent_tparams = None,  one_step=False, init_state=None):

    if one_step:
        assert init_state, 'previous state must be provided'


    nsteps = state_below.shape[0]
    print nsteps

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = state_below.shape[0]

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)


    # state_below is the input word embeddings input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences             |outputs-info | non-seqs
    def _step_slice(m_, x_, xx_, d_, g_s, h_,
            U, Ux, W_cond,
            trans_1_w, trans_1_b,
            z_mu_w, z_mu_b,
            z_sigma_w, z_sigma_b,
            inf_w, inf_b,
            inf_mu_w, inf_mu_b,
            inf_s_w, inf_s_b,
            gen_mu_w, gen_mu_b,
            gen_s_w, gen_s_b):

        p_z = tensor.nnet.softplus(tensor.dot(h_, trans_1_w) + trans_1_b)
        z_mu = tensor.dot(p_z, z_mu_w) + z_mu_b
        z_sigma = tensor.dot(p_z, z_sigma_w) + z_sigma_b

        if d_ is not None:
            encoder_hidden = tensor.nnet.softplus(tensor.dot(concatenate([h_, d_], axis=1), inf_w) + inf_b)
            encoder_mu = tensor.dot(encoder_hidden, inf_mu_w) + inf_mu_b
            encoder_sigma = tensor.dot(encoder_hidden, inf_s_w) + inf_s_b
            #eps_ = srng.normal(size = (50,50))
            tild_z_t = encoder_mu +  g_s * tensor.exp(0.5 * encoder_sigma)
            kld = gaussian_kld(encoder_mu, encoder_sigma, z_mu, z_sigma)
            kld = tensor.sum(kld, axis=-1)
            decoder_mu = tensor.dot(tild_z_t, gen_mu_w) + gen_mu_b
            decoder_sigma = tensor.dot(tild_z_t, gen_s_w) + gen_s_b
            decoder_mu = tensor.tanh(decoder_mu)
            recon_cost = (tensor.exp(0.5 * decoder_sigma) + tensor.sqr(d_ - decoder_mu)/(2 * tensor.sqr(tensor.exp(0.5 * decoder_sigma))))
            recon_cost = tensor.sum(recon_cost, axis=-1)
        else:
            tild_z_t = z_mu + g_s * tensor.exp(0.5 * z_sigma)
            kld = tensor.sum(tild_z_t, axis=-1) * 0.
            recon_cost = tensor.sum(tild_z_t, axis=-1) * 0.

        z = tild_z_t


        preact = tensor.dot(h_, U) + tensor.dot(z, W_cond)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, z, kld, recon_cost

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx, back_states, gaussian_s]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p('z_cond', 'W')],
                   latent_tparams[_p('trans_1', 'W')],
                   latent_tparams[_p('trans_1', 'b')],
                   latent_tparams[_p('z_mu', 'W')],
                   latent_tparams[_p('z_mu', 'b')],
                   latent_tparams[_p('z_sigma', 'W')],
                   latent_tparams[_p('z_sigma', 'b')],
                   latent_tparams[_p('inf', 'W')],
                   latent_tparams[_p('inf', 'b')],
                   latent_tparams[_p('inf_mu', 'W')],
                   latent_tparams[_p('inf_mu', 'b')],
                   latent_tparams[_p('inf_sigma', 'W')],
                   latent_tparams[_p('inf_sigma', 'b')],
                   latent_tparams[_p('gen_mu', 'W')],
                   latent_tparams[_p('gen_mu', 'b')],
                   latent_tparams[_p('gen_sigma', 'W')],
                   latent_tparams[_p('gen_sigma', 'b')]
                  ]

    # set initial state to all zeros
    if init_state is None:
        init_state = tensor.unbroadcast(tensor.alloc(0., n_samples, dim), 0)

    if one_step:
        rval = _step(*(seqs+[init_state]+shared_vars))
        updates = []
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state, None, None, None],
                                    non_sequences=shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    rval = [rval, updates]
    return rval


# initialize all parameters
def init_params(options):
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    params = get_layer('latent_lstm')[0](options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])
    U = numpy.concatenate([norm_weight(options['dim_z'], options['dim']),
                          norm_weight(options['dim_z'], options['dim']),
                          norm_weight(options['dim_z'], options['dim']),
                          norm_weight(options['dim_z'], options['dim'])], axis=1)
    params[_p('z_cond', 'W')] = U

    return params

def init_r_params(options):
    params = OrderedDict()
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm_r',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev_r',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_r',
                                nin=options['dim_word'],
                                nout=options['n_words'])


    return params

def init_latent_params(options):
    params = OrderedDict()
    #Prior Network params
    params = get_layer('ff')[0](options, params, prefix='trans_1', nin=options['dim'], nout=options['prior_hidden'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='z_mu', nin=options['prior_hidden'], nout=options['dim_z'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='z_sigma', nin=options['prior_hidden'], nout=options['dim_z'], ortho=False)

    #Inference network params
    params = get_layer('ff')[0](options, params, prefix='inf', nin = 2 * options['dim'], nout=options['encoder_hidden'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='inf_mu', nin = options['encoder_hidden'], nout=options['dim_z'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='inf_sigma', nin = options['encoder_hidden'], nout=options['dim_z'], ortho=False)

    #Generative Network params
    params = get_layer('ff')[0](options, params, prefix='gen_mu', nin = options['dim_z'], nout=options['dim'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='gen_sigma', nin = options['dim_z'], nout=options['dim'], ortho=False)
    return params

def build_rev_model(tparams, trparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))
    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    #backward rnn
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    embr_shifted = tensor.zeros_like(embr)
    embr_shifted = tensor.set_subtensor(embr_shifted[1:], embr[:-1])
    embr = embr_shifted
    opt_ret['embr'] = embr

    projr = get_layer(options['encoder'])[1](trparams,
                                             embr,
                                             options,
                                             prefix='encoder_r',
                                             mask=xr_mask)
    proj_hr = projr[0][0]
    updates = projr[1]
    opt_ret['proj_hr'] = proj_hr

    logit_lstm = get_layer('ff')[1](trparams, proj_hr, options,
                                    prefix='ff_logit_lstm_r', activ='linear')
    logit_prev = get_layer('ff')[1](trparams, embr, options,
                                    prefix='ff_logit_prev_r', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev)
    logit = get_layer('ff')[1](trparams, logit, options, prefix='ff_logit_r',
                               activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(
        logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    x_flat = xr.flatten()
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost_r = -tensor.log(probs.flatten()[x_flat_idx])
    cost_r = cost_r.reshape([xr.shape[0], xr.shape[1]])
    opt_ret['cost_per_sample'] = cost_r
    cost_r = (cost_r * xr_mask).sum(0)
    states_concat_disc = proj_hr
    get_proj_h = theano.function([x, x_mask], [states_concat_disc])
    return trng, use_noise, x, x_mask, opt_ret, cost_r, states_concat_disc, updates, get_proj_h


# build a training model
def build_model(tparams, trparams, options, x, x_mask, r_states, latent_params):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    #x = tensor.matrix('x_forward', dtype='int64')
    #x_mask = tensor.matrix('x_mask_forward', dtype='float32')
    gaussian_sampled = tensor.matrix('gaussian', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # input
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    opt_ret['emb'] = emb


    r_states = r_states[::-1]
    # pass through gru layer, recurrence here
    proj  = get_layer('latent_lstm')[1](tparams, state_below = emb, options = options,
                                            prefix='encoder',
                                            mask=x_mask,
                                            gaussian_s = gaussian_sampled,
                                            back_states= r_states,
                                            latent_tparams = latent_params)


    proj_h = proj[0][0]
    opt_ret['proj_h'] = proj_h
    updates = proj[1]
    kld, recon_cost = proj[0][3], proj[0][4]
    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit',
                               activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(
        logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # cost
    x_flat = x.flatten()
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost = -tensor.log(probs.flatten()[x_flat_idx])
    cost = cost.reshape([x.shape[0], x.shape[1]])
    opt_ret['cost_per_sample'] = cost
    cost = (cost * x_mask).sum(0)
    if x_mask:
        kld *= x_mask
        recon_cost *= x_mask
    get_kld_recons = theano.function([x, x_mask, r_states, gaussian_sampled], [kld, recon_cost], updates = updates)
    return trng, use_noise, x, x_mask, gaussian_sampled, opt_ret, cost, updates, get_kld_recons, kld, recon_cost, probs



def ELBOcost(weight, kld_cost, kl_rec_cost):
        kl_rec = tensor.sum(kl_rec_cost, axis=-1, keepdims=True)
        kld = tensor.sum(kld_cost, axis=-1, keepdims=True)
        something = theano.function([weight, kld_cost, kl_rec_cost], [weight * kld + weight_aux * kl_rec])
        return something, weight * kld + weight_aux * kl_rec



# build a sampler
#def build_model(tparams, trparams, options, x, x_mask, r_states, latent_params):
def build_sampler(tparams, options, trng, latent_tparams):
    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')
    gaussian_sampled = tensor.matrix('gaussian', dtype='float32')

    # if it's the first word, emb should be all zero
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                        tparams['Wemb'][y])

    # apply one step of gru layer
    proj = get_layer('latent_lstm')[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=None,
                                            one_step=True,
                                            gaussian_s = gaussian_sampled,
                                            back_states = None,
                                            latent_tparams = latent_tparams,
                                            init_state=init_state)
    next_state = proj[0][0]

    # compute the output probability dist and sample
    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    print 'Building f_next..',
    inps = [y, init_state, gaussian_sampled]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_next


# generate sample
def gen_sample(tparams, f_next, options, trng=None, maxlen=30, argmax=False):

    sample = []
    sample_score = 0

    # initial token is indicated by a -1 and initial state is zero
    next_w = -1 * numpy.ones((1,)).astype('int64')
    next_state = numpy.zeros((1, options['dim'])).astype('float32')

    for ii in xrange(maxlen):
        inps = [next_w, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if argmax:
            nw = next_p[0].argmax()
        else:
            nw = next_w[0]
        sample.append(nw)
        sample_score += next_p[0, nw]
        if nw == 0:
            break

    return sample, sample_score


# calculate the log probablities on a given corpus using language model
def pred_probs(f_log_probs, prepare_data, options, iterator, get_proj_h, verbose=True):
    probs = []

    n_done = 0

    for data in iterator.get_epoch_iterator():
        x = data[0].T
        x_mask = np.ones((options['maxlen'], options['batch_size'])).astype('float32')
        n_done += len(x)
        if x.shape[1] is not options['batch_size']:
            continue
        h_state = get_proj_h(x, x_mask)
        h_state = numpy.asarray(h_state).astype('float32')
        h_state = numpy.asarray(h_state).reshape(options['maxlen'], options['batch_size'], options['dim']).astype('float32')
        qwes = numpy.random.normal(loc=0.0, scale=1.0, size=(options['maxlen'], options['dim_z'])).astype('float32')

        # compute cost, grads and copy grads to shared variables
        pprobs = f_log_probs(x, x_mask, h_state, qwes)

        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost, beta1=0.9, beta2=0.999, e=1e-8):

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    updates = []

    t_prev = theano.shared(numpy.float32(0.))
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2**t) / (1. - beta1**t)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g**2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of GRU units
          encoder='lstm',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 weight decay penalty
          lrate=0.01,
          n_words=100000,  # vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq
          dataset='/data/lisatmp3/chokyun/wikipedia/extracted/wiki.tok.txt.gz',
          valid_dataset='../data/dev/newstest2011.en.tok',
          dictionary='/data/lisatmp3/chokyun/wikipedia/extracted/'
          'wiki.tok.txt.gz.pkl',
          use_dropout=False,
          reload_=False, kl_start = 0.2, kl_rate  = 0.0003):

    prior_hidden = 1200
    dim_z = 128
    encoder_hidden = 1200
    learn_h0 = False
    saveto = saveto + 'model_' + str(weight_aux) + '_weight_aux_' +  str(kl_start) + '_kl_Start_' + str(kl_rate) +  '_kl_rate.npz'
    # Model options
    model_options = locals().copy()
    train = get_stream('train', model_options['batch_size'], model_options['maxlen'], num_examples=None, augment=False)
    valid = get_stream('valid', model_options['batch_size'], model_options['maxlen'], num_examples=None, augment=False)

    print 'Building model'
    params = init_params(model_options)
    r_params = init_r_params(model_options)
    latent_params = init_latent_params(model_options)

    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    # create shared variables for parameters
    tparams = init_tparams(params)
    trparams = init_tparams(r_params)
    latent_tparams = init_tparams(latent_params)

    # build the symbolic computational graph
    trng, use_noise, \
        x_r, x_mask_r, \
        opt_ret, \
        cost_r, states_concat_disc, updates_1, get_proj_h = \
        build_rev_model(tparams, trparams, model_options)


    trng, use_noise, \
        x, x_mask, gaussian_sampled, \
        opt_ret, \
        cost, updates_2, get_kld_recons, kld, recon_cost, probs = \
        build_model(tparams, trparams, model_options, x_r, x_mask_r,states_concat_disc, latent_tparams)

    print 'Buliding sampler'
    #f_next = build_sampler(tparams, model_options, trng, latent_tparams)


    weight_f = tensor.scalar(name='weight_f')
    something_fn, vae_cost = ELBOcost(weight_f, kld, recon_cost)

    inps = [x, x_mask, states_concat_disc, gaussian_sampled]
    inps_r = [x, x_mask]
    inps_tot = [x, x_mask, states_concat_disc, gaussian_sampled, weight_f]
    print 'Building f_log_probs...',

    f_log_probs = theano.function(inps, cost, updates=updates_2, profile=profile)
    f_log_probs_r = theano.function(inps_r, cost_r, updates = updates_1,  profile=profile)
    f_log_probs_vae = theano.function(inps_tot, vae_cost, profile=profile)
    print 'Done'

    tot_cost = (cost +vae_cost).mean()
    vae_cost = vae_cost.mean()
    cost_r = cost_r.mean()
    cost = cost.mean()

    # after any regularizer - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost_total = theano.function(inps_tot, tot_cost, profile=profile)
    f_cost = theano.function(inps, cost, profile=profile)
    f_cost_r = theano.function(inps_r, cost_r, updates = updates_1, profile=profile)
    f_cost_vae = theano.function(inps_tot, vae_cost, updates = updates_1, profile=profile)
    print 'Done'

    print 'Computing gradient...',

    grads = tensor.grad(tot_cost, itemlist(tparams))
    r_grads = tensor.grad(cost_r, wrt=itemlist(trparams))
    vae_grads = tensor.grad(vae_cost, wrt=itemlist(latent_tparams))
    print 'Done'

    clip_grad = 1
    cgrads = [tensor.clip(g,-clip_grad, clip_grad) for g in grads]
    grads = cgrads


    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    lr2 = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared_r, f_update_r = eval(optimizer)(lr2, trparams, r_grads, inps_r, cost_r)
    print 'optimizer 1 done'
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps_tot, tot_cost)
    f_grad_shared_vae, f_update_vae = eval(optimizer)(lr, latent_tparams, vae_grads, inps_tot, vae_cost)
    print 'Done'

    print 'Optimization'

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    # Training loop
    uidx = 0
    estop = False
    bad_counter = 0
    kl_start = model_options['kl_start']
    kl_rate = model_options['kl_rate']

    for eidx in xrange(max_epochs):
        n_samples = 0

        for data in train.get_epoch_iterator():
            x = data[0].T
            x_mask = np.ones((model_options['maxlen'], model_options['batch_size'])).astype('float32')
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)
            # pad batch and create mask
            #x, x_mask = prepare_data(x, maxlen=maxlen, n_words=n_words)
            #if uidx %10 ==0:
            if kl_start < 1.:
                kl_start += kl_rate

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue
            if x.shape[1] is not model_options['batch_size']:
                uidx -= 1
                continue
            ud_start = time.time()
            h_state = get_proj_h(x, x_mask)
            h_state = numpy.asarray(h_state).reshape(model_options['maxlen'], model_options['batch_size'], model_options['dim']).astype('float32')

            # compute cost, grads and copy grads to shared variables
            qwes = numpy.random.normal(loc=0.0, scale=1.0, size=(model_options['maxlen'], model_options['dim_z'])).astype('float32')

            total_cost = f_grad_shared(x, x_mask, h_state, qwes, kl_start)
            cost_r = f_grad_shared_r(x, x_mask)
            vae_cost = f_grad_shared_vae(x, x_mask, h_state, qwes, kl_start)
            kld, recons = get_kld_recons(x, x_mask, h_state, qwes)
            cost = f_cost(x, x_mask, h_state, qwes)

            f_update(lrate)
            f_update_r(lrate)
            f_update_vae(lrate)
            ud = time.time() - ud_start

            # check for bad numbers
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Total Cost ', total_cost, 'Cost_r', cost_r, 'VAE_Cost', vae_cost, 'LL COST', cost, 'KL_start', kl_start

            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data, model_options, valid, get_proj_h)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print 'Valid ', valid_err

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples
        #valid_errs = pred_probs(f_log_probs, prepare_data, model_options, valid, get_proj_h)
        #valid_err = valid_errs.mean()
        #print 'VError', valid_err

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data,
                           model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                **params)

    return valid_err


if __name__ == '__main__':
    pass
