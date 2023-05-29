

#==============================================================================
#===  DATA LOAD AND PREP  =====================================================
#==============================================================================

''' prep binary classification data for mnist DIG_As vs DIG_Bs
'''

import numpy as np
import tqdm
from keras.datasets import mnist
np.random.seed(686)
DIG_A, DIG_B = 4, 9
SIDE = 28
MAX_PIX_VAL = 255
NB_TRAIN = 1000

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train is N_ x SIDE x SIDE array of integers [0, MAX_PIX_VAL]
# y_train is N_               array of integers [0, 10)
#
# where N_ = 60000

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  keep only two digits of interest  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def filter_digits(xs, ys):
    indices = np.logical_or(
        np.equal(ys, DIG_A),
        np.equal(ys, DIG_B)
        )
    xs = xs[indices]
    ys = ys[indices]
    return xs, ys
x_train, y_train = filter_digits(x_train, y_train)
x_test, y_test = filter_digits(x_test, y_test)
#
# x_train is N x SIDE x SIDE array of integers [0, MAX_PIX_VAL]
# y_train is N               array of integers {DIG_A, DIG_B}
#
# where N ~ 12000

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  normalize pixel intensities  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_train = x_train / float(MAX_PIX_VAL)
x_test = x_test / float(MAX_PIX_VAL)
#
# x_train is N x SIDE x SIDE array of floats   [0., 1.]
# y_train is N               array of integers {DIG_A, DIG_B}
#
# where N ~ 12000

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  shuffle data  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def shuffle(xs, ys):
    indices = np.arange(len(xs))
    np.random.shuffle(indices) # mutating shuffle
    xs = xs[indices]
    ys = ys[indices]
    return xs,ys
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

#-------  pare down training set  ---------------------------------------------
x_train = x_train[:NB_TRAIN]
y_train = y_train[:NB_TRAIN]

#-------  add noise to pixel intensities  -------------------------------------
x_train = x_train + np.random.randn(*x_train.shape)
x_train = np.maximum(0., np.minimum(1., x_train))

#-------  shape  --------------------------------------------------------------
assert len(x_train)==len(y_train)
N = len(x_train)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  sanity checks  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#close_enough = lambda a, b : abs(b-a)<1e-6
close_enough = lambda a, b : np.linalg.norm(np.array(b-a).flatten())<1e-6

assert x_train.shape == (N, SIDE, SIDE)
assert y_train.shape == (N,)
assert set(y_train) == {DIG_A, DIG_B}
assert close_enough(np.min(x_train), 0.)
assert close_enough(np.max(x_train), 1.)
assert abs(N-min(NB_TRAIN, 12000))<500

print('prepped {} training examples'.format(N))

#==============================================================================
#===  METRICS FOR SUCCESS  ====================================================
#==============================================================================

''' acc, loss

    prob p represents model's prob mass for DIG_B

    by `predictor` I mean a function that takes in an image and gives a prob
'''

def accuracy(predicted_labels, true_ys):
    return np.mean([1. if l==y else 0.
                    for l,y in zip(predicted_labels, true_ys)])

def cross_entropy_loss(predicted_probs, true_ys):
    return np.mean([ - np.log(p if y==DIG_B else 1.-p)
                    for p,y in zip(predicted_probs, true_ys)])

def judge(predictor, xs, ys, verbose=False):
    xs = tqdm.tqdm(xs) if verbose else xs
    probs = [predictor(x) for x in xs]
    labels = [DIG_B if p>.5 else DIG_A for p in probs]
    acc  = accuracy(labels, ys)
    loss = cross_entropy_loss(probs, ys)
    return {'acc':acc, 'loss':loss}

#-------  sanity checks using placeholder predictors  -------------------------
very_sure_A = lambda x : .01
very_sure_B = lambda x : .99
maybe_its_A = lambda x : .4
maybe_its_B = lambda x : .6
fifty_fifty = lambda x : .5

vsa = judge(very_sure_A, x_train, y_train)['acc']
vsb = judge(very_sure_B, x_train, y_train)['acc']
assert close_enough(vsa+vsb, 1.)

vsa = judge(very_sure_A, x_train[:1], [DIG_A])['acc']
vsb = judge(very_sure_A, x_train[:1], [DIG_B])['acc']
assert close_enough(vsa, 1.)
assert close_enough(vsb, 0.)

vsa = judge(very_sure_A, x_train, y_train)['loss']
vsb = judge(very_sure_B, x_train, y_train)['loss']
mia = judge(maybe_its_A, x_train, y_train)['loss']
mib = judge(maybe_its_B, x_train, y_train)['loss']
ffl = judge(fifty_fifty, x_train, y_train)['loss']
assert ffl < mia < vsa
assert ffl < mib < vsb
assert close_enough(ffl, np.log(2))

#==============================================================================
#===  LINEAR MODEL  ===========================================================
#==============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Manipulate Weights  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

linear_init = lambda : np.random.randn(SIDE*SIDE) / np.sqrt(SIDE*SIDE)

def linear_displace(w, coef, g):
    return w + coef * g

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Forward Model  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

clip = lambda z : np.maximum(-15., np.minimum(+15., z))
sigmoid = lambda z : 1./(1.+np.exp(-clip(z)))
def linear_predict(w, x):
    return sigmoid(w.dot(x.flatten()))

#-------  sanity checks  ------------------------------------------------------

w = linear_init()

vsa = judge(lambda x : linear_predict(+w, x), x_train, y_train)['acc']
vsb = judge(lambda x : linear_predict(-w, x), x_train, y_train)['acc']
assert close_enough(vsa+vsb, 1.)

ffl = judge(lambda x : linear_predict(0*w, x), x_train, y_train)['loss']
assert close_enough(ffl, np.log(2))

x = w.reshape(SIDE,SIDE)
vsa = judge(lambda x : linear_predict(w, x), [x], [DIG_A])['acc']
vsb = judge(lambda x : linear_predict(w, x), [x], [DIG_B])['acc']
assert close_enough(vsa, 0.)
assert close_enough(vsb, 1.)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Backward pass  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
''' For given x,y, we want derivative (with respect w) of
        l(w) = loss(sigmoid(w.dot(x)), y)
             = loss_at_y(sigmoid(dot_with_x(w)))
        where loss_at_y(p) = -log(p if y==DIG_B else 1-p)
        where sigmoid(z)   = 1/(1+exp(-z))
        where dot_with_x(w)= w.dot(x)
    By _CHAIN_RULE_:
        l'(w) = (
              loss_at_y'(sigmoid(dot_with_x(w)))
            * sigmoid'(dot_with_x(w))
            * dot_with_x'(w)
        ) = (
              loss_at_y'(p)
            * sigmoid'(z)
            * dot_with_x'(w)
        )
        where z = dot_with_x(w)
        where p = sigmoid(z)
    NOTE: appearance of terms from forward pass!
'''

def linear_backprop_unsimp(w, x, y):
    z = w.dot(x.flatten())
    p = sigmoid(z)
    #
    # this is correct...
    dl_dp = - (+1 if y==DIG_B else -1)/(p if y==DIG_B else 1-p)
    dp_dz = p * (1-p)
    dz_dw = x.flatten()
    #
    dl_dw = dl_dp * dp_dz * dz_dw
    return dl_dw

def linear_backprop(w, x, y):
    z = w.dot(x.flatten())
    p = sigmoid(z)
    #
    # ... and so is this...
    '''
        dl_dp = -1/p if y==DIG_B else +1/(1-p)
        dp_dz = p * (1-p)
        dl_dz = dl_dp * dp_dz = -(1-p) if y==DIG_B else +p
    '''
    # interpret dl_dz as error of p as estimator of one-hot version of y
    dl_dz = p - (1 if y==DIG_B else 0)
    dz_dw = x.flatten()
    #
    dl_dw = dl_dz * dz_dw
    return dl_dw

#-------  sanity checks  ------------------------------------------------------

for _ in range(10):
    w = linear_init()
    idx = np.random.randint(N)
    x = x_train[idx]
    y = y_train[idx]

    # check that simplification preserved answer
    g_unsimp = linear_backprop_unsimp(w, x, y)
    g        = linear_backprop       (w, x, y)
    assert close_enough(g_unsimp, g)

    # do a step of gradient descent, check loss decreased
    before = judge(lambda xx: linear_predict(w, xx), [x], [y])['loss']
    w = w - .01 * g
    after  = judge(lambda xx: linear_predict(w, xx), [x], [y])['loss']
    assert after < before

#==============================================================================
#===  VANILLA MODEL  ==========================================================
#==============================================================================
''' what architecture?  well, let's use this one:

    lrelu(z) = max(z/10, z)

    x
   h0 ---------> z1 ---> h1 ---------> z2 ---> h2 ---------> z3 ------> p
    |
    |             |       |
    |             |       |             |       |
    |     C*      | lrelu |     B*      | lrelu |     A*      | sigmoid |
    |             |       |             |       |
    |             |       |                     1
    |                     1
    1
    D0            D1      D1            D2      D2            D3        1
    SIDE*SIDE     32                    32                     1
'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Weight Helpers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

D0 = SIDE*SIDE
D1 = 32
D2 = 32
D3 = 1
def vanilla_init():
    A = np.random.randn(    D2) / np.sqrt( 1 + D2)
    B = np.random.randn(D2, D1) / np.sqrt(D2 + D1)
    C = np.random.randn(D1, D0) / np.sqrt(D1 + D0)
    return (A,B,C)

def vanilla_displace(abc, coef, g):
    A,B,C = abc
    gA,gB,gC = g
    return (A + coef * gA,
            B + coef * gB,
            C + coef * gC )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Forward pass  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

lrelu     = lambda z : np.maximum(z/10, z)
step      = lambda z : np.heaviside(z, .5)
dlrelu_dz = lambda z : .1 + (1.-.1)*step(z)
def vanilla_predict(abc, x):
    A,B,C = abc

    h0 = x.flatten()
    #
    z1 = C.dot(h0)
    h1 = lrelu(z1)
    #
    z2 = B.dot(h1)
    h2 = lrelu(z2)  # this is our learned featurization
    #
    z3 = A.dot(h2)  # linear classifier!
    p = sigmoid(z3)
    return p

#-------  checks  -------------------------------------------------------------

A,B,C = vanilla_init()

# check that linear layer makes sense:
vsa = judge(lambda x : vanilla_predict((+A,B,C), x), x_train, y_train)['acc']
vsb = judge(lambda x : vanilla_predict((-A,B,C), x), x_train, y_train)['acc']
assert close_enough(vsa+vsb, 1.)

ffl = judge(lambda x : vanilla_predict((0*A,B,C), x), x_train, y_train)['loss']
assert close_enough(ffl, np.log(2))

# check end-to-end positivity
x = x_train[0]
y = y_train[0]
A = np.abs(A)
B = np.abs(B)
C = np.abs(C)
acc_ppp = judge(lambda x : vanilla_predict((A, B, C), x), [x], [DIG_B])['acc']
acc_ppn = judge(lambda x : vanilla_predict((A, B,-C), x), [x], [DIG_B])['acc']
acc_pnp = judge(lambda x : vanilla_predict((A,-B, C), x), [x], [DIG_B])['acc']
acc_pnn = judge(lambda x : vanilla_predict((A,-B,-C), x), [x], [DIG_B])['acc']
assert close_enough(acc_ppp, 1.)
assert close_enough(acc_ppn, 0.)
assert close_enough(acc_pnp, 0.)
assert close_enough(acc_pnn, 1.)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Backward pass  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def vanilla_backprop(abc, x, y):
    A,B,C = abc

    h0 = x.flatten()
    #
    z1 = C.dot(h0)
    h1 = lrelu(z1)
    #
    z2 = B.dot(h1)
    h2 = lrelu(z2)  # this is our learned featurization
    #
    z3 = A.dot(h2)  # linear classifier!
    p = sigmoid(z3)

    dl_dz3 = p - (1 if y==DIG_B else 0)
    dl_dh2 = dl_dz3 * A
    dl_dz2 = dl_dh2 * dlrelu_dz(z2)
    dl_dh1 = dl_dz2.dot(B)
    dl_dz1 = dl_dh1 * dlrelu_dz(z1)

    dl_dA = dl_dz3 * h2
    dl_dB = np.outer(dl_dz2, h1)
    dl_dC = np.outer(dl_dz1, h0)

    return (dl_dA, dl_dB, dl_dC)

#-------  checks  -------------------------------------------------------------

for _ in range(10):
    abc= vanilla_init()
    idx = np.random.randint(N)
    x = x_train[idx]
    y = y_train[idx]

    # do a step of gradient descent, check loss decreased
    before = judge(lambda xx: vanilla_predict(abc, xx), [x], [y])['loss']
    g = vanilla_backprop(abc,x,y)
    abc = vanilla_displace(abc, -.01, g)
    after  = judge(lambda xx: vanilla_predict(abc, xx), [x], [y])['loss']
    assert after < before


#==============================================================================
#===  CONV MODEL  =============================================================
#==============================================================================

''' architecture

    in the chart below, we transform inputs (top) to outputs (bottom)

                    height x width x channels
    x                   28 x 28 x 1
        avgpool                                     2x2
    h0                  14 x 14 x 1
        conv                            weight C    5x5x8x1     stride 2x2
    z1                   5 x  5 x 8
        lrelu
    h1                   5 x  5 x 8
        conv                            weight B    1x1x4x8     stride 1x1
    z2                   5 x  5 x 4
        lrelu
    h2                   5 x  5 x 4
        dense                           weight A    1x(5*5*4)
    z3                            1
        sigmoid
    p                             1
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Weights Helpers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def conv_init():
    A = np.random.randn(  5*5*4) / np.sqrt( 1 + 5*5*4)
    B = np.random.randn(1,1,4,8) / np.sqrt( 4 + 1*1*8)
    C = np.random.randn(5,5,8,1) / np.sqrt( 8 + 5*5*1)
    return (A,B,C)

def conv_displace(abc, coef, g):
    A,B,C = abc
    gA,gB,gC = g
    return (A + coef * gA,
            B + coef * gB,
            C + coef * gC )




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Building Blocks  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def avgpool2x2(x):
    H,W,C = x.shape
    # return an array of shape (H/2 x W/2 x C)
    return ( x[0:H:2, 0:W:2]
            +x[0:H:2, 1:W:2]
            +x[1:H:2, 0:W:2]
            +x[1:H:2, 1:W:2])/4.

def conv(x, weights, stride=1):
    H,W,C = x.shape
    KH,KW,OD,ID = weights.shape
    assert C==ID
    HH, WW = int((H-KH+1)/stride),   int((W-KW+1)/stride)
    # return an array of shape HH x WW x OD
    return np.array(
      [[
         np.tensordot(
          weights         ,   # KH x KW x OD x ID
          x[h:h+KH,w:w+KW],   # KH x KW      x ID
          ((0,1,3),(0,1,2))
         )
         for w in range(0,WW*stride,stride)]
        for h in range(0,HH*stride,stride)  ]
    )

#-------  sanity checks for forward  ------------------------------------------

# scaling and shape tests
aa = np.ones((8,12,7))
pp = np.ones((4,6,7))
assert close_enough(avgpool2x2(aa), pp)
ww = np.ones((3,3,5,7))
cc = (3*3*7)*np.ones((6,10,5))
assert close_enough(conv(aa,ww,stride=1), cc)

# orientation test
bb = np.array([1*np.eye(4),3*np.eye(4)]) # 2 x 4 x 4
'''
  bb == [
    [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
    [[3,0,0,0], [0,3,0,0], [0,0,3,0], [0,0,0,3]]
    ]]
'''
pp = np.array([[[1,1,0,0],[0,0,1,1]]])
assert close_enough(avgpool2x2(bb),pp)
ww = np.zeros((2,2,1,4))
ww[0,0,:,:] = 1+np.arange(4)
cc = np.array([1,2,3])[np.newaxis,:,np.newaxis] # shape 1 x 3 x 1
assert close_enough(conv(bb,ww,stride=1),cc)

print('hooray!')

#-------  derivatives  --------------------------------------------------------

''' Why do we want dconv(x,w)/dw?  So and only so that we can compute
    dl/dw from dl/dconv(x,w).  We ease our lives by writing a function
    that directly gives dl/dw from dl/dconv(x,w).
'''
def Dw_conv(x,weights_shape,dl_dconv,stride=1):
    H,W,C = x.shape
    KH,KW,OD,ID = weights_shape
    assert C==ID
    HH, WW = int((H-KH+1)/stride),   int((W-KW+1)/stride)
    assert dl_dconv.shape == (HH,WW,OD)
    # return an array of shape KH x KW x OD x ID
    HS, WS = HH*stride, WW*stride
    dl_dw = np.array(
            [[np.tensordot(
                dl_dconv                          ,# HH x WW x OD
                x[dh:dh+HS:stride,dw:dw+WS:stride],# HH x WW x ID
                ((0,1),(0,1))
                )
              for dw in range(KW)]
             for dh in range(KH)]
            )
    return dl_dw

''' Why do we want dconv(x,w)/dx?  So and only so that we can compute
    dl/dx from dl/dconv(x,w).  We ease our lives by writing a function
    that directly gives dl/dx from dl/dconv(x,w).
'''

def Dx_conv(x_shape, weights, dl_dconv, stride):
    H,W,C = x_shape
    KH,KW,OD,ID = weights.shape
    assert C==ID
    HH, WW = int((H-KH+1)/stride),   int((W-KW+1)/stride)
    assert dl_dconv.shape == (HH,WW,OD)
    # return H x W x ID
    dl_dx = np.zeros((H,W,ID),dtype=np.float32)
    for h in range(KH):
        for w in range(KW):
            dl_dx[h:h+HH*stride:stride,w:w+WW*stride:stride] += (
                np.tensordot(
                    dl_dconv,       # HHxWWxOD
                    weights[h,w],   # OD x ID
                    ((2,),(0,))
                    )
                )
    return dl_dx

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Forward pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

''' architecture

    in the chart below, we transform inputs (top) to outputs (bottom)

                    height x width x channels
    x                   28 x 28 x 1
        avgpool                                     2x2
    h0                  14 x 14 x 1
        conv                            weight C    5x5x8x1     stride 2x2
    z1                   5 x  5 x 8
        lrelu
    h1                   5 x  5 x 8
        conv                            weight B    1x1x4x8     stride 1x1
    z2                   5 x  5 x 4
        lrelu
    h2                   5 x  5 x 4
        dense                           weight A    (5*5*4)
    z3                            1
        sigmoid
    p                             1
'''
def conv_predict(abc, x):
    A,B,C = abc

    h0 = avgpool2x2(x[:,:,np.newaxis])
    #
    z1 = conv(h0, C, stride=2)
    h1 = lrelu(z1)
    #
    z2 = conv(h1, B, stride=1)
    h2 = lrelu(z2)
    #
    z3 = A.dot(h2.flatten())
    #
    p = sigmoid(z3)

    return p

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Backward pass~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def conv_backprop(abc, x, y):
    A,B,C = abc

    h0 = avgpool2x2(x[:,:,np.newaxis])
    #
    z1 = conv(h0, C, stride=2)
    h1 = lrelu(z1)
    #
    z2 = conv(h1, B, stride=1)
    h2 = lrelu(z2)
    #
    z3 = A.dot(h2.flatten())
    #
    p = sigmoid(z3)

    dl_dz3 = p - (1 if y==DIG_B else 0)
    dl_dh2 = dl_dz3 * A.reshape(h2.shape)
    dl_dz2 = dl_dh2 * dlrelu_dz(z2)
    dl_dh1 = Dx_conv(h1.shape, B, dl_dz2, stride=1)
    dl_dz1 = dl_dh1 * dlrelu_dz(z1)

    dl_dA = dl_dz3 * h2.flatten()
    dl_dB = Dw_conv(h1, B.shape, dl_dz2, stride=1)
    dl_dC = Dw_conv(h0, C.shape, dl_dz1, stride=2)

    return (dl_dA, dl_dB, dl_dC)

#==============================================================================
#===  TRAINING LOOP  ==========================================================
#==============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Training Parameters  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

T = 15001
DT = 1000
LEARNING_RATE = .01
ANNEAL_T = 4000
DRAG_COEF = .1

idx = 0
def next_training_example():
    global idx, x_train, y_train
    xy = x_train[idx], y_train[idx]
    idx += 1
    if idx==N:
        idx=0
        x_train, y_train = shuffle(x_train, y_train)
    return xy

#-------  interface with model  -----------------------------------------------


FUNCS_BY_MODEL = {
 'linear':(linear_init, linear_backprop, linear_displace, linear_predict),
 'vanilla':(vanilla_init, vanilla_backprop, vanilla_displace, vanilla_predict),
 'conv':(conv_init, conv_backprop, conv_displace, conv_predict),
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  SGD: the __ENGINE__ of learning!  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for MODEL in ('linear', 'vanilla', 'conv'):
    print('\n'*4)
    print(MODEL)
    print('\n'*2)

    INIT,BACK,DISP,PRED = FUNCS_BY_MODEL[MODEL]

    w = INIT()
    m = DISP(w, -1., w) # hacky way to set m=0 of same shape as w
    for t in range(T):
        x, y = next_training_example()
        g = BACK(w, x, y)
        LR = LEARNING_RATE * float(ANNEAL_T) / (ANNEAL_T + t)
        m = DISP(m, -DRAG_COEF, m) # m forgets a bit of its past
        m = DISP(m, +1., g) # add gradient to momentum
        w = DISP(w, -LR, m) # update based on momentum

        if t%DT : continue

        xs = x_train[-1000:]
        ys = y_train[-1000:]
        mstr = judge(lambda x: PRED(w, x), xs, ys)
        xs = x_test[-1000:]
        ys = y_test[-1000:]
        mste = judge(lambda x: PRED(w, x), xs, ys)
        print('at step {:6d}'.format(t)
              +' tr acc {:4.2f} loss {:5.3f}'.format(mstr['acc'], mstr['loss'])
              +' te acc {:4.2f} loss {:5.3f}'.format(mste['acc'], mste['loss'])
              )

    xs = x_train[:]
    ys = y_train[:]
    mstr = judge(lambda x: PRED(w, x), xs, ys, verbose=True)
    xs = x_test[:]
    ys = y_test[:]
    mste = judge(lambda x: PRED(w, x), xs, ys, verbose=True)
    print('after all training'
          +' tr acc {:4.2f} loss {:5.3f}'.format(mstr['acc'], mstr['loss'])
          +' te acc {:4.2f} loss {:5.3f}'.format(mste['acc'], mste['loss'])
          )
