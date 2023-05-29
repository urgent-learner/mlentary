''' author: sam tenka
    change: 2023-05-29
    create: 2023-05-29
    descrp: Generate plots for the text's digit-classification examples.
    depend: keras
    jargon: we'll consistently use these abbreviations when naming variables:
                dec_func    --- decision function
                idx(s)      --- index/indices within list of all examples
                model       --- name of model ('linear', 'affine', etc)
                nb_         --- number of (whatever follows the underscore)
                side        --- sidelength of image, measured in pixels
                x           --- photo of handwritten digit
                y           --- digit-valued label
                y_sign      --- {-1,+1}-valued label
                z           --- feature vector
                vert        --- to do with a graph's vertical axis
                hori        --- to do with a graph's horizontal axis
    thanks: featurization idea inspired by abu-mostafa's book
    to use: Run `python3 mnist_models.py`.  On first run, expect a downloading
            progress bar to display and finish within 30 to 60 seconds; this
            is for downloading the MNIST dataset we'll use.
'''

#==============================================================================
#===  PREAMBLE  ===============================================================
#==============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Universal Constants  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#-------  imports  ------------------------------------------------------------

from keras.datasets import mnist, fashion_mnist
from matplotlib import pyplot as plt
import numpy as np
import tqdm

#-------  colors  -------------------------------------------------------------

WHITE        = np.array([1.0 ,1.0 ,1.0 ])
SMOKE        = np.array([ .9 , .9 , .9 ])
SLATE        = np.array([ .5 , .5 , .5 ])
SHADE        = np.array([ .1 , .1 , .1 ])
BLACK        = np.array([ .0 , .0 , .0 ])

BLUE    = np.array([0.05, 0.55, 0.85]) ###
ORANGE  = np.array([0.95, 0.65, 0.05]) #

def overlay_color(background, foreground, foreground_opacity=1.0):
    background += foreground_opacity * (foreground - background)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Global Parameters for ...  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#-------  ... reading and parsing  --------------------------------------------

DIG_A, DIG_B = 1, 3
DIG_SIDE = 28
MAX_PIX_VAL = 255

PRINT_TRAIN_FEATURES = False

#-------  ... preparing data  -------------------------------------------------

NB_TEST      = 400
NB_TRAIN     =  20
NB_TRAIN_MAX = 200
DARKNESS_THRESH = 0.5

FEAT_NMS = ['brightness', 'width']
DEGS = [0,1]

TRY_OUT_HAND_MADE_CLASSIFIER = False

#-------  ... learning  -------------------------------------------------------

L2_REG = 0.000

P_RANGE = 10.0
P_STEP  =   .25
PARAM_RANGE = np.linspace(-P_RANGE, P_RANGE, int((P_RANGE*2)/P_STEP+1))
MODEL = 'affine'

BRUTE_FORCE_SEARCH = True

#-------  ... plotting and writing  -------------------------------------------

NB_DIGITS_RENDERED = 25
PLT_SIDE = 320
MARG     = 4

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Global Initialization  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#-------  parameterize randomness for replicability  --------------------------

np.random.seed(1)

#-------  format of example data  ---------------------------------------------

DIG_1COORS = np.arange(DIG_SIDE)
DIG_2COORS = [(row,col) for row in DIG_1COORS for col in DIG_1COORS]

#-------  learning helpers  ---------------------------------------------------

FEAT_DIM = sum(len(FEAT_NMS)**d for d in DEGS)

if BRUTE_FORCE_SEARCH:
    BRUTE_PARAMS = [(0, aa, bb) for aa in PARAM_RANGE for bb in PARAM_RANGE]
    #BRUTE_PARAMS = [()]
    #for _ in range(FEAT_DIM):
    #    BRUTE_PARAMS = [p+(val,) for val in PARAM_RANGE for p in BRUTE_PARAMS]

#==============================================================================
#===  LOAD, PREPARE DATA  =====================================================
#==============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  1.0. Load Dataset Labels and Pixel Values  ~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  1.0.0. read images and labels from disk  ----------------------

# this line loads the (x=digitphotos, y=whichdigit) data set into memory
(all_x, all_y), _ = mnist.load_data()
# all_x[1000,14,14] == some floating point number from 0 to 255

#--------------  1.0.1. filter to only digits of interest  ---------------------

# DIG_A = 1
# DIG_B = 9
idxs = np.logical_or(np.equal(all_y, DIG_A), np.equal(all_y, DIG_B))
all_x = all_x[idxs]
all_y = all_y[idxs]

# now all_x contains only pictures of DIG_As and DIG_Bs (e.g. 1s and 9s)
# and all_y contains only             DIG_As and DIG_Bs (e.g. 1s and 9s)
# and they continue to correspond, e.g. all_ys[7] describes all_xs[7]

print('all_y has shape {}'.format(all_y.shape))
print('all_x has shape {}'.format(all_x.shape))

#--------------  1.0.2. normalize  ---------------------------------------------

all_x = (1.0/MAX_PIX_VAL) * all_x

#--------------  1.0.3. shuffle and split  -------------------------------------

idxs = np.arange(len(all_y))
np.random.shuffle(idxs)
all_x = all_x[idxs]
all_y = all_y[idxs]
all_y_signs = np.array([+1 if y==DIG_B else -1 for y in all_y])

train_idxs = np.arange(0           , NB_TRAIN            )
test_idxs  = np.arange(NB_TRAIN_MAX, NB_TRAIN_MAX+NB_TEST)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  1.1. Define Featurizations  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  1.1.0. statistics helpers  ------------------------------------

raw_moments = lambda w,a: tuple(np.sum(w*np.power(a,i)) for i in range(3))
std_moments = lambda w,a: (
        lambda r0,r1,r2: (r1/r0, r2/r0-(r1/r0)**2))(*raw_moments(w,a))
std_range = lambda a: (np.amax(a)-np.amin(a)) if len(a) else 0

#--------------  1.1.1. geometry helpers  --------------------------------------

dist2_vec   = lambda v, w : sum((ww-vv)**2 for vv,ww in zip(v,w))
closest_vec = lambda v, ws: min((dist2_vec(v,w), w) for w in ws)[1]

#--------------  1.1.2. features that compare to other training points  --------

nearest_neighbor = lambda v, train_xs, train_ys: min((dist2_vec(v,x), y) for y,x in zip(ys,xs) if dist2_vec(v,x)!=0)[1]

#--------------  1.1.3. mean darkness features  --------------------------------

darkness   = lambda x: np.mean(np.mean(x))
brightness = lambda x: 1. - darkness(x)

bottom_ink = lambda x: np.mean(np.mean(x[DIG_SIDE//2:,:]))
top_ink    = lambda x: np.mean(np.mean(x[:DIG_SIDE//2,:]))
left_ink   = lambda x: np.mean(np.mean(x[:,:DIG_SIDE//2]))
right_ink  = lambda x: np.mean(np.mean(x[:,DIG_SIDE//2:]))

center_ink = lambda x: x[DIG_SIDE//2, DIG_SIDE//2]

#--------------  1.1.4. spatial spread features  -------------------------------

height = lambda x: np.std([row for (row,col) in DIG_2COORS
                               if x[row,col]>DARKNESS_THRESH]) / (DIG_SIDE/2.0)
width  = lambda x: np.std([col for (row,col) in DIG_2COORS
                           if x[row,col]>DARKNESS_THRESH]) / (DIG_SIDE/2.0)

#--------------  1.1.5. collect into dictionary  -------------------------------

FEATS_BY_NAME = {
    'darkness'  : darkness   ,
    'brightness': brightness ,
    #
    'bottom_ink': bottom_ink ,
    'top_ink'   : top_ink    ,
    'left_ink'  : left_ink   ,
    'right_ink' : right_ink  ,
    #
    'center_ink': center_ink ,
    #
    'height'    : height     ,
    'width'     : width      ,
    #
    'nearest_neighbor': nearest_neighbor,
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Polynomial Features  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def tensor_power(vec, d):
    z = np.array([1.0])
    for _ in range(d):
        z = np.outer(vec, z).ravel()
    return z

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Actually do the Featurization  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

all_z_raw = np.array([[FEATS_BY_NAME[f](x) for f in FEAT_NMS] for x in all_x])

all_z = [np.concatenate(tuple(tensor_power(z, d) for d in DEGS))
         for z in all_z_raw]
all_z = np.array(all_z)
print('all_z has shape {}'.format(all_z.shape))

# (z0, z1)  -->  (1, z0, z1)
# (z0, z1)  -->  (1, z0, z1, z0**2, z0*z1, z1**2)
# QUESTION: what is the shape of  0 = A*1+ B*z0+C*z1+D*z0**2+E*z0*z1+F*z1**2
#             ANSWER: parabolas, hyperbolas, ellipses

#===============================================================================
#==  2. FIT MODELS  ============================================================
#===============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  2.0. Define Linear Classifiers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dec_func_maker = lambda w: lambda z: np.dot(w,z)

make_classifier = lambda dec_func: lambda z: DIG_A if dec_func(z)<=0 else DIG_B

is_correct = lambda classer, idx: 1 if all_y[idx]==classer(all_z[idx]) else 0

error_rate = lambda classer, idxs: np.mean([1.0-is_correct(classer, idx)
                                            for idx in idxs])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  2.1. Train and Test Classifiers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  2.1.0. loss functions and gradients  --------------------------

hinge = lambda z: max(0, 1+z)
dhinge =lambda z:   0 if   z<-1 else 1

#d(hinge(-ywx))/dw
#=
#(dhinge(z)/dz)(z=-ywx) * d(-ywx)/dw
#=
#(dhinge(z)/dz)(z=-ywx) * (-yx)

def gradient_of_hinge_loss_wrt_weight(weight, idxs=train_idxs):
    ''' mean over i of (0 if -yi*w*xi<-1 else 1) * (-yi*xi) '''
    return np.mean([
        dhinge(-all_y_signs[i]*np.dot(weight,all_z[i])) * (-all_y_signs[i]*all_z[i])
                    for i in idxs], axis=0) + L2_REG*weight

def hinge_loss(weight, idxs=train_idxs):
    #                                                     2 dimenl version of x
    # weight is the theta                                     v
    return np.mean([hinge(-all_y_signs[i]*np.dot(weight ,  all_z[i]))
                    for i in idxs])

def zero_one_error(weight, idxs=train_idxs):
    ''' mean over i of (0 if -yi*w*xi<0 else 1) '''
    return np.mean([(0 if (-all_y_signs[i]*np.dot(weight,all_z[i])<0) else 1)
                    for i in idxs])

#--------------  2.1.2. gradient descent  --------------------------------------

def report_status_during_descent(t, weight):
    print(('t={:6d}: '+
           'trn lss {:4.2f} , '+
           'tst lss {:4.2f} , '+
           'trn err {:2.0f}%, '+
           'tst err {:2.0f}%, '+
           'weight [{}]').format(
                t,
                hinge_loss(    weight, train_idxs)      ,
                hinge_loss(    weight, test_idxs )      ,
                zero_one_error(weight, train_idxs)*100.0,
                zero_one_error(weight, test_idxs )*100.0,
                ', '.join('{:+6.2f}'.format(w) for w in weight),
                #''.join(('+' if w>0 else '-') for w in weight),
            ))

def gradient_descend(learning_rate, nb_steps, report_every=None):
    weight = np.zeros(FEAT_DIM, dtype=np.float32)
    for t in range(nb_steps):
        g = gradient_of_hinge_loss_wrt_weight(weight, train_idxs)
        weight -= learning_rate * g
        #
        if report_every is None or t%report_every: continue
        report_status_during_descent(t, weight)

#gradient_descend(0.5, 2000, 100)

#--------------  2.1.1. brute force search  ------------------------------------

training_errors = {'linear':{}, 'affine':{}, 'quadratic':{}}
testing_errors  = {'linear':{}, 'affine':{}, 'quadratic':{}}

if BRUTE_FORCE_SEARCH:
    for param in tqdm.tqdm(BRUTE_PARAMS):
        #classifier = make_classifier(dec_func_maker_by_model[MODEL](*param))
        classifier = make_classifier(dec_func_maker(param))
        training_errors[MODEL][param] = error_rate(classifier, train_idxs) + 0.0001*abs(param[1]**2+param[2]**2 - 7.0**2)
        testing_errors [MODEL][param] = error_rate(classifier, test_idxs ) + 0.0001*abs(param[1]**2+param[2]**2 - 7.0**2)

    best_train = min((err,par) for par,err in training_errors[MODEL].items())
    best_test  = min((err,par) for par,err in testing_errors [MODEL].items())
    print('{} reduces train error to {:.2f}; has test error {:.2f}'.format(
        best_train[1],
        best_train[0],
        testing_errors[MODEL][best_train[1]],
        ))

    print('{} reduces test error to {:.2f}; has train error {:.2f}'.format(
        best_test[1],
        best_test[0],
        training_errors[MODEL][best_test[1]],
        ))

#===============================================================================
#==  3. PLOT  ==================================================================
#===============================================================================

#--------------  3.0.0. render some training digits  ---------------------------

render_digit = lambda x : SMOKE * (1.0-np.repeat(x[:,:,np.newaxis], 3, axis=2))

for i,idx in enumerate(train_idxs[:NB_DIGITS_RENDERED]):
    plt.imsave('mnist-trn-{:02d}.png'.format(i), render_digit(all_x[idx]))
    if not PRINT_TRAIN_FEATURES: continue
    print('train example {:02d} has darkness {:.2f} and height {:.2f}'.format(
        i, darkness(all_x[idx]), width(all_x[idx])
        ))

#--------------  3.0.1. define scatter plot initializer  -----------------------

def new_plot(data_h=PLT_SIDE, data_w=PLT_SIDE, margin=MARG,
             nb_vert_axis_ticks=10, nb_hori_axis_ticks=10):
    # white canvas
    scatter = np.ones((data_h+2*margin,
                       data_w +2*margin,3), dtype=np.float32)

    # grid lines
    for a in range(nb_vert_axis_ticks):
        s = int(data_h * float(a)/nb_vert_axis_ticks)
        scatter[margin+(data_h-1-s),margin:margin+data_w] = SMOKE
    for a in range(nb_hori_axis_ticks):
        s = int(data_w * float(a)/nb_hori_axis_ticks)
        scatter[margin:margin+data_h,margin+s]            = SMOKE

    # tick marks
    for a in range(nb_vert_axis_ticks):
        s = int(data_h * float(a)/nb_vert_axis_ticks)
        for i in range(nb_vert_axis_ticks)[::-1]:
            color = SLATE + 0.04*i*WHITE
            scatter[margin+(data_h-1-s)               ,  :margin+2+i] = color
    for a in range(nb_hori_axis_ticks):
        s = int(data_w * float(a)/nb_hori_axis_ticks)
        for i in range(nb_hori_axis_ticks)[::-1]:
            color = SLATE + 0.04*i*WHITE
            scatter[margin+data_h-2-i:2*margin+data_h , margin+s    ] = color

    # axes
    scatter[margin+data_h-1      , margin:margin+data_w] = SLATE
    scatter[margin:margin+data_h , margin              ] = SLATE

    return scatter

#--------------  3.0.2. define feature space scatter plot  --------------------

def plot_features(idxs=train_idxs,file_name='new-train.png', opacity_factor=1.0,
                  min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
                  interesting_params=[], dec_func_maker=None,
                  data_h=PLT_SIDE, data_w=PLT_SIDE, margin=MARG,
                  nb_vert_axis_ticks=10, nb_hori_axis_ticks=10):

    # initialize
    scatter = new_plot(data_h, data_w, margin,
                       nb_vert_axis_ticks, nb_hori_axis_ticks)

    # color in each hypothesis
    for p0,p1,p2 in interesting_params:
        for r in range(data_h):
            for c in range(data_w):
                z0 = 1
                z1 = min_vert + (max_vert-min_vert) * (1.0-float(r)/(data_h-1))
                z2 = min_hori + (max_hori-min_hori) * (    float(c)/(data_w-1))
                dec = dec_func_maker((p0,p1,p2))((z0,z1,z2))

                #opa = 0.2 * np.exp(-5.0*dec*dec)
                opa = 0.3 * np.exp(-5.0*dec*dec)
                color = BLUE if dec<=0 else ORANGE

                # hued flanks:
                overlay_color(scatter[margin+r,margin+c], color, opa)

                #intensity = float(p0**2+p1**2)/(data_h**2 + data_w**2)
                # thin black line:
                #if dec*dec < (p0**2+p1**2) * (4.00*intensity):
                #    opa = 0.5 * np.exp(-dec*dec/(0.30*intensity))
                #    overlay_color(scatter[margin+r,margin+c], BLACK, opa)

    # color in data scatter
    for idx in idxs:
        #z1 = min_vert + (max_vert-min_vert) * (1.0-float(r)/(data_h-1))
        #z2 = min_hori + (max_hori-min_hori) * (    float(c)/(data_w-1))
        r = margin+data_h-1-int(data_h * min(1.0, (all_z[idx][1]-min_vert)/(max_vert-min_vert)))
        c = margin+         int(data_w * min(1.0, (all_z[idx][2]-min_hori)/(max_hori-min_hori)))
        #r = margin+data_h-1-int(data_h * min(1.0, 1.0*all_z[idx][1]))
        #c = margin+         int(data_w * min(1.0, 1.0*all_z[idx][2]))
        color = BLUE if all_y[idx]==DIG_A else ORANGE
        for dr in range(-margin,margin+1):
            for dc in range(-margin,margin+1):
                opa = min(1., opacity_factor * (2.0/float(2.0 + dr*dr+dc*dc))**2)
                overlay_color(scatter[r+dr,c+dc], color, opa)

    # save
    plt.imsave(file_name, scatter)

#--------------  3.0.3. define loss landscape heatmap plot  --------------------

''' INTENDED JUST FOR MODELS WITH 2 PARAMETERS!! '''
def plot_weights(data_h=PLT_SIDE, data_w=PLT_SIDE, margin=MARG,
                 nb_vert_axis_ticks=10, nb_hori_axis_ticks=10,
                 error_by_param=None,
                 min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
                 interesting_params=[],
                 cross_size=2, box_size=3,
                 file_name='new-train-scat.png'):

    # initialize
    scatter = new_plot(data_h, data_w, margin,
                       nb_vert_axis_ticks, nb_hori_axis_ticks)

    # color in data
    for r in range(data_h):
        for c in range(data_w):
            p0 = min_vert + (max_vert-min_vert) * (1.0-float(r)/(data_h-1))
            p1 = min_hori + (max_hori-min_hori) * (    float(c)/(data_w-1))
            p0 = int(p0/P_STEP)*P_STEP
            p1 = int(p1/P_STEP)*P_STEP
            err = error_by_param[(0,p0,p1)]
            overlay_color(scatter[margin+r,margin+c], SLATE, err)

    # boxes at interesting parameters
    BS = box_size
    color = ORANGE
    for p0,p1 in interesting_params:
        rr = 1.0-(float(p0)-min_vert)/(max_vert-min_vert)
        cc =     (float(p1)-min_hori)/(max_hori-min_hori)
        r = margin+int((data_h-1)*rr)
        c = margin+int((data_w-1)*cc)
        overlay_color(scatter[r-BS        , c-BS:c+BS+1], color, 0.9)
        overlay_color(scatter[     r+BS   , c-BS:c+BS+1], color, 0.9)
        overlay_color(scatter[r-BS:r+BS+1 , c-BS       ], color, 0.9)
        overlay_color(scatter[r-BS:r+BS+1 ,      c+BS  ], color, 0.9)
        #
        overlay_color(scatter[r-BS:r+BS+1 , c-BS:c+BS+1], color, 0.5)
        #
        color = BLUE

    # cross at origin
    CS = cross_size
    cent_r = margin+int(data_h/2)
    cent_c = margin+int(data_w/2)
    scatter[cent_r-CS:cent_r+CS+1 , cent_c               ] = BLACK
    scatter[cent_r                , cent_c-CS:cent_c+CS+1] = BLACK

    # save
    plt.imsave(file_name, scatter)

#--------------  3.0.3. make all plots  ----------------------------------------

QQQ = (0, -2.0 , 8.25)           #(0, -1.25, +7.00)
WWW = (0, -1.75, 7.5 )           #(0, -1.50, +7.25),
QQ = tuple(list(QQQ)[1:])
WW = tuple(list(WWW)[1:])

plot_features(file_name='train-plain.png',
        idxs=train_idxs, opacity_factor=2.0,
        #min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
        min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
        #interesting_params=[( +2.32, +16.30, -10.47), (0.0, 20.0,-5.0), (0.0, -30.0,30.0), (0.0, 30.0, -20.0)],
        interesting_params=[
            #( +0.00,-3.00, +10.00),
            #(0, -1.25, + 7.00),
            (0, -2, +8),
            #(0, -1.50, +7.25 ),
            #( +2.70,-6.83, +13.82)
            ],
        #interesting_params=[( +4.05,  -7.59, +13.86)],#[(0.0, 20.0,-5.0), (0.0, -30.0,30.0), (0.0, 30.0, -20.0)],
        dec_func_maker=dec_func_maker,
             )

plot_features(file_name='train-features.png',
        idxs=train_idxs, opacity_factor=2.0,
        #min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
        min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
        #interesting_params=[( +2.32, +16.30, -10.47), (0.0, 20.0,-5.0), (0.0, -30.0,30.0), (0.0, 30.0, -20.0)],
        interesting_params=[
            QQQ,#WWW
            (0,-1*2,+3*2),
            (0,+4*2,-1*2)
            #( +0.00,-3.00, +10.00),
            #(0, -1.25, + 7.00),
            #(0, -1.50, +7.25 ),
            #( +2.70,-6.83, +13.82)
            ],
        #interesting_params=[( +4.05,  -7.59, +13.86)],#[(0.0, 20.0,-5.0), (0.0, -30.0,30.0), (0.0, 30.0, -20.0)],
        dec_func_maker=dec_func_maker,
             )

plot_features(file_name='test-features.png',
        idxs=test_idxs, opacity_factor=0.50,
        #min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
        min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
        #interesting_params=[( +2.32, +16.30, -10.47), (0.0, 20.0,-5.0), (0.0, -30.0,30.0), (0.0, 30.0, -20.0)],
        interesting_params=[
            QQQ#,WWW
            #( +0.00,-3.00, +10.00),
            #(0, -1.75, + 9.25),
            #(0, -1.50, +7.25 ),
            #( +2.70,-6.83, +13.82)
            ],
        #interesting_params=[( +4.05,  -7.59, +13.86)],#[(0.0, 20.0,-5.0), (0.0, -30.0,30.0), (0.0, 30.0, -20.0)],
        dec_func_maker=dec_func_maker,
             )


#plot_features(file_name='test-features.png',
#        idxs=test_idxs, opacity_factor=0.25,
#        min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
#        interesting_params=[(11.0,-3.0), (-10.0,10.0)],
#        dec_func_maker=dec_func_maker,
#             )
#
if BRUTE_FORCE_SEARCH:
    plot_weights(file_name='test-weights.png',
            error_by_param=testing_errors[MODEL],
            min_vert=-P_RANGE, max_vert=P_RANGE,  min_hori=-P_RANGE, max_hori=P_RANGE,
            #interesting_params=[(20.0,-5.0), (-20.0,20.0)],
            #interesting_params=[(-30.0, +97.5)],
            interesting_params=[
                WW,QQ
                #(-1.25, +7.00),
                #(-1.50, +7.25)
                ],
                 )
#
#plot_features(file_name='train-features.png',
#        idxs=train_idxs, opacity_factor=1.0,
#        min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
#        interesting_params=[(20.0,-5.0), (-30.0,30.0), (30.0, -20.0)],
#        dec_func_maker=dec_func_maker,
#             )
#
if BRUTE_FORCE_SEARCH:
    plot_weights(file_name='train-weights.png',
            error_by_param=training_errors[MODEL],
            min_vert=-P_RANGE, max_vert=P_RANGE, min_hori=-P_RANGE, max_hori=P_RANGE,
            #interesting_params=[(-30.0, +97.5)],
            interesting_params=[
                WW,QQ,
                (-2.5,+7.5),
                (+4,-1)
                #(-1.25, +7.00),
                #(-1.50, +7.25)
                ],
            #interesting_params=[(20.0,-5.0), (-30.0,30.0), (30.0, -20.0)],
                 )

