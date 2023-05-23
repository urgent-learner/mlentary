''' author: sam tenka
    change: 2022-06-01
    create: 2022-05-16
    descrp: Generate plots for the digit-classification example in the prologue
            pages of our 6.86x notes.  
    depend: keras 
    jargon: we'll consistently use these abbreviations when naming variables:
                dec_func    --- decision function
                idx(s)      --- index/indices within list of all examples 
                model       --- name of model ('linear', 'affine', etc)
                nb_         --- number of (whatever follows the underscore)
                side        --- sidelength of image, measured in pixels
                x           --- photo of handwritten digit
                y           --- digit-valued label
                z           --- feature vector
                vert        --- to do with a graph's vertical axis
                hori        --- to do with a graph's horizontal axis
    thanks: featurization idea inspired by abu-mostafa's book
    to use: Run `python3 example.py`.  On first run, expect a downloading
            progress bar to display and finish within 30 to 60 seconds; this
            is for downloading the MNIST dataset we'll use.
'''

#===============================================================================
#==  0. PREAMBLE  ==============================================================
#===============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  0.0. universal constants  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  0.0.0. import modules  ----------------------------------------

from keras.datasets import mnist, fashion_mnist 
from matplotlib import pyplot as plt                                            
import numpy as np                                                              
import tqdm

#--------------  0.2.1. colors  ------------------------------------------------

WHITE        = np.array([1.0 ,1.0 ,1.0 ])
SMOKE        = np.array([ .9 , .9 , .9 ])
SLATE        = np.array([ .5 , .5 , .5 ])
SHADE        = np.array([ .1 , .1 , .1 ])
BLACK        = np.array([ .0 , .0 , .0 ])

RED          = np.array([1.0 , .0 , .0 ])
BROWN        = np.array([0.5 ,0.5 , .0 ])
GREEN        = np.array([ .0 ,1.0 , .0 ])
CYAN         = np.array([ .0 , .5 , .5 ])
BLUE         = np.array([ .0 , .0 ,1.0 ])
MAGENTA      = np.array([ .5 , .0 , .5 ])

def overlay_color(background, foreground, foreground_opacity=1.0):
    background += foreground_opacity * (foreground - background)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  0.1. global parameters  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  0.1.0. reading and parsing  -----------------------------------

DIG_A, DIG_B = 1, 9
DIG_SIDE = 28 
MAX_PIX_VAL = 255

#--------------  0.1.1. data preparation ---------------------------------------

NB_TEST  = 400
NB_TRAIN =  25
THRESH   = 0.5

#--------------  0.1.1. learning parameters  -----------------------------------

P_RANGE = 100.0
P_STEP  = 5.0
PARAM_RANGES_BY_NAME = {
    'a': np.linspace(-P_RANGE, P_RANGE, int((P_RANGE*2)/P_STEP+1)), 
    'b': np.linspace(-P_RANGE, P_RANGE, int((P_RANGE*2)/P_STEP+1)),
    'c': np.linspace(-P_RANGE, P_RANGE, int((P_RANGE*2)/P_STEP+1)),
    'd': np.linspace(-P_RANGE, P_RANGE, int((P_RANGE*2)/P_STEP+1)),
    'e': np.linspace(-P_RANGE, P_RANGE, int((P_RANGE*2)/P_STEP+1)),
    'f': np.linspace(-P_RANGE, P_RANGE, int((P_RANGE*2)/P_STEP+1)),
}
PARAM_NAMES_BY_MODEL = {'linear':'bc', 'affine':'abc', 'quadratic':'abcdef'}
MODEL = 'linear'

#--------------  0.1.2. plotting and writing parameters  -----------------------

NB_DIGITS_RENDERED = 25
PLT_SIDE = 320
MARG     = 4

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  0.2. global initialization and computations  ~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  0.2.0. parameterize randomness for replicability  -------------

np.random.seed(0)

#--------------  0.2.1. format of example data  --------------------------------

DIG_1COORS = np.arange(DIG_SIDE)
DIG_2COORS = [(row,col) for row in DIG_1COORS for col in DIG_1COORS]

#--------------  0.2.2. learning helpers  --------------------------------------

PARAMS = [()]  
for param_name in PARAM_NAMES_BY_MODEL[MODEL]:
    PARAMS = [p+(val,) for val in PARAM_RANGES_BY_NAME[param_name]
                       for p in PARAMS]

#===============================================================================
#==  1. LOAD AND PREPARE DATA  =================================================
#===============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  1.0. Load Dataset Labels and Pixel Values  ~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  1.0.0. read images and labels from disk  ----------------------

# this line loads the (x=digitphotos, y=whichdigit) data set into computer memory
#(all_x, all_y), _ = mnist.load_data()
all_x = mnist.load_data()[0][0]
all_y = mnist.load_data()[0][1]

## ^ (N,28,28)
## all_x[1000,14,14] == some floating point number from 0 to 255

#--------------  1.0.1. filter to only digits of interest  ---------------------

# DIG_A == 1
# DIG_B == 9
idxs = np.logical_or(np.equal(all_y, DIG_A), np.equal(all_y, DIG_B))
all_x = all_x[idxs]
all_y = all_y[idxs]

# now all_x contains only pictures of 1s and 9s
# and all_y contains only             1s and 9s
# and they continue to correspond, e.g. all_ys[7] describes all_xs[7]

print('all_y has shape {}'.format(all_y.shape))
print('all_x has shape {}'.format(all_x.shape))

#--------------  1.0.2. normalize  ---------------------------------------------

all_x = (1.0/MAX_PIX_VAL) * all_x

#--------------  1.0.3. shuffle and split  -------------------------------------

idxs = np.arange(len(all_y))
np.random.shuffle(idxs)
all_x = all_x[idxs] # [x, x, x]
all_y = all_y[idxs]

train_idxs = np.arange(0       , NB_TRAIN        )
test_idxs  = np.arange(NB_TRAIN+175, NB_TRAIN+175+NB_TEST)

#darkness  = lambda x: np.mean(np.mean(x)) # darkness 
#height = lambda x: ...
#
#list_of_darknesses = [darkness(x) for x in all_x[train_idxs]]  # [0.3, 0.1, 0.02, ...]
#sum_train_darkness = 0.0
#element_count      = 0.0 
#for darkness_value in list_of_darknesses:
#    sum_train_darkness += darkness_value 
#    element_count += 1
#avg_train_darkness = sum_train_darkness/element_count 


#CANONICAL_NINE = all_x[0]
#CANONICAL_ONE  = all_x[1]
#
#PHOTO_TO_CLASSIFY = all_x[2]
#nines_overlap = 0
#ones_overlap  = 0
#for row in range(28):
#    for col in range(28): 
#        if abs(CANONICAL_NINE[row][col] - PHOTO_TO_CLASSIFY[row][col]) < 0.5:
#            nines_overlap = nines_overlap +1
#        if abs(CANONICAL_ONE [row][col] - PHOTO_TO_CLASSIFY[row][col]) < 0.5:
#            ones_overlap = ones_overlap +1
#print(nines_overlap, ones_overlap)
#if nines_overlap > ones_overlap:
#    print('i guess it is a nine')
#else:
#    print('i guess it is a one')
#print('truth is', all_y[2])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  1.1. Define Featurizations  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  1.1.0. statistical helpers  -----------------------------------

raw_moments = lambda w,a: tuple(np.sum(w*np.power(a,i)) for i in range(3)) 
std_moments = lambda w,a: (lambda r0,r1,r2: (r1/r0, r2/r0-(r1/r0)**2))(*raw_moments(w,a)) 
std_range = lambda a: (np.amax(a)-np.amin(a)) if len(a) else 0

#--------------  1.1.1. mean-darkness features  --------------------------------

darkness  = lambda x: np.mean(np.mean(x)) # darkness 

central_darkness  = lambda x: x[DIG_SIDE//2, DIG_SIDE//2] 
#central_darkness  = lambda x: x[DIG_SIDE//2-5:DIG_SIDE//2+6, DIG_SIDE//2-5:DIG_SIDE//2+6] 

bottomink = lambda x: np.mean(np.mean(x[DIG_SIDE//2:,:])) 
topink    = lambda x: np.mean(np.mean(x[:DIG_SIDE//2,:])) 
leftink   = lambda x: np.mean(np.mean(x[:,:DIG_SIDE//2])) 
rightink  = lambda x: np.mean(np.mean(x[:,DIG_SIDE//2:])) 

#--------------  1.1.2. spatial-spread features  -------------------------------

height = lambda x: np.std([row for (row,col) in DIG_2COORS if x[row,col]>THRESH]) / (DIG_SIDE/2.0)
width  = lambda x: np.std([col for (row,col) in DIG_2COORS if x[row,col]>THRESH]) / (DIG_SIDE/2.0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  1.2. Featurize Input Data  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

all_z = np.array([(darkness(x),height(x)) for x in all_x])

#===============================================================================
#==  2. FIT MODELS  ============================================================
#===============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  2.0. Define Linear, Affine, and Quadratic Classifiers  ~~~~~~~~~~~~~~

dec_func_maker_by_model = {
    'linear':    lambda   b,c      : lambda z0, z1:       b*z0 + c*z1, 
    'affine':    lambda a,b,c      : lambda z0, z1: a*1 + b*z0 + c*z1, 
    'quadratic': lambda a,b,c,d,e,f: lambda z0, z1: a*1 + b*z0 + c*z1 + d*z0*z0 + e*z0*z1 + f*z1*z1,
}

make_classifier = lambda dec_func: lambda z: DIG_A if dec_func(*z) <= 0 else DIG_B 
is_correct = lambda classifier, idx: 1.0 if all_y[idx]==classifier(all_z[idx]) else 0.0 
error_rate = lambda classifier, idxs: np.mean([1.0-is_correct(classifier, idx) for idx in idxs])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  2.1. Train and Test Classifiers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

training_errors = {'linear':{}, 'affine':{}, 'quadratic':{}}
testing_errors  = {'linear':{}, 'affine':{}, 'quadratic':{}}

# zeroone(-ywx) = 1 if -ywx<0 else 0
# hinge(-ywx) = max(0, 1-ywx)
# D hinge(z) = 0 if z<-1 else 1

# training loss is a sum (x_i, y_i) of hinge(-y_i*w_i*z_i)
# D training loss is a 

all_y_signs = np.array([+1 if y==9 else -1 for y in all_y]) 
def gradient_of_training_loss_wrt_weight(weight): 
    ''' returns sum (x_i, y_i) of (0 if -y_i*w_i*x_i<-1 else 1) * (-y_i*x_i) '''
    return np.mean([
        (0 if -all_y_signs[idx]*np.dot(weight,all_z[idx])<-1 else 1) *
             (-all_y_signs[idx]*all_z[idx])
        for idx in train_idxs], axis=0)

learn_rate = 0.1

weight = np.array([0.0, 0.0])
for t in range(1000):
    weight -= learn_rate * gradient_of_training_loss_wrt_weight(weight) 
    if t%100: continue 
    print(weight)
print(weight)

input('waiting for you')

for param in tqdm.tqdm(PARAMS):
    classifier = make_classifier(dec_func_maker_by_model[MODEL](*param))
    training_errors[MODEL][param] = error_rate(classifier, train_idxs) 
    testing_errors [MODEL][param] = error_rate(classifier, test_idxs ) 

best_train = min((err,par) for par,err in training_errors[MODEL].items())
best_test  = min((err,par) for par,err in testing_errors [MODEL].items())
print('param {} is best on training with training error {:.2f} and testing  error {:.2f}'.format(
    best_train[1], best_train[0], testing_errors[MODEL][best_train[1]]))
#print('param {} is best on testing with  testing  error {:.2f}'.format(best_test [1], best_test [0]))

#===============================================================================
#==  3. PLOT  ==================================================================
#===============================================================================

#--------------  3.0.0. render some training digits  ---------------------------

render_digit = lambda x : SMOKE * (1.0-np.repeat(x[:,:,np.newaxis], 3, axis=2)) 

for i,idx in enumerate(train_idxs[:NB_DIGITS_RENDERED]):
    #print('training example {:02d} has darkness {:.2f} and width {:.2f}'.format(
    #    i, darkness(all_x[idx]), width(all_x[idx])
    #    ))
    plt.imsave('mnist-trn-{:02d}.png'.format(i), render_digit(all_x[idx])) 

#--------------  3.0.1. define scatter plot initializer  -----------------------

def new_plot(data_height=PLT_SIDE, data_width=PLT_SIDE, margin=MARG,
             nb_vert_axis_ticks=10, nb_hori_axis_ticks=10): 
    # white canvas 
    scatter = np.ones((data_height+2*margin,
                       data_width +2*margin,3), dtype=np.float32) 

    # grid lines
    for a in range(nb_vert_axis_ticks): 
        s = int(data_height * float(a)/nb_vert_axis_ticks)
        scatter[margin+(data_height-1-s),margin:margin+data_width] = SMOKE
    for a in range(nb_hori_axis_ticks): 
        s = int(data_width * float(a)/nb_hori_axis_ticks)
        scatter[margin:margin+data_height,margin+s]                = SMOKE
    
    # tick marks
    for a in range(nb_vert_axis_ticks): 
        s = int(data_height * float(a)/nb_vert_axis_ticks)
        for i in range(nb_vert_axis_ticks)[::-1]:
            scatter[margin+(data_height-1-s),     0:margin+2+i] = SLATE + 0.04*i*WHITE
    for a in range(nb_hori_axis_ticks): 
        s = int(data_width * float(a)/nb_hori_axis_ticks)
        for i in range(nb_hori_axis_ticks)[::-1]:
            scatter[margin+data_height-2-i:2*margin+data_height,margin+s]                = SLATE + 0.04*i*WHITE
   
    # axes
    scatter[margin+data_height-1        , margin:margin+data_width] = SLATE
    scatter[margin:margin+data_height   , margin                  ] = SLATE

    return scatter

#--------------  3.0.2. define features plotting  ------------------------------

# INTENDED JUST FOR MODELS WITH 2 PARAMETERS!! 
def plot_features(idxs=train_idxs,file_name='new-train.png', opacity_factor=1.0,
                  min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
                  interesting_params=[], dec_func_maker=None,
                  data_height=PLT_SIDE, data_width=PLT_SIDE, margin=MARG,
                  nb_vert_axis_ticks=10, nb_hori_axis_ticks=10):

    # initialize
    scatter = new_plot(data_height, data_width, margin, nb_vert_axis_ticks, nb_hori_axis_ticks)

    # color in each hypothesis
    for p0,p1 in interesting_params:
        for r in range(data_height):
            for c in range(data_width):
                z0 = min_vert + (max_vert-min_vert) * (1.0 - float(r)/(data_height-1))
                z1 = min_hori + (max_hori-min_hori) * (      float(c)/(data_width -1))
                dec = dec_func_maker(p0,p1)(z0,z1)
                opa = 0.2 * np.exp(-5.0*dec*dec)  
                color = CYAN if dec<=0 else RED

                # hued flanks:
                overlay_color(scatter[margin+r,margin+c], color, opa)

                # thin black line: 
                if dec*dec < (p0**2+p1**2) * (1.00*float(p0**2+p1**2)/(data_height**2 + data_width**2)):
                    opa = 0.5 * np.exp(-dec*dec/(0.30*float(p0**2+p1**2)/(data_height**2 + data_width**2)))
                    overlay_color(scatter[margin+r,margin+c], BLACK, opa)

    # color in data scatter
    for idx in idxs:
        r = margin+data_height-1-int(data_height * min(1.0, 1.0 * all_z[idx][0])) # 4.0 * 
        c = margin+              int(data_width  * min(1.0, 1.0 * all_z[idx][1])) # 2.0 * 
        color = CYAN if all_y[idx]==DIG_A else RED
        for dr in range(-margin,margin+1):
            for dc in range(-margin,margin+1):
                opa = opacity_factor * (2.0/float(2.0 + dr*dr+dc*dc))**2
                overlay_color(scatter[r+dr,c+dc], color, opa)
    
    # save
    plt.imsave(file_name, scatter) 

#--------------  3.0.3. define parameters plotting  ----------------------------

dist2_vec   = lambda v, w : sum((ww-vv)**2 for vv,ww in zip(v,w))
closest_vec = lambda v, ws: min((dist2_vec(v,w), w) for w in ws)[1] 


# INTENDED JUST FOR MODELS WITH 2 PARAMETERS!! 
def plot_weights(data_height=PLT_SIDE, data_width=PLT_SIDE, margin=MARG,
                 nb_vert_axis_ticks=10, nb_hori_axis_ticks=10,
                 error_by_param=None,
                 min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
                 interesting_params=[],
                 cross_size=2, box_size=2,
                 file_name='new-train-scat.png'):

    # initialize
    scatter = new_plot(data_height, data_width, margin, nb_vert_axis_ticks, nb_hori_axis_ticks)

    # color in data
    for r in range(data_height): 
        for c in range(data_width): 
            p0 = min_vert + (max_vert-min_vert) * (1.0 - float(r)/(data_height-1))
            p1 = min_hori + (max_hori-min_hori) * (      float(c)/(data_width -1))
            #z0, z1 = closest_vec((z0,z1), PARAMS) 
            p0 = int(p0/P_STEP)*P_STEP
            p1 = int(p1/P_STEP)*P_STEP
            err = error_by_param[(p0,p1)] 
            overlay_color(scatter[margin+r,margin+c], SLATE, err)  

    # boxes at interesting parameters 
    BS = box_size
    for p0,p1 in interesting_params:
        r = margin + int((data_height-1) * (1.0 - (float(p0)-min_vert)/(max_vert-min_vert)))
        c = margin + int((data_width -1) * (      (float(p1)-min_hori)/(max_hori-min_hori)))
        overlay_color(scatter[r-BS        , c-BS:c+BS+1], BLACK, 0.5)
        overlay_color(scatter[     r+BS   , c-BS:c+BS+1], BLACK, 0.5)
        overlay_color(scatter[r-BS:r+BS+1 , c-BS       ], BLACK, 0.5)
        overlay_color(scatter[r-BS:r+BS+1 ,      c+BS  ], BLACK, 0.5)

    # cross at origin
    CS = cross_size
    cent_r = margin+int(data_height/2)
    cent_c = margin+int(data_width /2)
    scatter[cent_r-CS:cent_r+CS+1 , cent_c]                = BLACK 
    scatter[cent_r                , cent_c-CS:cent_c+CS+1] = BLACK 
       
    # save
    plt.imsave(file_name, scatter) 

#--------------  3.0.3. make all plots  ----------------------------------------

#plot_features(idxs=test_idxs, file_name='new-test.png', opacity_factor=1.0,
#              min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
#              interesting_params=[(11.0,-3.0), (-10.0,10.0)], dec_func_maker=dec_func_maker_by_model[MODEL],
#             )
#
#plot_weights(error_by_param=testing_errors[MODEL], file_name='new-test-scat.png',
#             min_vert=-100.0, max_vert=100.0,  min_hori=-100.0, max_hori=100.0,
#             interesting_params=[(20.0,-5.0), (-20.0,20.0)],
#             )

plot_features(idxs=train_idxs, file_name='new-train.png', opacity_factor=1.0,
              min_vert=0.0, max_vert=1.0,  min_hori=0.0, max_hori=1.0,
              #interesting_params=[(11.0,-3.0), (-10.0,10.0), (15.0, -10.0)],
              interesting_params=[(20.0,-5.0), (-30.0,30.0), (30.0, -20.0)],
              dec_func_maker=dec_func_maker_by_model[MODEL],
             )

plot_weights(error_by_param=training_errors[MODEL], file_name='new-train-scat.png',
             min_vert=-100.0, max_vert=100.0,  min_hori=-100.0, max_hori=100.0,
             #interesting_params=[(11.0,-3.0), (-10.0,10.0), (15.0, -10.0)],
             interesting_params=[(20.0,-5.0), (-30.0,30.0), (30.0, -20.0)],
             )


