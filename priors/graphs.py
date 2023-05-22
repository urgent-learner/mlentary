''' author: sam tenka
    change: 2022-07-28
    create: 2022-07-28
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
                y_sign      --- {-1,+1}-valued label
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

from matplotlib import pyplot as plt                                            
import numpy as np                                                              
import tqdm

#--------------  0.2.1. colors  ------------------------------------------------

WHITE        = np.array([1.0 ,1.0 ,1.0 ])
SMOKE        = np.array([ .9 , .9 , .9 ])
SLATE        = np.array([ .5 , .5 , .5 ])
SHADE        = np.array([ .1 , .1 , .1 ])
BLACK        = np.array([ .0 , .0 , .0 ])

RED          = np.array([1.0 , .0 , .0 ]) #####
ORANGE       = np.array([ .75,0.25, .0 ]) #    
BROWN        = np.array([ .5 ,0.5 , .0 ]) ###    # i.e., dark YELLOW
OLIVE        = np.array([ .25,0.75, .0 ]) #    
GREEN        = np.array([ .0 ,1.0 , .0 ]) #####
AGAVE        = np.array([ .0 , .75, .25]) #    
CYAN         = np.array([ .0 , .5 , .5 ]) ###  
JUNIPER      = np.array([ .0 , .25, .75]) #    
BLUE         = np.array([ .0 , .0 ,1.0 ]) ##### 
INDIGO       = np.array([ .25, .0 , .75]) #    
MAGENTA      = np.array([ .5 , .0 , .5 ]) ###  
AMARANTH     = np.array([ .75, .0 , .25]) #    

RAINBOW = [
    RED     ,
    ORANGE  ,
    BROWN   ,
    OLIVE   ,
    GREEN   ,
    AGAVE   ,
    CYAN    ,
    JUNIPER ,
    BLUE    ,
    INDIGO  ,
    MAGENTA ,
    AMARANTH,
]


def overlay_color(background, foreground, foreground_opacity=1.0):
    background += foreground_opacity * (foreground - background)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  0.1. global parameters  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  0.1.1. learning parameters  -----------------------------------

np.random.seed(0)

def sample(dim,p):
    return {
        1: lambda d: np.random.laplace(size=d),
        2: lambda d: np.random.normal(size=d),
    'inf': lambda d: np.random.uniform(size=d, low=-1.0, high=+1.0),
    }[p](dim) 

def norm(v,p):
    return { 
        1: np.mean(np.abs(v)), 
        2: np.sqrt(np.mean(np.square(v))), 
    'inf': np.max(np.abs(v)),
    }[p]

def sample_scaled(dim,p,q,scale=1.0):
    s = sample(dim, p)
    y = sample(dim, q)
    return scale * s * norm(y,q)/norm(s,p) 

dim = 30
halfheight=200
vert = 15 
hori = 6
for p in [1, 2, 'inf']:
    for q in [1, 2, 'inf']:
        scatter = np.ones(( halfheight*2, dim*hori, 4), dtype=np.float32) 
        scatter[:,:,3]  = 0
        for i in [0,+2]:
            scatter[halfheight-int(vert*i),:, 3]  = 1
            scatter[halfheight-int(vert*i),:,:3]  = 0
        samples = np.array([sample_scaled(dim, p, q) for _ in range(10001)]) 

        stddev = np.sqrt(np.mean(np.square(np.ndarray.flatten(samples)))) 
        samples /= np.sqrt(3)*stddev

        stddev = np.std(samples, axis=1)
        samples = samples[np.argsort(stddev)]
        samples = np.array([samples[0],samples[5000],samples[10000]])
        samples = np.abs(samples)
        samples = np.array([sorted(s) for s in samples])

        for color, s in tqdm.tqdm(zip([CYAN, BLUE, MAGENTA], samples)):
            for i,(a,b) in enumerate(zip(s, s[1:])):
                x,y =  hori*(i+0.5), halfheight- (a)*2*vert  
                X,Y =  hori*(i+1.5), halfheight- (b)*2*vert  
                l = np.sqrt((X-x)**2+(Y-y)**2)
                for t in np.linspace(0.0,1.0,int(10+1.5*l)):
                    for dx in np.linspace(-1.2,+1.2,9):
                        for dy in np.linspace(-1.2,+1.2,9):
                            opacity = 1.0/(1.0 + 10*(dx*dx + dy*dy))
                            xx = int(x+(X-x)*t+dx)
                            yy = int(y+(Y-y)*t+dy)
                            dd = (2.0*max(0.0, 0.50-abs(abs(a)+(abs(b)-abs(a))*t)))**0.5 
                            if 0<=yy<halfheight*2 and 0<=xx<dim*hori:
                                cc = color
                                scatter[yy][xx][:3] = (1.0-opacity)*scatter[yy][xx][:3] + opacity*(cc + (0.1+0.9*dd)*(WHITE-cc))
                                scatter[yy][xx][ 3] = 1.0 - (1.0-opacity) * (1.0 - scatter[yy][xx][ 3])
        plt.imsave('yo-{}-{}.png'.format(p,q), scatter)

##--------------  3.0.0. render some training digits  ---------------------------
#
#def new_plot(data_h=PLT_SIDE, data_w=PLT_SIDE, margin=MARG,
#             nb_vert_axis_ticks=10, nb_hori_axis_ticks=10): 
#    # white canvas 
#    scatter = np.ones((data_h+2*margin,
#                       data_w +2*margin,3), dtype=np.float32) 
#
#    # grid lines
#    for a in range(nb_vert_axis_ticks): 
#        s = int(data_h * float(a)/nb_vert_axis_ticks)
#        scatter[margin+(data_h-1-s),margin:margin+data_w] = SMOKE
#    for a in range(nb_hori_axis_ticks): 
#        s = int(data_w * float(a)/nb_hori_axis_ticks)
#        scatter[margin:margin+data_h,margin+s]            = SMOKE
#    
#    # tick marks
#    for a in range(nb_vert_axis_ticks): 
#        s = int(data_h * float(a)/nb_vert_axis_ticks)
#        for i in range(nb_vert_axis_ticks)[::-1]:
#            color = SLATE + 0.04*i*WHITE
#            scatter[margin+(data_h-1-s)               ,  :margin+2+i] = color
#    for a in range(nb_hori_axis_ticks): 
#        s = int(data_w * float(a)/nb_hori_axis_ticks)
#        for i in range(nb_hori_axis_ticks)[::-1]:
#            color = SLATE + 0.04*i*WHITE
#            scatter[margin+data_h-2-i:2*margin+data_h , margin+s    ] = color
#   
#    # axes
#    scatter[margin+data_h-1      , margin:margin+data_w] = SLATE
#    scatter[margin:margin+data_h , margin              ] = SLATE
#
#    return scatter
#
##--------------  3.0.2. define feature space scatter plot  --------------------
#
#def plot_samples():
#    # initialize
#    scatter = new_plot(data_h, data_w, margin,
#                       nb_vert_axis_ticks, nb_hori_axis_ticks)
#
#    # save
#    plt.imsave(file_name, scatter) 
#
#
