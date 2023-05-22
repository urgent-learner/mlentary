''' author: sam tenka
    change: 2022-05-24
    create: 2022-05-16
    descrp: Generate plots for the digit-classification example in the prologue
            page of our 6.86x notes.  
    depend: keras 
    thanks: featurization idea inspired by abu-mostafa's book
    to use: Run `python3 example.py`.  On first run, expect a downloading
            progress bar to display and finish within 30 to 60 seconds; this
            is for downloading the MNIST dataset we'll use.
'''

#===============================================================================
#==  0. PREAMBLE  ==============================================================
#===============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  0.0. import modules  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from keras.datasets import mnist, fashion_mnist
from matplotlib import pyplot as plt                                            
import numpy as np                                                              

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  0.1. global free parameters  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  0.1.0. reading parameters  ------------------------------------

DIG_A, DIG_B = 1, 9
DIG_SIDE = 28 
MAX_PIX_VAL = 255

#--------------  0.1.1. processing parameters  ---------------------------------

NB_TEST  = 800
NB_TRAIN = 200
HARD_THRESH = 0.05

#--------------  0.1.2. writing parameters  ------------------------------------

PLT_SIDE = 320
MARG     = 2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  0.2. global initialization  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

np.random.seed(0)

DIG_COORS = np.arange(DIG_SIDE)

#===============================================================================
#==  1. PREPROCESSING  =========================================================
#===============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  1.0. Load Dataset Labels and Pixel Values  ~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  1.0.0. read  --------------------------------------------------

(trn_x, trn_y), _ = mnist.load_data()
# ^ (N,28,28)
# trn_x[1000,14,14] == some floating point number from 0 to 255

#--------------  1.0.1. filter  ------------------------------------------------

idxs = np.logical_or(np.equal(trn_y, DIG_A), np.equal(trn_y, DIG_B))
trn_x = trn_x[idxs]
trn_y = trn_y[idxs]
print('trn_y has shape {}'.format(trn_y.shape))

#--------------  1.0.2. normalize  ---------------------------------------------

trn_x = (1.0/MAX_PIX_VAL) * trn_x

#--------------  1.0.3. shuffle  -----------------------------------------------

idxs = np.arange(len(trn_y))
np.random.shuffle(idxs)
trn_x = trn_x[idxs]
trn_y = trn_y[idxs]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  1.1. Define Featurizations  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#--------------  1.1.0. statistical helpers  -----------------------------------

raw_moments = lambda w,a: tuple(np.sum(w*np.power(a,i)) for i in range(3)) 
std_moments = lambda w,a: (lambda r0,r1,r2: (r1/r0, r2/r0-(r1/r0)**2))(*raw_moments(w,a)) 
std_range = lambda a: (np.amax(a)-np.amin(a)) if len(a) else 0

#--------------  1.1.1. mean-darkness features  --------------------------------

darkness  = lambda x: np.mean(np.mean(x)) # darkness 
bottomink = lambda x: np.mean(np.mean(x[DIG_SIDE//2:,:])) 
topink    = lambda x: np.mean(np.mean(x[:DIG_SIDE//2,:])) 
leftink   = lambda x: np.mean(np.mean(x[:,:DIG_SIDE//2])) 
rightink  = lambda x: np.mean(np.mean(x[:,DIG_SIDE//2:])) 

#--------------  1.1.2. spatial-spread features  -------------------------------

height = lambda x: np.std([row for col in range(DIG_SIDE) for row in range(DIG_SIDE) if x[col][row]>0.5]) / (DIG_SIDE/2.0)
width  = lambda x: np.std([col for col in range(DIG_SIDE) for row in range(DIG_SIDE) if x[col][row]>0.5]) / (DIG_SIDE/2.0)

#height = lambda x: (lambda m,v: np.sqrt(v)*2.0/DIG_SIDE)(*std_moments(np.mean(x,axis=0), DIG_COORS)) 
#width  = lambda x: (lambda m,v: np.sqrt(v)*2.0/DIG_SIDE)(*std_moments(np.mean(x,axis=1), DIG_COORS)) 

#height_hard= lambda x: (lambda a: float(std_range(a))/DIG_SIDE)(DIG_COORS[np.mean(x,axis=0)>HARD_THRESH]) 
#width_hard = lambda x: (lambda a: float(std_range(a))/DIG_SIDE)(DIG_COORS[np.mean(x,axis=1)>HARD_THRESH]) 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  1.2. Featurize Input Data  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

features = np.array([(darkness(x),width(x)) for x in trn_x])

#===============================================================================
#==  2. FIT MODELS  ============================================================
#===============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  2.0. Fit Linear Classifiers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~  2.1. Fit Gaussian Generative Classifiers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~

#===============================================================================
#==  3. PLOT  ==================================================================
#===============================================================================


#--------------  3.0.0. plot initialization  -----------------------------------

render = lambda x : 0.95 * (1.0-np.repeat(x[:,:,np.newaxis], 3, axis=2)) 
#

scatter = np.ones((PLT_SIDE+2,PLT_SIDE+2,3), dtype=np.float32) 

for a in range(10):
    s = int(float(a)/10.0*PLT_SIDE)
    scatter[1+(PLT_SIDE-1-s),1:PLT_SIDE] = [0.9,0.9,0.9]
    scatter[1:PLT_SIDE,1+s]       = [0.9,0.9,0.9]
for a in range(10):
    s = int(float(a)/10.0*PLT_SIDE)
    for i in range(10)[::-1]:
        scatter[PLT_SIDE-2-i:PLT_SIDE+2,1+s] = [0.5+0.04*i,0.5+0.04*i,0.5+0.04*i]
        scatter[1+(PLT_SIDE-1-s),0:1+2+i]    = [0.5+0.04*i,0.5+0.04*i,0.5+0.04*i]

scatter[PLT_SIDE,1:PLT_SIDE] = [0.5,0.5,0.5]
scatter[1:PLT_SIDE,1] = [0.5,0.5,0.5]

#PLT_HALF = int(PLT_SIDE/2)
#scatter = np.ones((PLT_HALF+2*MARG,PLT_SIDE+2*MARG,3), dtype=np.float32) 
#
## grid lines
#for a in range(5): 
#    s = int(float(a)/5.0 *PLT_HALF)
#    scatter[MARG+(PLT_HALF-1-s),MARG:PLT_SIDE] = [0.9,0.9,0.9]
#for a in range(10): 
#    s = int(float(a)/10.0*PLT_SIDE)
#    scatter[MARG:PLT_HALF,MARG+s]       = [0.9,0.9,0.9]
#
## tick marks
#for a in range(10):
#    s = int(float(a)/10.0*PLT_SIDE)
#    for i in range(10)[::-1]:
#        scatter[MARG+PLT_HALF-1-2-i:PLT_SIDE+2,MARG+s] = [0.5+0.04*i,0.5+0.04*i,0.5+0.04*i]
#        if a<5:
#          scatter[MARG+(PLT_HALF-1-s),0:MARG+2+i]  = [0.5+0.04*i,0.5+0.04*i,0.5+0.04*i]
#
## axes
#scatter[MARG+PLT_HALF-1,MARG:PLT_SIDE] = [0.5,0.5,0.5]
#scatter[MARG:PLT_SIDE,MARG]            = [0.5,0.5,0.5]





#
make_classifier = lambda a,b : lambda f: DIG_A if a*f[0]+b*f[1] <= 0 else DIG_B 

B   = 200#0 #100#0 #100
NN  = 800#25#900#48#900 
RES = 200
best = (float('inf'), (None,None),(None,None))

r=0
while r<PLT_SIDE:
    if r%20==0: print(r)
    dr = 1 if abs(r-PLT_SIDE/2) < 30 else 2 if abs(r-PLT_SIDE/2) < 60 else 4  
    dc = 1 if abs(r-PLT_SIDE/2) < 30 else 2 if abs(r-PLT_SIDE/2) < 60 else 2  
    for c in range(0,PLT_SIDE,dc):
        a = int(RES*((PLT_SIDE-r)/float(PLT_SIDE)))-RES//2
        b = int(RES*(c/float(PLT_SIDE)))-RES//2
        err = sum((1 for y,f in zip(trn_y[B:B+NN], features[B:B+NN]) if (y!=make_classifier(a,b)(f))))
        err = float(err)/NN
        if (r,c)==(12,120):
            print(err)
        if err<=best[0]:
            best = (err,(r,c),(a,b))
        color = [0.6,0.5,0.4] # reddish gray
        scatter[r+1:r+1+dr,c+1:c+1+dc] += err * (np.array(color)-scatter[r+1:r+1+dr,c+1:c+1+dc]) 

        #if err < 0.2  :
        #    color = [0.0,1.0,0.0] # bright green
        #    scatter[r+1:r+3,c+1:c+1+dc] += 0.3 * (np.array(color)-scatter[r+1:r+3,c+1:c+1+dc]) 
    r += dr

#r,c = best[1]
#for r,c in ((100-80,101-20),(100-92,101-22),(100-80,101-40),(100+30,101+30)):
#for r,c in ((100-80,101-20),(100-92,101-22)                                ):
for r,c in ((100-80,101-20),                (100-80,101-40),(100+30,101+30)):
    scatter[r+1-2      ,c+1-2:c+1+3] += 0.5 * ( np.array([0.0,0.0,0.0]) - scatter[r+1-2      ,c+1-2:c+1+3] )
    scatter[      r+1+2,c+1-2:c+1+3] += 0.5 * ( np.array([0.0,0.0,0.0]) - scatter[      r+1+2,c+1-2:c+1+3] )
    scatter[r+1-2:r+1+3,c+1-2      ] += 0.5 * ( np.array([0.0,0.0,0.0]) - scatter[r+1-2:r+1+3,c+1-2      ] )
    scatter[r+1-2:r+1+3,      c+1+2] += 0.5 * ( np.array([0.0,0.0,0.0]) - scatter[r+1-2:r+1+3,      c+1+2] )
scatter[100-2:100+3,101] = 0
scatter[100,101-2:101+3] = 0
print(best)
plt.imsave('train-scat.png', scatter) 
#plt.imsave('test-scat.png', scatter) 

##curve = lambda y,x: float(10.0*(0.270* (1.0 - 56*(x-0.7625)**2 - 0.1*(y-1)**2) - (y-0.5*x)-0.5*0.7625)) 
##
##a,b = 15, 15    
##for r in range(PLT_SIDE):
##    for c in range(PLT_SIDE):
##        f0, f1 = (PLT_SIDE-1-r)/float(PLT_SIDE), c/float(PLT_SIDE)
##        dec = curve(f0,f1)
##        opa = 0.1 * np.exp(-0.5*dec*dec)  
##        color = [0.0,0.5,0.5] if dec<=0 else [1.0,0.0, 0.0]
##        scatter[r+1,c+1] += opa * (np.array(color)-scatter[r+1,c+1]) 
##        if dec*dec < (a*a+b*b) * (0.75*float(a*a+b*b)/(PLT_SIDE*PLT_SIDE)):
##            color = [0.0,0.0,0.0]
##            opa = 0.5 * np.exp(-dec*dec/(0.30*float(a*a+b*b)/(PLT_SIDE*PLT_SIDE)))
##            scatter[r+1,c+1] += opa * (np.array(color)-scatter[r+1,c+1]) 

#a,b = 92, -22 
#for r in range(PLT_SIDE):
#    for c in range(PLT_SIDE):
#        f0, f1 = (PLT_SIDE-1-r)/float(PLT_SIDE), c/float(PLT_SIDE)
#        dec = a*f0+b*f1 
#        opa = 0.2 * np.exp(-0.5*dec*dec)  
#        color = [0.0,0.5,0.5] if dec<=0 else [1.0,0.0, 0.0]
#        scatter[r+1,c+1] += opa * (np.array(color)-scatter[r+1,c+1]) 
#        if dec*dec < (a*a+b*b) * (0.75*float(a*a+b*b)/(PLT_SIDE*PLT_SIDE)):
#            color = [0.0,0.0,0.0]
#            opa = 0.5 * np.exp(-dec*dec/(0.30*float(a*a+b*b)/(PLT_SIDE*PLT_SIDE)))
#            scatter[r+1,c+1] += opa * (np.array(color)-scatter[r+1,c+1]) 
#
#a,b = 80,-20   
#for r in range(PLT_SIDE):
#    for c in range(PLT_SIDE):
#        f0, f1 = (PLT_SIDE-1-r)/float(PLT_SIDE), c/float(PLT_SIDE)
#        dec = float(a*f0+b*f1) 
#        opa = 0.2 * np.exp(-0.5*dec*dec)  
#        color = [0.0,0.5,0.5] if dec<=0 else [1.0,0.0, 0.0]
#        scatter[r+1,c+1] += opa * (np.array(color)-scatter[r+1,c+1]) 
#        if dec*dec < (a*a+b*b) * (0.75*float(a*a+b*b)/(PLT_SIDE*PLT_SIDE)):
#            color = [0.0,0.0,0.0]
#            opa = 0.5 * np.exp(-dec*dec/(0.30*float(a*a+b*b)/(PLT_SIDE*PLT_SIDE)))
#            scatter[r+1,c+1] += opa * (np.array(color)-scatter[r+1,c+1]) 

#a,b = 80, -40
#for r in range(PLT_SIDE):
#    for c in range(PLT_SIDE):
#        f0, f1 = (PLT_SIDE-1-r)/float(PLT_SIDE), c/float(PLT_SIDE)
#        dec = a*f0+b*f1 
#        opa = 0.2 * np.exp(-0.5*dec*dec)  
#        color = [0.0,0.5,0.5] if dec<=0 else [1.0,0.0, 0.0]
#        scatter[r+1,c+1] += opa * (np.array(color)-scatter[r+1,c+1]) 
#        if dec*dec < (a*a+b*b) * (0.75*float(a*a+b*b)/(PLT_SIDE*PLT_SIDE)):
#            color = [0.0,0.0,0.0]
#            opa = 0.5 * np.exp(-dec*dec/(0.30*float(a*a+b*b)/(PLT_SIDE*PLT_SIDE)))
#            scatter[r+1,c+1] += opa * (np.array(color)-scatter[r+1,c+1]) 
#
#a,b = -30, 30
#for r in range(PLT_SIDE):
#    for c in range(PLT_SIDE):
#        f0, f1 = (PLT_SIDE-1-r)/float(PLT_SIDE), c/float(PLT_SIDE)
#        dec = a*f0+b*f1 
#        opa = 0.2 * np.exp(-0.5*dec*dec)  
#        color = [0.0,0.5,0.5] if dec<=0 else [1.0,0.0, 0.0]
#        scatter[r+1,c+1] += opa * (np.array(color)-scatter[r+1,c+1]) 
#        if dec*dec < (a*a+b*b) * (0.75*float(a*a+b*b)/(PLT_SIDE*PLT_SIDE)):
#            color = [0.0,0.0,0.0]
#            opa = 0.5 * np.exp(-dec*dec/(0.30*float(a*a+b*b)/(PLT_SIDE*PLT_SIDE)))
#            scatter[r+1,c+1] += opa * (np.array(color)-scatter[r+1,c+1]) 


#B  = 0 #200 # 0 #200 #100  
#N  =25 #800 #25 #800 #1900 
#OF =1.0#0.50#1.0#1.00#0.50
#for i in range(B,B+N):
#    r = 1+PLT_SIDE-1-int(PLT_SIDE * min(1.0, 2.0 * darkness (trn_x[i])))
#    c = 1+           int(PLT_SIDE * min(1.0, 2.0 * width (trn_x[i])))  # TODO: get from FEATURES
#    color = [0.0,0.5,0.5] if trn_y[i]==DIG_A else [1.0,0.0, 0.0]
#    for reps in range(10 if i < 2 else 1):
#      for dr in range(-1,2):
#          for dc in range(-1,2):
#              opa = {0:0.7, 1:0.3, 2:0.1}[dr*dr+dc*dc] * OF
#              scatter[r+dr,c+dc] += opa * (np.array(color)-scatter[r+dr,c+dc]) 

#B  = 0 #200 # 0 #200 #100  
#N  =25 #800 #25 #800 #1900 
#OF =1.0#0.50#1.0#1.00#0.50
#for i in range(B,B+N):
#    r = MARG+PLT_HALF-1-int(PLT_HALF * min(1.0, 4.0 * darkness (trn_x[i])))
#    c = MARG+           int(PLT_SIDE * min(1.0, 2.0 * width (trn_x[i])))  # TODO: get from FEATURES
#    color = [0.0,0.5,0.5] if trn_y[i]==DIG_A else [1.0,0.0, 0.0]
#    for reps in range(6  if i < 2 else 1):
#      for dr in range(-MARG,MARG+1):
#          for dc in range(-MARG,MARG+1):
#              #opa = {0:0.7 ,   1:0.6 ,   2:0.5 ,   3:0.3 ,   4:0.1 ,
#              #                 5:0.07,   6:0.04,   7:0.01,   8:0.00}[dr*dr+dc*dc] * OF
#              opa = OF * (2.0/float(2.0 + dr*dr+dc*dc))**2
#              scatter[r+dr,c+dc] += opa * (np.array(color)-scatter[r+dr,c+dc]) 

#plt.imsave('train-plain.png', scatter) 
#plt.imsave('test-plain.png', scatter) 
#plt.imsave('train.png', scatter) 
#plt.imsave('test.png', scatter) 

#err = sum((1 for y,f in zip(trn_y[B:B+N], features[B:B+N])
#           if ((curve(2*f[0],2*f[1])<=0 and y!=DIG_A) or
#               (curve(2*f[0],2*f[1])> 0 and y!=DIG_B))))
#print(float(err)/N)

#B=200
#N=800
#err = sum((1 for y,f in zip(trn_y[B:B+N], features[B:B+N])
#           if ((13*f[0]-3*f[1])<=0 and y!=DIG_A or
#               (13*f[0]-3*f[1])> 0 and y!=DIG_B)))
#print(float(err)/N)

#for i in range(25):
#    print(i, darkness(trn_x[i]), width(trn_x[i]))

#    plt.imsave('mnist-trn-{:02d}.png'.format(i), render(trn_x[i])) 

#counts = [0 for _ in range(10)]
#for i in range(len(trn_y)):
#    if counts[DIG_A]==3 and counts[DIG_B]==3: break
#    if counts[trn_y[i]]==3: continue
#    plt.imsave('mnist-trn-{:02d}.png'.format(sum(counts)), render(trn_x[i])) 
#    counts[trn_y[i]] += 1
#for i in range(20):
#    plt.imsave('mnist-trn-{:02d}.png'.format(i), render(trn_x[i])) 
#print(trn_y)
