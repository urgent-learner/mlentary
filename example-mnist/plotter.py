''' author: samtenka
    change: 2023-02-20
    create: 2022-05-16
    descrp:
    to use:
'''

#-------  imports  ------------------------------------------------------------

from matplotlib import pyplot as plt
from pyrecord import Record
import numpy as np
import tqdm

#-------  colors  -------------------------------------------------------------

BLUE    = np.array([ .05,  .55,  .85]) # hues (colorblindness-friendly pair)
ORANGE  = np.array([ .95,  .65,  .05]) #

WHITE   = np.array([1.  , 1.  , 1.  ]) # shades
SMOKE   = np.array([ .9 ,  .9 ,  .9 ])
SLATE   = np.array([ .5 ,  .5 ,  .5 ])
SHADE   = np.array([ .1 ,  .1 ,  .1 ])
BLACK   = np.array([ .0 ,  .0 ,  .0 ])

def overlay(background, foreground, foreground_opacity=1.0):
    background += foreground_opacity * (foreground - background)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Plot Class Basics  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#-------  example usage  ------------------------------------------------------

''' We'll define a `Plot` class to support uses such as in the example below.

    for #PROGRAMMERS: I prefer real code to code written in a comment, since
    code-in-a-comment can become outdated without us noticing.  There are tools
    to extract unit tests from comments, but I don't use them.
'''

def small_example_to_illustrate_comment():
    (blank(data_height=480, data_width=480, margin=6)
        .add_gridlines().add_ticks().add_axes()
        .box_at(0.6, 0.8, color=BLUE)
        .scatter_points([(.314,.271), (.159,.828)], color=ORANGE)
        .save_to('hello.png')
    )

#-------  datatype   ----------------------------------------------------------

''' A `Plot` instance keeps an image bitmap `pixels` (array of pixel
    intensities) and a coordinate system for that bitmap represented as
    functions `cell_from` and `coor_from` that convert between pixel
    (row,column) indices and abstract (y-coordinate,x-coordinate) pairs.  It
    also has a tuple `HWM` for height, width, and margin, measured in pixels.

    for #PROGRAMMERS: We write our class as "plain old data" manipulated by
    "functions we add externally", albeit sugared with "method call syntax".
    This style is not idiomatic for Python, but I prefer it because it allows
    us to arrange code as we would for C or Haskell.
'''

Plot = Record.create_type('Plot', 'pixels', 'cell_from', 'coor_from', 'HWM')

def bind_to_class(c, name):
    ''' Returns a decorator that binds the function in question as a method of
        the class `c` with methodname `name`.  For instance, if object `moo` is
        an instance of the class `Plot`, and if `c=Plot` and `name='save_to'`,
        then this decorator allows us to call functions (defined at top level
        scope) via `moo.save_to(...)`.
    '''
    def decorator(f):
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        setattr(c, name, wrapped)
        return wrapped
    return decorator

#-------  create and consume  -------------------------------------------------

def blank(data_height, data_width, margin, color=WHITE,
          ymin=0., ymax=1., xmin=0., xmax=1.           ):
    H,W,M = data_height, data_width, margin
    pixels = np.ones((H+2*M,
                      W+2*M,3)) * color
    cell_from = lambda y,x : (int(M+(H-1) * (1.-(y-ymin)/(ymax-ymin))),
                              int(M+(W-1) * (   (x-xmin)/(xmax-xmin))) )
    coor_from = lambda r,c : (ymin + (ymax-ymin) * (1.-(r-margin)/float(H-1)),
                              xmin + (xmax-xmin) * (   (c-margin)/float(W-1)) )
    return Plot(pixels, cell_from, coor_from, (H, W, M))

@bind_to_class(Plot, 'save_to')
def save_to(plot, file_name):
    plt.imsave(file_name, plot.pixels)
    return plot

#-------  more examples to illustrate all features  ---------------------------

def examples()
    pp = (blank(data_height=480, data_width=480, margin=6)
             .add_gridlines()
             .add_ticks()
             .add_axes()
             .shade_heatmap(lambda weights: weights.dot(np.array([1.,-2.])))
             .cross_at(0.4, 0.9)
             .box_at(0.6, 0.8, color=ORANGE)
             .box_at(0.9, 0.2, color=ORANGE)
             .shade_hypothesis(lambda feats: feats.dot(np.array([10.,-20.])))
             .shade_hypothesis(lambda feats: 5. + feats.dot(np.array([-10.,0.])))
             .scatter_points([(.55,.55), (.65,.85), (.45,.55)], color=ORANGE)
             .scatter_points([(.15,.55), (.35,.85), (.95,.55)], color=[BLUE,ORANGE,BLUE])
             .save_to('yo.png')
         )

    qq = (blank(480, 480, 6, ymin=-1.,ymax=+1.,xmin=-1.,xmax=+1.)
             .add_gridlines(np.arange(-.8,1.,.2), np.arange(-.8,1.,.2), opacity=1.)
             .shade_heatmap(lambda weights: (
                 np.mean(weights**2,axis=2)
                 +0.5+0.5*np.sin(weights.dot(np.array([10.,-20.])))
                 +.25* (weights[:,:,0]-weights[:,:,1]**2)**2
                 ))
             .box_at(0.6, 0.8, color=ORANGE)
             .box_at(0.9, 0.2, color=ORANGE)
             .add_gridlines(np.arange(-.8,1.,.2), np.arange(-.8,1.,.2), opacity=.2)
             .cross_at()
             .save_to('cow.png')
         )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Plot Class: Draw Various Graphic Elements  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#-------  axes and friends  ---------------------------------------------------

INNER_DECADES = np.arange(.1, 1., .1)

@bind_to_class(Plot, 'add_gridlines')
def add_gridlines(plot, yticks=INNER_DECADES, xticks=INNER_DECADES,
                  color=SMOKE, opacity=1.                          ):
    _,_,M = plot.HWM
    for y in yticks:
        r,_ = plot.cell_from(y,0)
        overlay(plot.pixels[r,M:-M], color, opacity)
    for x in xticks:
        _,c = plot.cell_from(0,x)
        overlay(plot.pixels[M:-M,c], color, opacity)
    return plot

@bind_to_class(Plot, 'add_ticks')
def add_ticks(plot, yticks=INNER_DECADES, xticks=INNER_DECADES, yy=0., xx=0.,
              color=SLATE                                                    ):
    H,W,M = plot.HWM
    rr,cc = plot.cell_from(yy,xx)
    mask = np.linspace(1., 0., 3*M+1)[:,np.newaxis]
    for y in yticks:
        r,_ = plot.cell_from(y,0)
        overlay(plot.pixels[r, cc-M:cc+2*M+1], color, mask      )
    for x in xticks:
        _,c = plot.cell_from(0,x)
        overlay(plot.pixels[rr-2*M:rr+M+1, c], color, mask[::-1])
    return plot

@bind_to_class(Plot, 'add_axes')
def add_axes(plot, yy=0., xx=0., color=SLATE):
    H,W,M = plot.HWM
    rr,cc = plot.cell_from(yy,xx)
    plot.pixels[rr   , M:M+W] = color
    plot.pixels[M:M+H, cc   ] = color
    return plot

#-------  special markers  ----------------------------------------------------

@bind_to_class(Plot, 'cross_at')
def cross_at(plot, yy=0., xx=0., size=5, color=BLACK, opacity=1.):
    r,c = plot.cell_from(yy,xx)
    S = size
    overlay(plot.pixels[r-S:r+S+1, c         ], color, opacity)
    overlay(plot.pixels[r         , c-S:c+S+1], color, opacity)
    return plot

@bind_to_class(Plot, 'box_at')
def box_at(plot, yy=0., xx=0., size=5, color=BLACK, opacity=1.):
    r,c = plot.cell_from(yy,xx)
    S = size
    overlay(plot.pixels[r-S       , c-S:c+S+1], color, .9 * opacity)
    overlay(plot.pixels[    r+S   , c-S:c+S+1], color, .9 * opacity)
    overlay(plot.pixels[r-S:r+S+1 , c-S      ], color, .9 * opacity)
    overlay(plot.pixels[r-S:r+S+1 ,     c+S  ], color, .9 * opacity)
    overlay(plot.pixels[r-S:r+S+1 , c-S:c+S+1], color, .3 * opacity)
    return plot

#-------  display data  -------------------------------------------------------

@bind_to_class(Plot, 'scatter_points')
def scatter_points(plot, coors, color=BLACK, opacity=1., rad=3.):
    _,_,M = plot.HWM
    color = color if type(color)==type([]) else [color for _ in coors]
    for col,(y,x) in zip(color, coors):
        r,c = plot.cell_from(y,x)
        mask = np.mgrid[-M:M+1 , -M:M+1]
        mask = rad**2 / (rad**2 + mask[0]**2 + mask[1]**2)
        mask = np.minimum(1., opacity * mask**2)[:,:,np.newaxis]
        overlay(plot.pixels[r-M:r+M+1, c-M:c+M+1], col, mask)
    return plot

@bind_to_class(Plot, 'shade_hypothesis')
def shade_hypothesis(plot, decfunc, color_pos=ORANGE, color_neg=BLUE,
                     opacity=.10, sharpness=1.4):
    # TODO: separate out (y,x)->(color,opacity) logic??
    H,W,M = plot.HWM
    for dy in [-.3,+.3]:
        for dx in [-.3,+.3]:
            cells = np.mgrid[M+dy:M+H+dy, M+dx:M+W+dx]
            coors = plot.coor_from(cells[0], cells[1])
            coors = np.moveaxis(coors, 0, -1)
            dec = decfunc(coors)
            mask = opacity * np.exp(-sharpness**2 * dec**2) [:,:,np.newaxis]
            ## TODO: vectorize the following line
            colors = np.array([[color_pos if dd<0 else color_neg for dd in d]
                               for d in dec                                  ])
            overlay(plot.pixels[M:-M,M:-M], colors, mask)
    return plot

@bind_to_class(Plot, 'shade_heatmap')
def shade_heatmap(plot, intensity, color=SHADE):
    H,W,M = plot.HWM
    cells = np.mgrid[M:M+H, M:M+W]
    coors = plot.coor_from(cells[0], cells[1])
    coors = np.moveaxis(coors, 0, -1)
    mask  = np.maximum(0.,np.minimum(1., intensity(coors) ))
    overlay(plot.pixels[M:-M,M:-M], color, mask[:,:,np.newaxis])
    return plot

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Use the Plot Class for ML Plots  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#-------  _  ------------------------------------------------------------------

def plot_feature_space(feature_vectors, labels, file_name,
                       ymin=0., ymax=1., xmin=0., xmax=1.,
                       weights=[], dec_func_maker=None,
                       opacity=1.0,                       ):
    plot = blank(480, 480, 6, ymin, ymax, xmin, xmax)
    plot.add_gridlines(np.linspace(ymin,ymax,11)[1:-1],
                       np.linspace(xmin,xmax,11)[1:-1] )
    plot.add_ticks().add_axes()

    for w in weights:
        decfunc = decfunc_maker(w)
        plot.shade_hypothesis(decfunc)
    plot.scatter_points(feature_vectors, opacity=opacity,
                        color=[ORANGE if l==+1 else BLUE for l in labels])

    plot.save_to(file_name)

#-------  _  ------------------------------------------------------------------

#def plot_weight_space(error_by_param, file_name,
#                      ymin=-10., ymax=+10., xmin=-10., xmax=+10.,
#                      weight_color_pairs=[],                     ):
#
#    plot = blank(480, 480, 6, ymin, ymax, xmin, xmax)
#    plot.add_gridlines(np.linspace(ymin,ymax,11)[1:-1],
#                       np.linspace(xmin,xmax,11)[1:-1], opacity=1.)
#
#    # TODO: fill in this essence!  using kd tree voronoi??
#    #plot.shade_heatmap(lambda ws:[]
#
#    for (y,x), col in weight_color_pairs:
#        plot.box_at(y, x, color=col)
#
#    plot.add_gridlines(np.linspace(ymin,ymax,11)[1:-1],
#                       np.linspace(xmin,xmax,11)[1:-1], opacity=1.)
#    plot.save_to(file_name)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~  Render Digits  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO: decide where to put this
#render_digit = lambda x : SMOKE * (1.0-np.repeat(x[:,:,np.newaxis], 3, axis=2))

#for i,idx in enumerate(train_idxs[:NB_DIGITS_RENDERED]):
#    plt.imsave('mnist-trn-{:02d}.png'.format(i), render_digit(all_x[idx]))
#    if not PRINT_TRAIN_FEATURES: continue
#    print('train example {:02d} has darkness {:.2f} and height {:.2f}'.format(
#        i, darkness(all_x[idx]), width(all_x[idx])
#        ))


