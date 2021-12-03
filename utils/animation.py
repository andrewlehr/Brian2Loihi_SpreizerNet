import pylab as pl
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

def images(ax, h, *args, **kwargs):

    # First set up the figure, the axis, and the plot element we want to animate
    im = ax.imshow(h[0], *args, **kwargs)

    # initialization function: plot the background of each frame
    def init():
        im.set_array([])
        return im,

    def animate(ii):
        im.set_array(h[ii])
        ax.set_title('%s'%ii)
        pl.draw()
        return im,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(ax.get_figure(), animate, frames=len(h), interval=50, blit=True)

    return anim


def make_animation(gs, ts, bin_size, simtime, nPopE, output_file, fps):
    nRowE = int(np.sqrt(nPopE))
    ts_bins = np.arange(0., simtime + 1, bin_size)
    h = np.histogram2d(ts, gs, bins=[ts_bins, range(nPopE + 1)])[0]
    hh = h.reshape(-1, nRowE, nRowE)

    fig, ax = plt.subplots(1)
    a = images(ax, hh, vmin=0, vmax=np.max(hh))
    a.save('%s.mp4' % ('movies/' + output_file), fps=fps,
            extra_args=['-vcodec', 'libx264'])
    plt.show()

#anim.save('animation_image.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
