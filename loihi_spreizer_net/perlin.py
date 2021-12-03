import matplotlib.pyplot as plt
import numpy as np
import noise
# adopted from
# https://github.com/babsey/spatio-temporal-activity-sequence/blob/master/scripts/lib/connectivity_landscape.py


def generate_perlin(nrow, perlin_scale, save=False, seed_value=0):
    size = perlin_scale
    assert(size > 0)

    x = y = np.linspace(0, size, nrow)
    n = [[noise.pnoise2(i, j, repeatx=size, repeaty=size, base=seed_value)
          for j in y] for i in x]
    m = np.concatenate(n)
    sorted_idx = np.argsort(m)
    max_val = nrow * 2
    idx = len(m) // max_val
    for ii, val in enumerate(range(max_val)):
        m[sorted_idx[ii * idx:(ii + 1) * idx]] = val
    landscape = (m - nrow) / nrow
    landscape = np.add(-min(landscape), landscape)
    landscape = np.divide(landscape, max(landscape))
    landscape = np.multiply(2*np.pi, landscape)
    landscape = np.reshape(landscape, (nrow, nrow))
    if save:
        plt.figure(figsize=(4.5, 4.5))
        plt.imshow(landscape.T, cmap='hsv',
                   origin='lower', extent=[0, 1, 0, 1])
        plt.xlabel('x position', fontsize=13)
        plt.ylabel('y position', fontsize=13)
        plt.title('Perlin noise angular map', fontsize=14)
        cb = plt.colorbar()
        cb.set_label('$\phi$', fontsize=13)
        plt.savefig('figures/scale_'+str(perlin_scale)+'_perlinnoise')
        plt.close()

    return landscape
