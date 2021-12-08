import numpy as np

# Network dimensions
network_dimensions = {
    'n_pop_e' : 3600, #14400,         # Number of e neurons (must be a square number)
    'n_pop_i' : 900, #3600            # Number of i neurons (must be a square number)
    }

network_dimensions['n_row_e'] = network_dimensions['n_col_e'] = int(np.sqrt(network_dimensions['n_pop_e']))
network_dimensions['n_row_i'] = network_dimensions['n_col_i'] = int(np.sqrt(network_dimensions['n_pop_i']))

assert (int(np.sqrt(network_dimensions['n_pop_e'])) == np.sqrt(network_dimensions['n_pop_e'])) and \
            (int(np.sqrt(network_dimensions['n_pop_i'])) == np.sqrt(network_dimensions['n_pop_i'])), \
                'n_pop_e or n_pop_i is not a square number!'

spike_generator_params = {
    'p_sg': .05, #0.1,   # connection probability
    'w': 2       #12     # synaptic weight
    }

neuron_params = {
    'thr_v_mant': 2188, #8750,  # threshold mantissa (threshold = thr_v_mant * 2^6)
    'tau_ref': 2,               # refractory period
    'decay_v': 400,             # voltage decay
    'decay_I': 380              # current decay
    }

# Synapse parameters
synapse_params = {
    'Je' : 30,                  # excitatory synaptic current
    'g' :  8,                   # ratio of recurrent inhibition and excitation
    'sigma_e' : 0.067,          # excitatory connectivity width
    'sigma_i' : 0.1,            # inhibitory connectivity width
    'p_e' : 0.05,               # connection probability of e neurons
    'p_i' : 0.05                # connection probability of i neurons
    }

synapse_params['Ji'] = -synapse_params['g'] * synapse_params['Je']
synapse_params['amp_e'] = synapse_params['p_e'] / (2 * np.pi * synapse_params['sigma_e']**2)
synapse_params['amp_i'] = synapse_params['p_i'] / (2 * np.pi * synapse_params['sigma_i']**2)

# Connectivity profiles
p_con = {
    'ee' : 'amp_e*exp(-(torus_distance(x_pre+x_shift_pre, x_post, y_pre+y_shift_pre, y_post)**2)/(2*sigma_e**2))',
    'ie' : 'amp_e*exp(-(torus_distance(x_pre, x_post, y_pre, y_post)**2)/(2*sigma_e**2))',
    'ei' : 'amp_i*exp(-(torus_distance(x_pre, x_post, y_pre, y_post)**2)/(2*sigma_i**2))',
    'ii' : 'amp_i*exp(-(torus_distance(x_pre, x_post, y_pre, y_post)**2)/(2*sigma_i**2))'
    }

# Perlin scale and grid offset
perlin_scale = 4
grid_offset = 1
