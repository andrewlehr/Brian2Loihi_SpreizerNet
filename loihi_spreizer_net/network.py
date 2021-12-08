# Original code Copyright 2021 Leo Hiselius
# Modified code Copyright 2021 Andrew Lehr
# The MIT License

from brian2_loihi import *
import numpy as np
from scipy import interpolate
import warnings
from .perlin import generate_perlin
from .torus_distance import torus_distance
from .params import *

class LoihiSpreizerNet:  

    def __init__(self):
        self.network = LoihiNetwork()
        self.neuron_groups = {'e' : None, 'i' : None}
        self.synapses = {'ee' : None, 'ie' : None, 'ei' : None, 'ii' : None}
        self.spike_monitors = {'e' : None, 'i' : None}
        self.state_monitors = {'e' : None, 'i' : None}
        self.spike_generator = {'sg_e' : None}
        self.spike_generator_synapses = {'sg_e' : None}
        self.p_con = p_con
      
    def set_seed(self, seed_value=0):
        """Sets the seed for any stochastic elements of the simulation.

        Args:
            seed_value (int, optional): The seed value. Defaults to 0.
        """        
        # seed(seed_value) not supported by brian2loihi, i think
        np.random.seed(seed_value)
  
    def populate(self):     
        """Fills the network with e and i neurons on evenly spaced [0,1]x[0,1]-grid.
        """  
        
        # pass this to the LoihiNeuronGroup into its eqn_str which adds it to the neuron model
        pos_variables = '''
                        x : 1
                        y : 1
                        x_shift : 1
                        y_shift : 1
                        '''
        
        # Instantiate the excitatory and inhibitory networks
        self.neuron_groups['e'] = LoihiNeuronGroup(network_dimensions['n_pop_e'],
                                                   refractory = neuron_params['tau_ref'],
                                                   threshold_v_mant = neuron_params['thr_v_mant'],
                                                   decay_v = neuron_params['decay_v'],
                                                   decay_I = neuron_params['decay_I'],
                                                   name = 'e_neurons',
                                                   eqn_str = pos_variables)
        
        self.neuron_groups['i'] = LoihiNeuronGroup(network_dimensions['n_pop_i'], 
                                                   refractory = neuron_params['tau_ref'], 
                                                   threshold_v_mant = neuron_params['thr_v_mant'],
                                                   decay_v = neuron_params['decay_v'],
                                                   decay_I = neuron_params['decay_I'],
                                                   name = 'i_neurons',
                                                   eqn_str = pos_variables)

        # Place neurons on evenly spaced grid [0,1]x[0,1]. i neurons are shifted to lay in between e neurons.
        n_row_e = float(network_dimensions['n_row_e'])
        n_row_i = float(network_dimensions['n_row_i'])
        n_col_e = float(network_dimensions['n_col_e'])
        n_col_i = float(network_dimensions['n_col_i'])
        self.neuron_groups['e'].x = '(i // n_col_e) / n_col_e'
        self.neuron_groups['e'].y = '(i % n_row_e) / n_row_e'
        self.neuron_groups['i'].x = '(i // n_col_i) / n_col_i + 1/(2*n_col_e)'
        self.neuron_groups['i'].y = '(i % n_row_i) / n_row_i + 1/(2*n_row_e)'

        # Add to network attribute
        self.network.add(self.neuron_groups['e'])
        self.network.add(self.neuron_groups['i'])
    
    def connect(self, allow_multiple_connections=True, perlin_seed_value=0):
        """Connects neurons with synapses.

        Args:
            allow_multiple_connections (bool, optional): If True, multiple connections can be made where probability
                                                            of connecting is greater than one. Defaults to True.
            perlin_seed (int, optional): Seed passed to generate_perlin(). Defaults to 0.
        """      
        
        # Generate perlin map
        perlin_map = generate_perlin(int(np.sqrt(network_dimensions['n_pop_e'])), perlin_scale,
            seed_value=perlin_seed_value)

        idx = 0
        for i in range(network_dimensions['n_col_e']):
            for j in range(network_dimensions['n_row_e']):
                self.neuron_groups['e'].x_shift[idx] = grid_offset / network_dimensions['n_col_e'] * np.cos(perlin_map[i, j])
                self.neuron_groups['e'].y_shift[idx] = grid_offset / network_dimensions['n_row_e'] * np.sin(perlin_map[i, j])
                idx += 1
      
        # Define synapses
        self.synapses['ee'] = LoihiSynapses(self.neuron_groups['e'], 
                                            sign_mode=synapse_sign_mode.EXCITATORY)
        self.synapses['ie'] = LoihiSynapses(self.neuron_groups['e'], self.neuron_groups['i'], 
                                            sign_mode=synapse_sign_mode.EXCITATORY)
        self.synapses['ei'] = LoihiSynapses(self.neuron_groups['i'], self.neuron_groups['e'], 
                                            sign_mode=synapse_sign_mode.INHIBITORY)
        self.synapses['ii'] = LoihiSynapses(self.neuron_groups['i'], 
                                            sign_mode=synapse_sign_mode.INHIBITORY)
        
        # Synapse parameters
        for param in synapse_params:
            self.synapses['ee'].namespace[param] = synapse_params[param]
            self.synapses['ie'].namespace[param] = synapse_params[param]
            self.synapses['ei'].namespace[param] = synapse_params[param]
            self.synapses['ii'].namespace[param] = synapse_params[param]
        
        # Make synapses
        synapse_names = ['ee', 'ie', 'ei', 'ii']
        if allow_multiple_connections:
            for syn_name in synapse_names:
                p_con = self.p_con[syn_name]
                n_con = '(int(' + p_con + ')+1)'                    # ceil the likelihood. This is number of connections
                p_con += '/' + n_con                                # divide the likelihood with its ceil
                self.synapses[syn_name].connect(p=p_con, n=n_con)   # n_con connections are made with p=p_con
        else:
            for syn_name in synapse_names:
                self.synapses[syn_name].connect(p=p_con[syn_name])
                
        self.synapses['ee'].w = synapse_params['Je']
        self.synapses['ie'].w = synapse_params['Je']
        self.synapses['ei'].w = synapse_params['Ji']
        self.synapses['ii'].w = synapse_params['Ji']

        # Add to network attribute
        self.network.add(self.synapses['ee'])
        self.network.add(self.synapses['ie'])
        self.network.add(self.synapses['ei'])
        self.network.add(self.synapses['ii'])

    def print_average_projections(self):
        """Prints the average number of outgoing projections for the e and i population, respectively.
        """        
        e_out = np.add(self.synapses['ee'].N_outgoing_pre, self.synapses['ie'].N_outgoing_pre)
        i_out = np.add(self.synapses['ei'].N_outgoing_pre, self.synapses['ii'].N_outgoing_pre)
        print('Number of outgoing projections for e network is ' +
              str(np.mean(e_out)) + ' +/- ' + str(np.std(e_out)))
        print('Number of outgoing projections for i network is ' +
              str(np.mean(i_out)) + ' +/- ' + str(np.std(i_out)))

    def set_initial_potentials(self, start_at_rest=True, exc_potentials=[], inh_potentials=[]):
        """Sets the initial membrane potentials for all neurons

        Args:
            start_at_rest (bool, optional):   Controls if neurons start at rest or uniformly random between Vr and Vt. 
                                              Defaults to True.
            exc_potentials (array, optional): Use this to explicitly set the initial voltages. Must set start_at_rest=False
                                              and provide inh_potentials as well. Defaults to empty list.
        """        
        if start_at_rest:      
            self.neuron_groups['e'].v = 'Vr'
            self.neuron_groups['i'].v = 'Vr'
        elif len(exc_potentials):
            self.neuron_groups['e'].v = 'Vr + rand() * (Vt - Vr)'
            self.neuron_groups['i'].v = 'Vr + rand() * (Vt - Vr)'
        elif len(exc_potentials) and len(inh_potentials):
            self.neuron_groups['e'].v = exc_potentials
            self.neuron_groups['i'].v = inh_potentials
        else:
            print('Error in setting initial potentials.')
     
    def connect_spike_generator(self, N, r, T):
        
        # define spikes
        n_spikes = int(r * T / 1000)
        n_time_steps = T # since defaultclock.dt = 1 in brian2loihi
        ts = np.zeros((N,n_spikes))
        ids = np.zeros((N,n_spikes))
        for neuron in range(N):
            ts[neuron, :] = np.sort(np.random.choice(n_time_steps, size=n_spikes, replace=False))
            ids[neuron, :] = neuron*np.ones(n_spikes)
        ts = ts.reshape(N*n_spikes)
        ids = ids.reshape(N*n_spikes)
        
        # Create spike generator group
        self.spike_generator['sg_e'] = LoihiSpikeGeneratorGroup(N, ids, ts)
        
        # Create synapses
        self.spike_generator_synapses['sg_e'] = LoihiSynapses(self.spike_generator['sg_e'], 
                                                         self.neuron_groups['e'],
                                                         sign_mode=synapse_sign_mode.EXCITATORY)
            
        # connect
        self.spike_generator_synapses['sg_e'].connect(p=spike_generator_params['p_sg'])
        self.spike_generator_synapses['sg_e'].w = spike_generator_params['w']
        
        self.network.add(self.spike_generator['sg_e'])
        self.network.add(self.spike_generator_synapses['sg_e'])
        
    def connect_spike_monitors(self):
        """Connect spike monitors to neuron_groups.
        """        
        self.spike_monitors['e'] = LoihiSpikeMonitor(self.neuron_groups['e'])
        self.spike_monitors['i'] = LoihiSpikeMonitor(self.neuron_groups['i'])
        self.network.add(self.spike_monitors['e'])
        self.network.add(self.spike_monitors['i'])


    def store_network(self):
        """Stores network. restore() restores to this point.
        """        
        self.network.store()

    def run_sim(self, simulation_time=1):
        """Runs the simulation.

        Args:
            simulation_time (second, optional): duration of the simulation. Defaults to 1*second.
            is_report (bool, optional): determines if simulation progress is reported in terminal or not. Defaults to True
        """   
        # Print warning if the spike_monitors are not connected
        if self.spike_monitors['e'] is None or self.spike_monitors['i'] is None:
            warnings.warn('Simulation running without SpikeMonitors!')

        # Run simulation
        self.network.run(simulation_time)

    def save_monitors(self, simulation_name, SUPERDIR=''):
        """Saves any existing montitors (spike and state) to folders spike_monitors and state_monitors, respectively.

        Args:
            simulation_name (str): The name of the simulation.
            SUPERDIR (str, optional): The directory in which the "saves"-directory exists. Defaults to ''.
        """        
        # Save spike_monitors
        if self.spike_monitors['e'] is not None: 
            np.save(SUPERDIR+'saves/spike_monitors/'+simulation_name+'_e_i', self.spike_monitors['e'].i[:])
            np.save(SUPERDIR+'saves/spike_monitors/'+simulation_name+'_e_t', self.spike_monitors['e'].t[:])
        if self.spike_monitors['i'] is not None:
            np.save(SUPERDIR+'saves/spike_monitors/'+simulation_name+'_i_i', self.spike_monitors['i'].i[:])
            np.save(SUPERDIR+'saves/spike_monitors/'+simulation_name+'_i_t', self.spike_monitors['i'].t[:])

        # Save state_monitors
        if self.state_monitors['e'] is not None:
            np.save(SUPERDIR+'saves/state_monitors/'+simulation_name+'_e_v', self.state_monitors['e'].v[:])
            np.save(SUPERDIR+'saves/state_monitors/'+simulation_name+'_e_t', self.state_monitors['e'].t[:])
        if self.state_monitors['i'] is not None:
            np.save(SUPERDIR+'saves/state_monitors/'+simulation_name+'_i_v', self.state_monitors['i'].v[:])
            np.save(SUPERDIR+'saves/state_monitors/'+simulation_name+'_i_t', self.state_monitors['i'].t[:])

    def restore_network(self):
        """Restores all attributes of the current class instance to the point where store_network() was called.
        """        
        if hasattr(self.network, 'sorted_objects'):
            if self.spike_generator in self.network.sorted_objects:
                self.network.remove(self.spike_generator)
                self.network.remove(self.spike_generator_synapses)
        self.network.restore()
