import jax
import jax.numpy as jnp
from jax import lax, vmap
import numpy as np
from jax import random, grad, jit
from flax import linen as nn
from flax.training import train_state
import optax
import orbax.checkpoint
import flax.nnx as nnx
import flax.linen as linen

import os
import numpy as np

import orbax.checkpoint as ocp

from .nn import ISSM_NN

import warnings

# Classes
class GWSpectrum:
    """Class for GW spectrum"""
    
    def __init__(self):
        """
        Initialize the class.

        """

        # Dimensionless wave number: k / R_*
        self.K = np.logspace(-3,np.log10(300),100)

        # load the NN model
        self.model, self.graphdef, self.state = \
            self.load_model('../models/model_dict', hidden_sizes=(512,512), from_dict=True)

    # load NN model
    def load_model(self, model_path, hidden_sizes=(512,512), from_dict=False):
        abstract_model = nnx.eval_shape(
            lambda: ISSM_NN(hidden_sizes=hidden_sizes))
        graphdef, abstract_state = nnx.split(abstract_model)
        if(from_dict == False):
            checkpointer = ocp.StandardCheckpointer()
            state_restored = checkpointer.restore(model_path + '/state', abstract_state)
        else:
            import pickle 
            with open(model_path, 'rb') as f:
                state_restored = pickle.load(f)
            abstract_state.replace_by_pure_dict(state_restored)
            state_restored = abstract_state
            
        model = nnx.merge(graphdef, state_restored)
        return model, graphdef, state_restored

    # return the GW spectrum
    def get_sepctrum(self, f, vw, alpha, Htau, HR, Ts, g_st=100, extrapolate=True):
        """
        Function to compute the spectrum of a signal based on given model parameters.
    
        Parameters:
        - f          : Frequency at which to evaluate the spectrum (array-like).
        - vw         : Bubble wall velocity.
        - alpha      : Phase transition strength.
        - Htau       : H_* \tau_SW, duration of phase transition.
        - HR         : H_* R_*, bubble seperation.
        - Ts         : Phase transition temperature T_*.
        - g_st       : Degrees of freedom of relativistic species (default: 100).
        - extrapolate: Boolean indicating whether to extrapolate the spectrum. If False, return 1e-100.
    
        Returns:
        - Spectrum values corresponding to the input frequencies `f`.
        """
        Oms = self.model( jnp.log10(jnp.array((vw, alpha, Htau, HR))) )
        
        fs = self.K * (2.626e-06) * (Ts / 100) / HR * (g_st / 100)**(1/6)
        if(vw < 0.01 or vw > 0.99 
           or alpha < 0.01 or alpha > 0.33
           or Htau < 0.0001 or Htau > 1
          or HR < 0.0001 or HR > 1):
            warnings.warn("Some of the parameters are out of the recommanded range, please consider change them.")
        
        if( f.max() > fs.max() or f.min() < fs.min()):
            warnings.warn("The input Freuqncies are out of the range of supported frequencies!")
            if(extrapolate == True):
                warnings.warn("Extrapolations are used!")
            else:
                warnings.warn("Those outliers are set to 1e-100!")

        
        if(extrapolate == True):
            return 10**jnp.interp(jnp.log10(f), jnp.log10(fs), Oms, 
                              left='extrapolate', right='extrapolate') \
                 * 1.6e-5 * (100 / g_st)**(1/3)  * 3 * (4/3)**2
        else:
            return 10 ** jnp.interp(jnp.log10(f), jnp.log10(fs), Oms, left=-100, right=-100) \
                * 1.6e-5 * (100 / g_st)**(1/3)  * 3 * (4/3)**2
