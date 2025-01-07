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

# Classes
class GWSpectrum:
    """An example class for this module."""
    
    def __init__(self):
        """
        Initialize the class with a name.

        Args:
            name (str): The name of the instance.
        """

        self.K = np.logspace(-3,np.log10(300),100)
        
        self.model, self.graphdef, self.state = \
            self.load_model('../models/model_dict', hidden_sizes=(512,512), from_dict=True)

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

    def get_sepctrum(self, f, vw, alpha, Htau, HR, Ts, g_st=100, extrapolate=True):
    
        Oms = self.model( jnp.log10(jnp.array((vw, alpha, Htau, HR))) )
        
        fs = self.K * (2.626e-06) * (Ts / 100) / HR * (g_st / 100)**(1/6)

        if(extrapolate==True):
            return 10**jnp.interp(jnp.log10(f), jnp.log10(fs), Oms, 
                              left='extrapolate', right='extrapolate') \
                 * 1.6e-5 * (100 / g_st)**(1/3)  * 3 * (4/3)**2
        else:
            return 10 ** jnp.interp(jnp.log10(f), jnp.log10(fs), Oms, left=-100, right=-100) \
                * 1.6e-5 * (100 / g_st)**(1/3)  * 3 * (4/3)**2
