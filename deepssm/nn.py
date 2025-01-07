import jax

import flax.nnx as nnx

# Neural Network Model (same as before)
class ISSM_NN(nnx.Module):
    def __init__(self, key=42, hidden_sizes=(256,256), out_size=100, dropout_rate=0):
        keys = nnx.Rngs(key)
        num_hidden = len(hidden_sizes)
        
        self.layers = [nnx.Linear(4, hidden_sizes[0], rngs=keys)]
        self.actv = nnx.relu
        
        self.layers.append(self.actv)
        
        for i in range(1, num_hidden):
            self.layers.append(nnx.Linear(hidden_sizes[i-1], hidden_sizes[i], rngs=keys))
            self.layers.append(self.actv)
            if(dropout_rate > 0):
                self.layers.append(nnx.Dropout(dropout_rate, rngs=keys))
        
        self.layers.append( nnx.Linear(hidden_sizes[-1], out_size, rngs=keys))
        
    
    def __call__(self, x):
        # Forward pass with ReLU activations
        for op in self.layers:
            x = op(x)
        return x.squeeze()