#%% 
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from tqdm.auto import tqdm
from jaxtyping import Array, Float

@dataclass
class Args:
    device: str = "0" # or cuda device e.g., "0" or "7"
    learning_rate: float = 2e-2 # synaptic learning rate
    Kx: int = 10 # number of hidden units in x-direction
    Ky: int = 10 # number of hidden units in y-direction
    mu: float = 0.0 # mean of the initial weights
    sigma: float = 1.0 # std of the initial weights
    num_epochs: int = 200 # number of epochs
    batch_size: int = 2000 # batch size
    prec: float = 1e-30 # precision.
    delta: float = 0.0 # Strength of the anti-hebbian learning
    p: float = 2.0 # Lebesgue norm of the weights
    k: int = 2 # ranking parameter, must be integer that is bigger or equal than 2

    def get_nhid(self):
        return self.Kx * self.Ky # Total number of hidden units

args = Args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

## Load data
if os.path.exists("Xtrain.npy"):
    Xtrain = np.load("Xtrain.npy")
else:
    data = load_dataset("mnist").with_format("numpy")
    Xtrain = np.array(data['train']['image'])
    np.save("Xtrain.npy", Xtrain)

M = Xtrain.reshape(Xtrain.shape[0], -1) / 255.
Nc = 10
Ns = M.shape[0]
D = M.shape[-1]

## Draw the weights, red=positive, blue=negative
def draw_weights(fig, synapses, Kx, Ky):
    yy=0
    HM=np.zeros((28*Ky,28*Kx))
    for y in range(Ky):
        for x in range(Kx):
            HM[y*28:(y+1)*28,x*28:(x+1)*28]=synapses[yy,:].reshape(28,28)
            yy += 1
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    fig.canvas.draw()   

eps0=args.learning_rate # learning rate
Kx=args.Kx
Ky=args.Ky
hid=args.get_nhid()    # number of hidden units that are displayed in Ky by Kx array
mu=args.mu
sigma=args.sigma
Nep=args.num_epochs # number of epochs
Num=args.batch_size # size of the minibatch
jnp.finfo(jnp.float32).tiny
prec=args.prec
delta=args.delta    # Strength of the anti-hebbian learning
p=args.p
k=args.k          # ranking parameter, must be integer that is bigger or equal than 2

fig=plt.figure(figsize=(12.9,10))

key, rng = jr.split(jax.random.PRNGKey(45))
synapses = jr.normal(key, (hid, D)) * sigma + mu

def pnormed(W, p=2.):
    sign = jnp.sign(W)
    pW = jnp.abs(W) ** (p-1)
    return sign * pW

@jax.jit
def learning_rule(inp: Float[Array, "D"], synapses: Float[Array, "H D"], eps: float):
    hid, D = synapses.shape
    y=pnormed(synapses, p) @ inp # (H,)

    # Make one-hot softmax of hidden activations
    argy_sorted=jnp.argsort(y, axis=0, descending=True)
    yact = jnp.zeros((hid,)) # (H,)
    yact = yact.at[argy_sorted[0]].set(1.0) # (H,)

    # Anti-Hebbian update kth strongest memory (only if delta != 0)
    yact = yact.at[argy_sorted[k]].set(-delta) # (H,)

    # Sum all maximum activations across minibatch
    xx = yact * y # (H,)

    ds0 = yact[:, None] * inp[None] # (H, D) Hebbian term: increase weights of strongest memory, decrease weights of kth strongest memory
    ds1 = xx[:, None] * synapses # (H, D) Anti-Hebbian term. 
    ds = ds0 - ds1 # (H, D)

    return ds

@jax.jit
def step_batch(inputs: Float[Array, "N D"], synapses: Float[Array, "H D"], eps: float):
    ds = jnp.sum(jax.vmap(learning_rule, in_axes=(0, None, None))(inputs, synapses, eps), axis=0) # JAX handles reduction efficiently when jitted
    nc = jnp.max(jnp.absolute(ds))
    nc = jax.lax.select(nc < prec, prec, nc)
    synapses = synapses + eps*jnp.true_divide(ds,nc)
    return synapses

for nep in tqdm(range(Nep)):
    eps=eps0*(1-nep/Nep)
    key, rng = jr.split(rng)
    M=M[jr.permutation(key, jnp.arange(Ns)),:]
    for i in range(Ns//Num):
        inputs=M[i*Num:(i+1)*Num,:]
        synapses = step_batch(inputs, synapses, eps)
        
    draw_weights(fig, synapses, Kx, Ky)
# %% Old

# @jax.jit
# def step_batch(inputs: Float[Array, "N D"], synapses: Float[Array, "H D"], eps: float):
#     # Compute the normalized total input to hidden layer
#     y=pnormed(synapses, p) @ inputs.T # (H, N)

#     # Make one-hot softmax of hidden activations
#     argy_sorted=jnp.argsort(y, axis=0, descending=True)
#     yact = jnp.zeros((hid,Num)) # (H, N)
#     yact = yact.at[argy_sorted[0], jnp.arange(Num)].set(1.0) # (H, N)

#     # Anti-Hebbian update kth strongest memory (only if delta != 0)
#     yact = yact.at[argy_sorted[k], jnp.arange(Num)].set(-delta) # (H, N)

#     # Sum all maximum activations across minibatch
#     xx = (yact * y).sum(1) # (H,)

#     ds0 = yact @ inputs # (H, D) Hebbian term: increase weights of strongest memory, decrease weights of kth strongest memory
#     ds1 = xx[:, None] * synapses # (H, D) Anti-Hebbian term. 
#     ds = ds0 - ds1 # (H, D)

#     nc = jnp.max(jnp.absolute(ds))
#     nc = jax.lax.select(nc < prec, prec, nc)
#     synapses = synapses + eps*jnp.true_divide(ds,nc) # Why is ds1 subtracted?
#     return synapses
