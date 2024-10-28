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
import functools as ft
from fastcore.meta import delegates
import tyro
from einops import rearrange
from typing import *

@dataclass
class Args:
    device: str = "0" # or cuda device e.g., "0" or "7"
    initial_lr: float = 2e-2 # synaptic learning rate
    Kx: int = 10 # number of hidden units in x-direction
    Ky: int = 10 # number of hidden units in y-direction
    mu: float = 0.0 # mean of the initial weights
    sigma: float = 1.0 # std of the initial weights
    num_epochs: int = 200 # number of epochs
    batch_size: int = 1000 # batch size
    prec: float = 1e-30 # precision.
    delta: float = 0.0 # Strength of the anti-hebbian learning
    p: float = 2.0 # Lebesgue norm of the weights
    k: int = 2 # ranking parameter, must be integer that is bigger or equal than 2
    seed: int = 45 # random seed
    redraw_canvas_every: Optional[int] = 1 # redraw the canvas every n epochs. If None, only redraw at the end of training.

    def get_nhid(self):
        return self.Kx * self.Ky # Total number of hidden units

args = Args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

## Load data
def load_data():
    if os.path.exists("Xtrain.npy"):
        Xtrain = np.load("Xtrain.npy")
    else:
        data = load_dataset("mnist").with_format("numpy")
        Xtrain = np.array(data['train']['image'])
        np.save("Xtrain.npy", Xtrain)

    M = Xtrain.reshape(Xtrain.shape[0], -1) / 255.
    return M
    Ns = M.shape[0]

## Draw the weights, red=positive, blue=negative
def draw_weights(fig, synapses, Kx, Ky):
    HM = rearrange(synapses, "(kx ky) (h w) -> (kx h) (ky w)", kx=Kx, ky=Ky, h=28, w=28)
    fig.clf()
    ax = fig.gca()
    nc=np.amax(np.absolute(HM))
    im=ax.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    ax.set_axis_off()
    fig.canvas.draw()   
    return HM

def pnormed(W, p=2.):
    sign = jnp.sign(W)
    pW = jnp.abs(W) ** (p-1)
    return sign * pW

@jax.jit
def learning_rule(inp: Float[Array, "D"], synapses: Float[Array, "H D"], delta: float=0.0, k: int=2, p: float=2.):
    hid, D = synapses.shape

    # "Forward pass" through synapse
    y=pnormed(synapses, p) @ inp # (H,)

    # Make one-hot softmax of hidden activations
    argy_sorted=jnp.argsort(y, axis=0, descending=True)
    yact = jnp.zeros((hid,)) # (H,)
    yact = yact.at[argy_sorted[0]].set(1.0) # (H,)

    # Add anti-Hebbian update to kth strongest memory (only if delta != 0)
    yact = yact.at[argy_sorted[k]].set(-delta) # (H,)

    ds0 = yact[:, None] * inp[None] # (H, D) Standard Hebbian term: increase weights of strongest memory, decrease weight of kth strongest memory
    ds1 = (yact * y)[:, None] * synapses # (H, D) Decay on the modified weights
    ds = ds0 - ds1 # (H, D)
    return ds

@jax.jit
@delegates(learning_rule)
def step_batch(inputs: Float[Array, "N D"], synapses: Float[Array, "H D"], eps: float, prec: float=1e-30, **kwargs):
    lrule = ft.partial(learning_rule, **kwargs)
    ds = jnp.sum(jax.vmap(lrule, in_axes=(0, None))(inputs, synapses), axis=0) # JAX handles reduction efficiently when jitted
    nc = jnp.max(jnp.absolute(ds))
    nc = jax.lax.select(nc < prec, prec, nc)
    synapses = synapses + eps*jnp.true_divide(ds,nc)
    return synapses

def is_interactive() -> bool:
    """Check if the code is running in an interactive python script, or as a real script."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if __name__ == "__main__":
    if is_interactive():
        args = Args()
    else:
        args = tyro.cli(Args)

    ## Training loop
    M = load_data()
    Ns = M.shape[0]
    D = M.shape[-1]

    key, rng = jr.split(jax.random.PRNGKey(args.seed))
    synapses = jr.normal(key, (args.get_nhid(), D)) * args.sigma + args.mu

    fig=plt.figure(figsize=(12.9,10))
    img_checkpoints = []
    # camera = Camera(fig)
    for nep in tqdm(range(args.num_epochs)):
        eps=args.initial_lr*(1-nep/args.num_epochs)
        key, rng = jr.split(rng)
        shuffle_idxs = jr.permutation(key, jnp.arange(Ns))
        M=M[shuffle_idxs]
        for i in range(Ns//args.batch_size):
            batch_idxs = slice(i*args.batch_size, (i+1)*args.batch_size)
            synapses = step_batch(M[batch_idxs], synapses, eps, args.prec, delta=args.delta, k=args.k, p=args.p)
            
        if args.redraw_canvas_every is not None and nep % args.redraw_canvas_every == 0:
            img = draw_weights(fig, synapses, args.Kx, args.Ky)
            img_checkpoints.append(img)
            # camera.snap()

    if args.redraw_canvas_every is None:
        img = draw_weights(fig, synapses, args.Kx, args.Ky)
        img_checkpoints.append(img)

#%%
    import matplotlib.animation as animation
    print("Saving animation...")
    fig, ax = plt.subplots()
    img = img_checkpoints[0]
    nc = np.amax(np.absolute(img))
    im = ax.imshow(img, cmap='bwr', vmin=-nc, vmax=nc)
    colorbar = fig.colorbar(im, ticks=[np.amin(img), 0, np.amax(img)])

    # Create FuncAnimation
    def update(frame: int):
        ax.clear()
        img = img_checkpoints[frame]
        nc=np.amax(np.absolute(img))
        im=ax.imshow(img,cmap='bwr',vmin=-nc,vmax=nc)
        colorbar.mappable.set_clim(vmin=-nc, vmax=nc)
        colorbar.set_ticks([-nc, 0, nc])
        ax.set_axis_off()
        fig.canvas.draw()   
        return [im]

    # Save the animation object to prevent garbage collection
    anim = animation.FuncAnimation(
        fig=fig, 
        func=update, 
        frames=len(img_checkpoints), 
        interval=80,
        blit=True  # This can make the animation more efficient
    )

    # anim.save('animation.gif', writer='pillow')
    anim.save('animation.mp4', writer='ffmpeg')
    print("Animation saved to animation.mp4")