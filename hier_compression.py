from hps import Hyperparams, cifar10, parse_args_and_update_hparams
from train_helpers import set_up_hyperparams, load_vaes, add_vae_arguments
from train import set_up_data, get_sample_for_visualization
import argparse, imageio
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler, DataLoader
from torch.utils.data import TensorDataset
import torch
import pickle
import seaborn as sns

def RD_points_step(vae, x, preprocess_fn, return_samples=False):
    data_input, target = preprocess_fn(x)
    batch_size = data_input.shape[0]
    ndims = np.prod(data_input.shape[1:])
    with torch.no_grad():
        latents = vae.forward_get_latents(data_input)
    zs = [s['z'].cuda() for s in latents]
    num_latent_per_layer = [np.prod(z.shape[1:]) for z in zs]
    num_latent_per_layer /= ndims
    kls = [s['kl'].cuda() for s in latents]
    rate_per_layer = [kl.sum(dim=(1,2,3)).cpu().detach().numpy() for kl in kls]
    rate_per_layer /= ndims
    #import pdb; pdb.set_trace()
    distortion_up_to_layer = [0] * len(kls)
    samples = [None] * len(kls)
    for i in range(len(zs)): #brute force, no caching
        with torch.no_grad():
            samples[i], distortion_up_to_layer[i] = vae.forward_samples_set_latents(batch_size, zs[:i], x_target=target)
    if not return_samples:
        return num_latent_per_layer, rate_per_layer, [distortion.cpu().detach().numpy() for distortion in distortion_up_to_layer]
    else:
        return samples, num_latent_per_layer, rate_per_layer, [distortion.cpu().detach().numpy() for distortion in distortion_up_to_layer]

    
def RD_points(vae, data, preprocess_fn, batch_size=16, transform_fn=None, file_name_to_save=None, **kwargs):
    step = 0
    rate_per_layer, distortion_up_to_layer = None, None
    for x in DataLoader(data, batch_size=batch_size, drop_last=True, pin_memory=True):#, sampler=valid_sampler):
        print(step)
        step += 1
        if transform_fn is not None:
            torch.random.manual_seed(42)
            x = transform_fn(x, **kwargs)
#            import pdb; pdb.set_trace()
        num_latent_per_layer, rate_per_layer_batch, distortion_up_to_layer_batch = RD_points_step(vae, x, preprocess_fn)
        if rate_per_layer is None:
            rate_per_layer = rate_per_layer_batch
        else:
            rate_per_layer = [np.append(rate_per_layer[i], rate_per_layer_batch[i]) for i in range(len(rate_per_layer))]
            
        if distortion_up_to_layer is None:
            distortion_up_to_layer = distortion_up_to_layer_batch
        else:
            distortion_up_to_layer = [np.append(distortion_up_to_layer[i], distortion_up_to_layer_batch[i]) for i in range(len(distortion_up_to_layer))]
    RD = {'num_latent': num_latent_per_layer, 'rate': rate_per_layer, 'distortion': distortion_up_to_layer}
    if file_name_to_save:
        with open(file_name_to_save, 'wb') as f:
            pickle.dump(RD , f)
    return RD


def get_cum_rate_from_RD(RD):
    return np.array(RD['rate']).cumsum(axis=0)
    

def preprocess_RD_for_mean_plot(RD, replace_infs=None, drop_nans=False, return_num_nans=False):
    cum_rate = get_cum_rate_from_RD(RD)
    cum_num_latents = np.array(RD['num_latent']).cumsum()
    cum_num_latents = cum_num_latents
    distortion = np.array(RD['distortion'])
    num_nans = np.isnan(cum_rate).sum(axis=1)
    if replace_infs is not None:
        cum_rate = np.nan_to_num(cum_rate, nan=replace_infs, posinf=replace_infs, neginf=-replace_infs)
        np.clip(cum_rate, -replace_infs, replace_infs, out=cum_rate)
        distortion = np.nan_to_num(distortion,  nan=replace_infs, posinf=replace_infs, neginf=-replace_infs)
        np.clip(distortion, -replace_infs, replace_infs, out=distortion)
    
    if drop_nans:
        cum_rate_mean = np.nanmean(cum_rate, axis=1)
        distortion_mean = np.nanmean(distortion, axis=1)
    else:
        cum_rate_mean = cum_rate.mean(axis=1)
        distortion_mean = distortion.mean(axis=1)
    if return_num_nans:
        return cum_rate_mean, cum_num_latents, distortion_mean, num_nans
    else:
        return cum_rate_mean, cum_num_latents, distortion_mean


def permute_channels(x):
    return [x[0][..., [2,0,1]]]


#def flip_images(x, axis=1):
#    assert axis in (0, 1), 'axis must be 0 for horizontal, 1 for vertical flip'
#    return [torch.flip(x[0], [axis])]

def flip_images(x, axis=0):
    assert axis in (0, 1), 'axis must be 0 for up-down, 1 for left-right flip'
    return [torch.flip(x[0], [axis + 1])]


def add_gaussian_noise_to_images(x, std=0.1):
    return [(x[0] + torch.randn(x[0].size()) * std).clamp(min=0, max=255).to(torch.uint8)]