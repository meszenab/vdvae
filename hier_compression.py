from hps import Hyperparams, cifar10, parse_args_and_update_hparams
from train_helpers import set_up_hyperparams, load_vaes, add_vae_arguments
from train import set_up_data, get_sample_for_visualization
import argparse, imageio
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler, DataLoader, TensorDataset, Subset
import torch
import pickle
import seaborn as sns
from torchvision.utils import save_image, make_grid

def RD_points_step(vae, x, preprocess_fn, t=None, return_samples=False, return_L2_dist=False):
    data_input, target = preprocess_fn(x)
    #import pdb; pdb.set_trace()
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
    distortion_up_to_layer = [0] * len(kls)
    L2_dist_up_to_layer = [0] * len(kls)
    samples = [None] * len(kls)
    for i in range(len(zs) + 1): #brute force, no caching
        with torch.no_grad():
            if return_L2_dist:
                samples[i], distortion_up_to_layer[i], L2_dist_up_to_layer[i] = vae.forward_samples_set_latents(batch_size, zs[:i], t=t, x_target=target, x_orig=x[0])
            else:
                samples[i], distortion_up_to_layer[i] = vae.forward_samples_set_latents(batch_size, zs[:i], t=t, x_target=target)
    if not return_samples:
        if return_L2_dist:
            return num_latent_per_layer, rate_per_layer, [distortion.cpu().detach().numpy() for distortion in distortion_up_to_layer], [L2_dist.cpu().detach().numpy() for L2_dist in L2_dist_up_to_layer]
        else:
            return num_latent_per_layer, rate_per_layer, [distortion.cpu().detach().numpy() for distortion in distortion_up_to_layer]
    else:
        if return_L2_dist:
            return samples, num_latent_per_layer, rate_per_layer, [distortion.cpu().detach().numpy() for distortion in distortion_up_to_layer], [L2_dist.cpu().detach().numpy() for L2_dist in L2_dist_up_to_layer]
        else:
            return samples, num_latent_per_layer, rate_per_layer, [distortion.cpu().detach().numpy() for distortion in distortion_up_to_layer]

    
def RD_points(vae, data, preprocess_fn, return_L2_dist=False, batch_size=16, transform_fn=None, file_name_to_save=None, t=None, return_samples=False, idx_list=None, **kwargs):
    if idx_list is not None:
        data = Subset(data, idx_list)
    step = 0
    samples, rate_per_layer, distortion_up_to_layer = None, None, None
    
    for x in DataLoader(data, batch_size=batch_size, drop_last=False, pin_memory=True):#, sampler=valid_sampler):
        print(step)
        step += 1
        if transform_fn is not None:
            torch.random.manual_seed(42)
            x = transform_fn(x, **kwargs)
#            import pdb; pdb.set_trace()
        if return_L2_dist:
            samples_batch, num_latent_per_layer, rate_per_layer_batch, distortion_up_to_layer_batch, L2_dist_up_to_layer_batch = RD_points_step(vae, x, preprocess_fn, t=t, return_L2_dist=return_L2_dist, return_samples=True)
        else:
            samples_batch, num_latent_per_layer, rate_per_layer_batch, distortion_up_to_layer_batch = RD_points_step(vae, x, preprocess_fn, t=t, return_samples=True)

        
        if rate_per_layer is None:
            rate_per_layer = rate_per_layer_batch
        else:
            rate_per_layer = [np.append(rate_per_layer[i], rate_per_layer_batch[i]) for i in range(len(rate_per_layer))]
            
        if distortion_up_to_layer is None:
            distortion_up_to_layer = distortion_up_to_layer_batch
        else:
            distortion_up_to_layer = [np.append(distortion_up_to_layer[i], distortion_up_to_layer_batch[i]) for i in range(len(distortion_up_to_layer))]
            
        if return_L2_dist:
            if L2_dist_up_to_layer is None:
                L2_dist_up_to_layer = L2_dist_up_to_layer_batch
            else:
                L2_dist_up_to_layer = [np.append(L2_dist_up_to_layer[i], L2_dist_up_to_layer_batch[i]) for i in range(len(L2_dist_up_to_layer))]
        
        if return_samples:
            if samples is None:
                samples = samples_batch
            else:
                samples = [np.concatenate((samples[i], samples_batch[i]), axis=0) for i in range(len(samples))]
        
    RD = {'num_latent': num_latent_per_layer, 'rate': rate_per_layer, 'distortion': distortion_up_to_layer}
    if return_L2_dist:
        RD['L2_distortion'] = L2_dist_up_to_layer
    if return_samples:
        RD['samples'] = samples
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


def get_image_name(dataset_name, index, dataset_type='test', seed=None, layer=None, temperature=1, extension='png'):
    if layer is None:
        return f"{dataset_name}_{dataset_type}_{index}.{extension}"
    else:
        transformation_string = get_transformation_string(seed=seed, layer=layer, temperature=temperature)
        return f"{dataset_name}_{transformation_string}_{dataset_type}_{index}.{extension}"
    
    
def get_transformation_string(seed=None, layer=None, temperature=None):
    if temperature is None:
        temperature = 1
    assert seed is not None
    assert layer is not None
    temperature_format = str(temperature).replace(".", "")
    return f"{layer}_{temperature_format}_{seed}"


def save_orig_image(data, index, folder, dataset_name, name_index=None, dataset_type='test', extension='png'):
    if name_index is None:
        name_index = index
    image_name = get_image_name(dataset_name, name_index, dataset_type=dataset_type, extension=extension)
    path = f"{folder}/{image_name}"
    image = data[index][0]
    save_image(image.permute([2,0,1]).type(torch.float), path, normalize=True)
    return path


def save_transformed_image(samples, index, name_index, folder, dataset_name, layer, dataset_type='test', extension='png', seed=42, t=None):
    image_name = get_image_name(dataset_name, name_index, dataset_type=dataset_type, seed=seed, layer=layer, temperature=t, extension=extension)
    path = f"{folder}/{image_name}"
    image = samples[layer][index]
    save_image(torch.from_numpy(image).permute([2,0,1]).type(torch.float), path, normalize=True)
    print(path)
    
    
def save_orig_images(data, idx_list, folder, dataset_name, dataset_type='test', extension='png'):
    for index in idx_list:
        save_orig_image(data, index, folder, dataset_name, dataset_type='test', extension=extension)
        
        
def save_first_transformed_images(inputs, idx_list, folder, dataset_name, dataset_type='test', extension='png'):
    for index, name_index in enumerate(idx_list):
        save_orig_image(inputs, index, folder, dataset_name, name_index=name_index ,dataset_type='test', extension=extension)
        
        
def save_transformed_images(samples, idx_list, folder, dataset_name, layers, dataset_type='test', extension='png', seed=42, t=None):
    for index, name_index in enumerate(idx_list):
        for layer in layers:
            save_transformed_image(samples, index, name_index, folder, dataset_name, layer, dataset_type=dataset_type, extension=extension, seed=seed, t=t)
            
            
def transform_images(vae, data, preprocess_fn, idx_list, folder, dataset_name, layers, batch_size=4, seed=42, t=None, dataset_type='test', extension='png'):
    save_orig_images(data, idx_list, folder, dataset_name, dataset_type=dataset_type, extension=extension)
    torch.manual_seed(seed)
    samples = RD_points(vae, data, preprocess_fn, batch_size=batch_size, idx_list=idx_list, return_samples=True, t=t)['samples']
    save_transformed_images(samples, idx_list, folder, dataset_name, layers, dataset_type=dataset_type, extension=extension, seed=seed, t=t)
    
    
def output_to_input(outputs, layer):
    return TensorDataset(torch.stack([torch.from_numpy(output) for output in outputs[layer]], dim=0))


def transform_images2(vae, data, preprocess_fn, idx_list, folder, dataset_name, first_layer, layers, batch_size=4, seed=42, first_t=None,
                      t=None, dataset_type='test', extension='png'):
    print('Transforming original dataset')
    first_transformed_samples = RD_points(vae, data, preprocess_fn, batch_size=batch_size, idx_list=idx_list, return_samples=True, t=first_t)['samples']
    print('Original dataset transformed')
    first_transformation_string = get_transformation_string(seed=seed, layer=first_layer, temperature=first_t)
    transformed_dataset_name = f"{dataset_name}_{first_transformation_string}"
    inputs = output_to_input(first_transformed_samples, first_layer)
    save_first_transformed_images(inputs, idx_list, folder, transformed_dataset_name, dataset_type=dataset_type, extension=extension)
    print('Transorming transformed images')
    samples = RD_points(vae, inputs, preprocess_fn, batch_size=batch_size, seed=seed, return_samples=True, t=t)['samples']
    print('Transformed images transformed again')
    save_transformed_images(samples, idx_list, folder, transformed_dataset_name, layers, dataset_type=dataset_type, extension=extension, seed=seed, t=t)
    
    
