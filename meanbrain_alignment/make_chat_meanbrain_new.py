"""
Make Meanbrain from anatomical scans

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""
#%%
import os
import numpy as np
import ants
import matplotlib.pyplot as plt
import time
import datetime
from tqdm import tqdm, trange
import glob
from joblib import Parallel, delayed

from glom_pop import dataio
from panglom_suite import plotting

base_dir = dataio.get_config_file()['base_dir']

meanbrain_tag = '20251013M'
brain_dir = os.path.join(base_dir, 'mean_brain', meanbrain_tag)



# %%
def build_percentile_mask(img, p_lo=1, p_hi=98, radius=1, show_mask=False):
    arr = img.clone().numpy()
    non0 = arr[arr > 0]
    lo, hi = np.percentile(non0, (p_lo, p_hi))
    mask = ants.threshold_image(img, low_thresh=lo, high_thresh=hi, inval=1, outval=0)
    mask = ants.iMath(mask,   "MD", radius)           # 1‑voxel morphological *dilation*
    mask = ants.iMath(mask,   "ME", radius)           # followed by *erosion*  → closing
    # mask = ants.iMath(mask, "GetLargestComponent")

    if show_mask:
        ants.plot(img, overlay=mask, axis=2, title='Mask', cmap='Reds', overlay_cmap='Grays', overlay_alpha=0.5)

    return mask

def preload_brain_images(brain_directory, do_bias_correction=False, normalize=False, mask_percentiles=(1, 99), show_mask=False):
    """
    Preload all anatomical brain images from the directory.
    
    Args:
        brain_directory: Directory containing anatomical ANTs images to register
        do_bias_correction: Whether to apply bias field correction to the image
        normalize: Whether to normalize the image intensities
    
    Returns:
        dict: Dictionary mapping file paths to tuples of (individual_red, individual_green) channels
        list: Sorted list of file paths
    """
    t0 = time.time()
    brain_images = {}
    file_paths = sorted(glob.glob(os.path.join(brain_directory, '*_anatomical.nii.gz')))
    brain_names = [os.path.basename(fp).split('.nii')[0] for fp in file_paths]
    
    for fp, brain_key in zip(file_paths, brain_names):
        print(f'Loading {brain_key}...')
        brain_image = ants.image_read(fp)
        individual_red = ants.split_channels(brain_image)[0]
        individual_green = ants.split_channels(brain_image)[1]
        
        if (do_bias_correction or normalize) and mask_percentiles is not None:
            # Compute mask based on the percentiles
            mask = build_percentile_mask(individual_red, p_lo=mask_percentiles[0], p_hi=mask_percentiles[1], radius=1, show_mask=show_mask)
        else:
            mask = None

        # Apply bias correction if requested
        if do_bias_correction:
            individual_red = ants.n4_bias_field_correction(individual_red, mask=mask)
            individual_green = ants.n4_bias_field_correction(individual_green, mask=mask)
        
        # Normalize the image intensities if requested
        if normalize:
            individual_red = ants.iMath(individual_red, 'Normalize') # the mask was left out by accident for 20250510
            individual_green = ants.iMath(individual_green, 'Normalize')

        # Store the preprocessed images in the dictionary
        brain_images[brain_key] = (individual_red, individual_green)
    
    print(f'Preloaded {len(file_paths)} brain images ({time.time()-t0:.2f} sec)')
    return brain_images, file_paths

def show_brain(brain, stride=8, title=None):
    """Quick display z slices of 2 channel brain."""
    num_slices = int(brain.shape[2]/stride)
    fh, ax = plt.subplots(2, num_slices, figsize=(num_slices*3, 4))
    [x.set_axis_off() for x in ax.ravel()]
    for z_ind in range(num_slices):
        ax[0, z_ind].imshow(ants.split_channels(brain)[0][:, :, z_ind*stride].numpy().T, cmap='Reds')
        ax[1, z_ind].imshow(ants.split_channels(brain)[1][:, :, z_ind*stride].numpy().T, cmap='Greens')
    if title is not None:
        fh.suptitle(title)

# %% Preload all brain images
brains, file_paths = preload_brain_images(brain_dir, do_bias_correction=False, normalize=True, mask_percentiles=(5, 95), show_mask=True)

#%%
# Show all brains
for brain_name, (red, green) in brains.items():
    print(brain_name)
    show_brain(ants.merge_channels([red, green]), stride=8, title=brain_name)
    plt.show()
    plt.close()


# %% SEED BRAIN CANDIDATES

seed_candidates = ['TSeries-20241231-001_anatomical',
                    'TSeries-20250107-001_anatomical', # 20250127 meanbrain
                    'TSeries-20250111-009_anatomical',
                    'TSeries-20250118-001_anatomical',]

_=[show_brain(ants.merge_channels(brains[cand_name]), stride=8, title=cand_name) for cand_name in seed_candidates]

# %%
# Choose a seed brain

seed_name = seed_candidates[1] # Chosen

seed_red, seed_green = brains[seed_name]
seed_brain = ants.merge_channels([seed_red, seed_green])
_=show_brain(seed_brain, stride=8, title='seed brain: {}'.format(seed_name))



#%%
def warp_one_brain(mov_red, mov_green, tpl_red, tpl_green, weight=(1.0,0.25), mask_percentiles=(5, 95), show_mask=False, brain_name=None):

    if brain_name is not None:
        print(f'  Registering brain {brain_name}...')

    mov_mask = build_percentile_mask(mov_red, p_lo=mask_percentiles[0], p_hi=mask_percentiles[1], radius=1, show_mask=show_mask)
    tpl_mask  = build_percentile_mask(tpl_red, p_lo=mask_percentiles[0], p_hi=mask_percentiles[1], radius=1, show_mask=show_mask)

    print('Percentile masks built.')

    # ---------- 1. Rigid+Affine+coarse SyN ----------
    print('Registration: Rigid+Affine+SyN...')
    tx1 = ants.registration(
            fixed=tpl_red,  moving=mov_red,
            fixed_image_multi=tpl_green, # I don't think this is real
            moving_image_multi=mov_green,
            mask=tpl_mask, moving_mask=mov_mask,
            type_of_transform='SyN',          # Rigid+Affine+SyN inside
            grad_step=0.10, flow_sigma=6, total_sigma=0,   # "radius ≈ 6 vox"
            syn_metric='CC', syn_sampling=4,
            multivariate_exponent=weight,
            reg_iterations=(50,50,20),
            shrink_factors=(4,2,1),
            smoothing_sigmas=(2,1,0),
            histogram_matching=False)

    # ---------- 2. fine SyN‑only (no new affine) ----------
    print('Registration: fine SyN only...')
    tx2 = ants.registration(
            fixed=tpl_red,  moving=mov_red,
            fixed_image_multi=tpl_green,
            moving_image_multi=mov_green,
            mask=tpl_mask, moving_mask=mov_mask,
            initial_transform=tx1['fwdtransforms'],
            type_of_transform='SyNOnly',      # just a second SyN
            grad_step=0.05, flow_sigma=3, total_sigma=0,   # "radius ≈ 3 vox"
            syn_metric='CC', syn_sampling=4,
            multivariate_exponent=weight,
            reg_iterations=(20,10),           # one or two finest levels
            shrink_factors=(1,1),
            smoothing_sigmas=(0,0),
            histogram_matching=False)

    # 3  Apply the transform to BOTH channels
    print('Applying transforms to both channels...')
    wr_red   = ants.apply_transforms(fixed=tpl_red,  
                                      moving=mov_red,
                                      transformlist=tx2['fwdtransforms'], 
                                      interpolator='bSpline', 
                                      defaultvalue=0)
    wr_green = ants.apply_transforms(fixed=tpl_green,
                                      moving=mov_green,
                                      transformlist=tx2['fwdtransforms'],
                                      interpolator='bSpline',
                                      defaultvalue=0)
    
    if brain_name is not None:
        print(f'  Done with brain {brain_name}')

    return wr_red, wr_green

# %%
# Parallel version
K = 2
n_jobs = 8
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "3"

template_red, template_green = seed_red, seed_green
for k in trange(K):
    print(f'-- Iteration {k+1}/{K} --')

    # results = Parallel(n_jobs=n_jobs, backend='loky')(
    #                 delayed(warp_one_brain)(mov_red, mov_green, template_red, template_green, mask_percentiles=(5,95), brain_name=brain_name)
    #             for brain_name, (mov_red, mov_green) in brains.items())
    
    # Non-parallel version
    results = []
    for brain_name, (mov_red, mov_green) in brains.items():
        wr_red, wr_green = warp_one_brain(mov_red, mov_green, template_red, template_green, mask_percentiles=(5,95), brain_name=brain_name)
        results.append((wr_red, wr_green))

    print(f'  Done registering all brains, now computing new template...')

    red_stack   = np.stack([r[0].numpy() for r in results])
    green_stack = np.stack([r[1].numpy() for r in results])

    # 2.5  Compute new template  (mean or median)
    # Convert all 0s to NaNs (non-brain domain)
    red_stack[red_stack==0]     = np.nan
    green_stack[green_stack==0] = np.nan

    # Compute mean across all brains
    template_red   = np.nanmean(red_stack, axis=0)
    template_green = np.nanmean(green_stack, axis=0)

    # convert NaNs to 0
    template_red[np.isnan(template_red)] = 0
    template_green[np.isnan(template_green)] = 0

    template_red   = ants.from_numpy(template_red,
                                spacing=seed_red.spacing,
                                origin=seed_red.origin,
                                direction=seed_red.direction)
    template_green = ants.from_numpy(template_green,
                                spacing=seed_green.spacing,
                                origin=seed_green.origin,
                                direction=seed_green.direction)
    
    # 2.6  Optional: mild smoothing & intensity normalisation
    template_red   = ants.smooth_image(template_red,   sigma=0.5)
    template_green = ants.smooth_image(template_green, sigma=0.5)

    template_red   = ants.iMath(template_red,  'Normalize')
    template_green = ants.iMath(template_green,'Normalize')

    # Show the new template
    show_brain(ants.merge_channels([template_red, template_green]), stride=8, title=f'Iteration {k+1}')
    plt.show()
    plt.close()

    # Save the intermediate meanbrain
    template_brain = ants.merge_channels([template_red, template_green])
    save_path = os.path.join(brain_dir, f'chat_meanbrain_{meanbrain_tag}_{k+1}_iters.nii.gz')
    ants.image_write(template_brain, save_path)

    print(f'-- Iteration {k+1}/{K} finished --')



# %% Save final meanbrain
save_path = os.path.join(brain_dir, f'chat_meanbrain_{meanbrain_tag}.nii.gz')
ants.image_write(template_brain, save_path)

# save individual channels
ants.image_write(ants.split_channels(template_brain)[0], os.path.join(brain_dir, f'chat_meanbrain_{meanbrain_tag}_ch1.nii.gz'))
ants.image_write(ants.split_channels(template_brain)[1], os.path.join(brain_dir, f'chat_meanbrain_{meanbrain_tag}_ch2.nii.gz'))


# %%
# Non-parallel version

# K = 2

# template_red, template_green = seed_red, seed_green
# for k in range(K):
#     warped_red_sum   = None       # running sum for the mean
#     warped_green_sum = None
#     voxel_count      = None       # count of non‑zero voxels (for median/trim)

#     n_brains = len(brains)

#     print(f'-- Iteration {k+1}/{K} --')
#     for i, (brain_name, (mov_red, mov_green)) in enumerate(tqdm(brains.items())):
#         print(f'  Registering brain {i+1}/{n_brains} ({brain_name})')
#         # 2.2  Register *one* channel pair but GIVE BOTH channels
#         wr_red, wr_green = warp_one_brain(mov_red, mov_green,
#                                             template_red, template_green,
#                                             weight=(1.0,0.25), 
#                                             mask_percentiles=(5, 95), show_mask=False)

#         # 2.4  Accumulate running mean  (memory‑safe: one image at a time)
#         if warped_red_sum is None:
#             warped_red_sum   = np.zeros_like(wr_red)
#             warped_green_sum = np.zeros_like(wr_green)
#             voxel_count      = np.zeros_like(wr_red, dtype=int)

#         not_nan_mask = ~np.isnan(wr_red)
#         warped_red_sum[not_nan_mask]   += wr_red[not_nan_mask]
#         warped_green_sum[not_nan_mask] += wr_green[not_nan_mask]
#         voxel_count                    += (~np.isnan)

#     # 2.5  Compute new template  (mean or median)
#     template_red   = warped_red_sum   / voxel_count
#     template_green = warped_green_sum / voxel_count

#     # 2.6  Optional: mild smoothing & intensity normalisation
#     template_red   = ants.smooth_image(template_red,   sigma=0.5)
#     template_green = ants.smooth_image(template_green, sigma=0.5)

#     template_red   = ants.iMath(template_red,  'Normalize')
#     template_green = ants.iMath(template_green,'Normalize')

#     # Show the new template
#     show_brain(ants.merge_channels([template_red, template_green]), stride=8, title=f'Iteration {k+1}')
#     plt.show()
#     plt.close()

#     print(f'-- Iteration {k+1}/{K} finished --')

# template_brain = ants.merge_channels([template_red, template_green])

# %%
