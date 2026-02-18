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
import glob

from glom_pop import dataio
from panglom_suite import plotting

base_dir = dataio.get_config_file()['base_dir']

meanbrain_tag = '20260217_rewrite'
brain_dir = os.path.join(base_dir, 'mean_brain', meanbrain_tag)


# %% REFERENCE BRAIN CANDIDATES

def show_reference_brain(reference_fn):
    reference_brain = ants.image_read(os.path.join(brain_dir, reference_fn))
    fh, ax = plt.subplots(2, 2, figsize=(7, 4))
    ax[0,0].imshow(ants.split_channels(reference_brain)[0].mean(axis=2).T, cmap='Reds')
    ax[0,1].imshow(ants.split_channels(reference_brain)[1].mean(axis=2).T, cmap='Greens')
    ax[1,0].imshow(ants.split_channels(reference_brain)[0].max(axis=2).T, cmap='Reds')
    ax[1,1].imshow(ants.split_channels(reference_brain)[1].max(axis=2).T, cmap='Greens')
    ax[0,0].set_ylabel('Mean')
    ax[1,0].set_ylabel('Max')
    ax[0,0].set_title('tdTomato')
    ax[0,1].set_title('syt1GCaMP6f')
    fh.suptitle(reference_fn)

reference_fns = ['TSeries-20241231-001_anatomical.nii.gz',
                  'TSeries-20250107-001_anatomical.nii.gz', # 20250127 meanbrain
                  'TSeries-20250111-009_anatomical.nii.gz',
                  'TSeries-20250118-001_anatomical.nii.gz',]

_=[show_reference_brain(fn) for fn in reference_fns]

# %%

chosen_reference_fn = reference_fns[0]

_=show_reference_brain(chosen_reference_fn)
reference_brain = ants.image_read(os.path.join(brain_dir, chosen_reference_fn))
spacing = reference_brain.spacing
print('Brain spacing is {}'.format(spacing))


# %%
def preload_brain_images(brain_directory, do_bias_correction=False):
    """
    Preload all anatomical brain images from the directory.
    
    Args:
        brain_directory: Directory containing anatomical ANTs images to register
        do_bias_correction: Whether to apply bias field correction to the image
    
    Returns:
        dict: Dictionary mapping file paths to tuples of (individual_red, individual_green) channels
        list: Sorted list of file paths
    """
    t0 = time.time()
    brain_images = {}
    file_paths = sorted(glob.glob(os.path.join(brain_directory, '*_anatomical.nii.gz')))
    
    for fp in file_paths:
        brain_image = ants.image_read(fp)
        individual_red = ants.split_channels(brain_image)[0]
        individual_green = ants.split_channels(brain_image)[1]
        
        # Apply bias correction if requested
        if do_bias_correction:
            individual_red = ants.n4_bias_field_correction(individual_red)
            individual_green = ants.n4_bias_field_correction(individual_green)
            
        brain_images[fp] = (individual_red, individual_green)
    
    print(f'Preloaded {len(file_paths)} brain images ({time.time()-t0:.2f} sec)')
    return brain_images, file_paths


def computeMeanbrain(reference_brain,
                     brains,
                     file_paths,
                     type_of_transform='SyN',
                     smooth_reference=False,
                     smooth_moving=False,
                     plot_alignment_multipanel=True, **reg_kwargs):
    """
    Generate a meanbrain from a list of anatomical scans.

    :reference_brain: two-channel ANTs image to register each brain to
    :brains: dictionary mapping file paths to (red_channel, green_channel) tuples
    :file_paths: list of file paths to process (keys in the brains dictionary)
    :type_of_transform: for ants.registration()
    :smooth_reference: whether to smooth the reference brain
    :smooth_moving: whether to smooth the moving brain
    :plot_alignment_multipanel: whether to plot the alignment of brains
    """
    t0 = time.time()
    corrected_red = []
    corrected_green = []

    if plot_alignment_multipanel:
        stride = 8
        num_slices = int(reference_brain.shape[2]/stride)
        fh, ax = plt.subplots(num_slices, len(file_paths)+1, figsize=((len(file_paths)+1)*2, num_slices*3.5))
        [plotting.clean_axes(x) for x in ax.ravel()]

        for z_ind in range(num_slices):
            ax[z_ind, 0].imshow(ants.split_channels(reference_brain)[0][:, :, z_ind*stride].numpy().T, cmap='Purples')
        ax[z_ind, 0].set_ylabel('Reference')

    for fp_ind, fp in enumerate(file_paths):
        t0_fp = time.time()
        
        individual_red, individual_green = brains[fp]

        # Temporary copy to use to compute registration
        moving_red = individual_red.clone()
        fixed_red, fixed_green = ants.split_channels(reference_brain)
        
        moving_green = individual_green.clone()

        if smooth_reference:
            fixed_red = ants.smooth_image(fixed_red, sigma=[1.0, 1.0, 0.0], sigma_in_physical_coordinates=False)
            fixed_green = ants.smooth_image(fixed_green, sigma=[1.0, 1.0, 0.0], sigma_in_physical_coordinates=False)
        if smooth_moving:
            moving_red = ants.smooth_image(moving_red, sigma=[1.0, 1.0, 0.0], sigma_in_physical_coordinates=False)
            moving_green = ants.smooth_image(moving_green, sigma=[1.0, 1.0, 0.0], sigma_in_physical_coordinates=False)

        reg_kwargs_local = dict(reg_kwargs)
        # supports_multivariate = (
        #     type_of_transform == 'SyNOnly'
        #     or (isinstance(type_of_transform, str) and type_of_transform.startswith('antsRegistrationSyN'))
        # )
        supports_multivariate = False
        if supports_multivariate:
            reg_kwargs_local.setdefault(
                'multivariate_extras',
                [('CC', fixed_green, moving_green, 0.7, 4)],
            )
        elif 'multivariate_extras' in reg_kwargs_local:
            # multivariate_extras are only valid for SyNOnly / antsRegistrationSyN* transforms.
            reg_kwargs_local.pop('multivariate_extras')

        reg = ants.registration(fixed=fixed_red,
                                moving=moving_red,
                                type_of_transform=type_of_transform,
                                **reg_kwargs_local)
        
        red_reg = ants.apply_transforms(fixed=fixed_red,
                                        moving=individual_red,
                                        transformlist=reg['fwdtransforms'],
                                        interpolator='bSpline',
                                        spline_order=3,
                                        defaultvalue=0,
                                        )

        green_reg = ants.apply_transforms(fixed=fixed_green,
                                          moving=individual_green,
                                          transformlist=reg['fwdtransforms'],
                                          interpolator='bSpline',
                                          spline_order=3,
                                          defaultvalue=0)

        red_reg_np = red_reg.clone().numpy()
        red_reg_np[red_reg_np == 0] = np.nan

        green_reg_np = green_reg.clone().numpy()
        green_reg_np[green_reg_np == 0] = np.nan

        corrected_red.append(red_reg_np)
        corrected_green.append(green_reg_np)

        if plot_alignment_multipanel:
            for z_ind in range(num_slices):
                ax[z_ind, fp_ind+1].imshow(red_reg_np[:, :, z_ind*stride].T, cmap='Greens')
            ax[0, fp_ind+1].set_title(fp_ind)

        print('Done with brain {}:{} ({:.2f} sec)'.format(fp_ind, fp.split('/')[-1], time.time()-t0_fp))

    meanbrain_red = np.nanmean(np.stack(corrected_red, -1), axis=-1)
    meanbrain_green = np.nanmean(np.stack(corrected_green, -1), axis=-1)

    # occluded back to 0
    meanbrain_red[np.isnan(meanbrain_red)] = 0
    meanbrain_green[np.isnan(meanbrain_green)] = 0

    # Convert to ANTs image and merge channels
    meanbrain_red = ants.from_numpy(meanbrain_red, spacing=reference_brain.spacing, origin=reference_brain.origin, direction=reference_brain.direction)
    meanbrain_green = ants.from_numpy(meanbrain_green, spacing=reference_brain.spacing, origin=reference_brain.origin, direction=reference_brain.direction)
    meanbrain = ants.merge_channels([meanbrain_red, meanbrain_green])

    print('Computed meanbrain ({} sec)'.format(time.time()-t0))
    return meanbrain


def showBrain(brain, stride):
    """Quick display z slices of 2 channel brain."""
    num_slices = int(brain.shape[2]/stride)
    fh, ax = plt.subplots(2, num_slices, figsize=(num_slices*3, 4))
    [x.set_axis_off() for x in ax.ravel()]
    for z_ind in range(num_slices):
        ax[0, z_ind].imshow(ants.split_channels(brain)[0][:, :, z_ind*stride].numpy().T, cmap='Reds')
        ax[1, z_ind].imshow(ants.split_channels(brain)[1][:, :, z_ind*stride].numpy().T, cmap='Greens')


# %% Preload all brain images
brains, file_paths = preload_brain_images(brain_dir, do_bias_correction=False)

# %% Compute meanbrain 0:
# Affine smoothed brains, to get things roughly aligned
meanbrain_0 = computeMeanbrain(
    reference_brain=reference_brain,
    brains=brains,
    file_paths=file_paths,
    type_of_transform='Rigid',
    smooth_reference=True,
    smooth_moving=True,
    plot_alignment_multipanel=True
)
showBrain(meanbrain_0, stride=8)

# %% Compute meanbrain 1:
# Affine
meanbrain_1 = computeMeanbrain(
    reference_brain=meanbrain_0,
    brains=brains,
    file_paths=file_paths,
    type_of_transform='Affine',
    smooth_reference=True,
    smooth_moving=True,
    plot_alignment_multipanel=True,
    aff_metric='MI',                # Mutual‑Information
    grad_step=0.1,
    aff_sampling=32,                # 32 histogram bins
    aff_sampling_strategy='Regular',
    reg_iterations=(1000, 500, 250, 0),
    aff_shrink_factors=(8, 4, 2, 1),
    aff_smoothing_sigmas=(3, 2, 1, 0),
    histogram_matching=False,
    verbose=False,
)
showBrain(meanbrain_1, stride=8)

# %% Compute meanbrain 2:
# Elastic SyN
meanbrain_2 = computeMeanbrain(
    reference_brain=meanbrain_1,
    brains=brains,
    file_paths=file_paths,
    type_of_transform='ElasticSyN',
    smooth_reference=True,
    smooth_moving=True,
    plot_alignment_multipanel=True, 
    grad_step=0.20,                 # ElasticSyN[0.2,6,0]
    flow_sigma=6,
    total_sigma=0.75,
    syn_metric='CC',                # Cross‑Correlation
    syn_sampling=4,                 # CC radius
    reg_iterations=(20, 10, 0),
    histogram_matching=False,
)
showBrain(meanbrain_2, stride=8)

# %% Compute meanbrain 3:
# ElasticSyN Fine
meanbrain_3 = computeMeanbrain(
    reference_brain=meanbrain_2,
    brains=brains,
    file_paths=file_paths,
    type_of_transform='ElasticSyN',
    smooth_reference=False,
    smooth_moving=False,
    plot_alignment_multipanel=True, 
    grad_step=0.10,                 # ElasticSyN[0.1,3,0]
    flow_sigma=3,
    total_sigma=0.5,
    syn_metric='CC',                # Cross‑Correlation
    syn_sampling=3,                 # CC radius
    reg_iterations=(30, 15, 0),
    histogram_matching=False,
)
showBrain(meanbrain_3, stride=8)

# %% Compute final meanbrain:
# SyN
meanbrain = computeMeanbrain(
    reference_brain=meanbrain_3,
    brains=brains,
    file_paths=file_paths,
    type_of_transform='SyN',
    smooth_reference=False,
    smooth_moving=False,
    plot_alignment_multipanel=True, 
    grad_step=0.10,                 # SyN[0.1,6,0]
    flow_sigma=3,
    total_sigma=0,
    syn_metric='CC',                # Cross‑Correlation
    syn_sampling=2,                 # CC radius
    reg_iterations=(30, 15, 1),
    histogram_matching=False,
)
showBrain(meanbrain, stride=8)

# %% Save final meanbrain
save_path = os.path.join(brain_dir, 'chat_meanbrain_{}.nii'.format(meanbrain_tag))
ants.image_write(meanbrain, save_path)

# save individual channels
ants.image_write(ants.split_channels(meanbrain)[0], os.path.join(brain_dir, 'chat_meanbrain_{}_ch1.nii'.format(meanbrain_tag)))
ants.image_write(ants.split_channels(meanbrain)[1], os.path.join(brain_dir, 'chat_meanbrain_{}_ch2.nii'.format(meanbrain_tag)))

# %%
