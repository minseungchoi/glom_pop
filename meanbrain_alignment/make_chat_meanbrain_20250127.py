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

meanbrain_tag = '20250127'
brain_dir = os.path.join(base_dir, 'mean_brain', meanbrain_tag)


# %% REFERENCE BRAIN
reference_filename = 'TSeries-20250107-001_anatomical.nii'

# 2-channel xyz
reference_brain = ants.image_read(os.path.join(brain_dir, reference_filename))
spacing = reference_brain.spacing

print('Brain spacing is {}'.format(spacing))

fh, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(ants.split_channels(reference_brain)[0].max(axis=2).T, cmap='Reds')
ax[1].imshow(ants.split_channels(reference_brain)[1].max(axis=2).T, cmap='Greens')

# %%


def computeMeanbrain(brain_directory,
                     reference_brain,
                     type_of_transform='SyN',
                     do_bias_correction=True,
                     do_smoothing=True,
                     plot_alignment_multipanel=True):
    """
    Generate a meanbrain from a list of anatomical scans.

    :brain_directory: contains anatomical ANTs images to register, fns end in '_anatomical.nii'
    :reference_brain: two-channel ANTs image to register each brain to
    :type_of_transform: for ants.registration()
    :do_bias_correction: [bool]
    :do_smoothing: [bool]
    """
    t0 = time.time()
    corrected_red = []
    corrected_green = []
    file_paths = glob.glob(os.path.join(brain_directory, '*_anatomical.nii'))

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
        individual_red = ants.split_channels(ants.image_read(fp))[0]
        individual_green = ants.split_channels(ants.image_read(fp))[1]

        # Temporary copy to use to compute registration
        moving_red = individual_red.clone()
        fixed_red, fixed_green = ants.split_channels(reference_brain)
        if do_bias_correction:
            moving_red = ants.n3_bias_field_correction(moving_red)
        if do_smoothing:
            fixed_red = ants.smooth_image(fixed_red, sigma=[1.0, 1.0, 0.0], sigma_in_physical_coordinates=False)
            moving_red = ants.smooth_image(moving_red, sigma=[1.0, 1.0, 0.0], sigma_in_physical_coordinates=False)

        # # print male_meanbrain and save_meanbrain's data type
        # print(f"fixed_red dtype: {fixed_red.dtype}")
        # print(f"moving_red dtype: {moving_red.dtype}")

        # # print shapes of both meanbrains
        # print(f"fixed_red shape: {fixed_red.shape}")
        # print(f"moving_red shape: {moving_red.shape}")

        # # print number of NaNs in both meanbrains
        # print(f"Number of NaNs in fixed_red: {np.sum(np.isnan(fixed_red.numpy()))}")
        # print(f"Number of NaNs in moving_red: {np.sum(np.isnan(moving_red.numpy()))}")

        reg = ants.registration(fixed=fixed_red,
                                moving=moving_red,
                                type_of_transform=type_of_transform,
                                grad_step=0.2,
                                flow_sigma=3,
                                total_sigma=0,
                                verbose=False)
        
        red_reg = ants.apply_transforms(fixed=fixed_red,
                                        moving=individual_red,
                                        transformlist=reg['fwdtransforms'],
                                        interpolator='nearestNeighbor',
                                        defaultvalue=0,
                                        verbose=False)

        green_reg = ants.apply_transforms(fixed=fixed_green,
                                          moving=individual_green,
                                          transformlist=reg['fwdtransforms'],
                                          interpolator='nearestNeighbor',
                                          defaultvalue=0)

        red_reg = red_reg.numpy().astype('float')
        red_reg[red_reg == 0] = np.nan

        green_reg = green_reg.numpy().astype('float')
        green_reg[green_reg == 0] = np.nan

        corrected_red.append(red_reg)
        corrected_green.append(green_reg)

        if plot_alignment_multipanel:
            for z_ind in range(num_slices):
                ax[z_ind, fp_ind+1].imshow(red_reg[:, :, z_ind*stride].T, cmap='Greens')
            ax[0, fp_ind+1].set_title(fp_ind)

        print('Done with brain {}:{} ({:.2f} sec)'.format(fp_ind, fp.split('/')[-1], time.time()-t0_fp))

    meanbrain_red = np.nanmean(np.stack(corrected_red, -1), axis=-1)
    meanbrain_green = np.nanmean(np.stack(corrected_green, -1), axis=-1)

    # occluded back to 0
    meanbrain_red[np.isnan(meanbrain_red)] = 0
    meanbrain_green[np.isnan(meanbrain_green)] = 0

    # Convert to ANTs image and merge channels
    meanbrain_red = ants.from_numpy(meanbrain_red, spacing=reference_brain.spacing)
    meanbrain_green = ants.from_numpy(meanbrain_green, spacing=reference_brain.spacing)
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


# # %% Compute meanbrain 1:
# # Align smoothed brains, to get things roughly aligned
# meanbrain_0 = computeMeanbrain(brain_directory=brain_dir,
#                                reference_brain=reference_brain,
#                                type_of_transform='Affine',
#                                do_smoothing=True)
# showBrain(meanbrain_0, stride=8)

# %% Compute meanbrain 1:
# Align smoothed brains, to get things roughly aligned
meanbrain_1 = computeMeanbrain(brain_directory=brain_dir,
                               reference_brain=reference_brain, # meanbrain_0,
                               type_of_transform='SyN',
                               do_bias_correction=True,
                               do_smoothing=True)

showBrain(meanbrain_1, stride=8)

# %% Compute meanbrain 2:
# No smoothing, elastic alignment
meanbrain_2 = computeMeanbrain(brain_directory=brain_dir,
                               reference_brain=meanbrain_1,
                               type_of_transform='ElasticSyN',
                               do_bias_correction=True,
                               do_smoothing=False)

showBrain(meanbrain_2, stride=8)
# %% Compute final meanbrain:
meanbrain = computeMeanbrain(brain_directory=brain_dir,
                             reference_brain=meanbrain_2,
                             type_of_transform='ElasticSyN',
                             do_bias_correction=True,
                             do_smoothing=False)

showBrain(meanbrain, stride=8)

# %% Save final meanbrain
save_path = os.path.join(brain_dir, 'chat_meanbrain_{}.nii'.format(meanbrain_tag))
ants.image_write(meanbrain, save_path)

# save individual channels
ants.image_write(ants.split_channels(meanbrain)[0], os.path.join(brain_dir, 'chat_meanbrain_{}_ch1.nii'.format(meanbrain_tag)))
ants.image_write(ants.split_channels(meanbrain)[1], os.path.join(brain_dir, 'chat_meanbrain_{}_ch2.nii'.format(meanbrain_tag)))

# %%
