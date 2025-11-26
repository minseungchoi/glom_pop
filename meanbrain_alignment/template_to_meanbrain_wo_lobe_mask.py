"""
Align template brain & glom map to meanbrain


maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""
#%%
import os
import numpy as np
import nibabel as nib
import ants
import matplotlib.pyplot as plt
import pandas as pd
import colorcet as cc
from glom_pop import dataio

base_dir = dataio.get_config_file()['base_dir']
meanbrain_tag = '20241028_tall'
meanbrain_dir = os.path.join(base_dir, 'mean_brain', meanbrain_tag)

meanbrain_fn = 'chat_meanbrain_{}.nii'.format(meanbrain_tag)

# %% LOAD
# Load meanbrain
meanbrain = ants.image_read(os.path.join(meanbrain_dir, meanbrain_fn))
[meanbrain_red, meanbrain_green] = ants.split_channels(meanbrain)
meanbrain_1ch = meanbrain_red

# Load template & glom mask
template = ants.image_read(os.path.join(base_dir, 'template_brain', 'JRC2018_FEMALE_38um_iso_32f_z_inverted_resliced.nii'))
glom_mask = ants.image_read(os.path.join(base_dir, 'template_brain', 'glom_mask_4_r_z_inverted_resliced.nii'))

# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))

# %%
# Print shapes of meanbrain and template
print('Meanbrain shape:', meanbrain_red.shape)
print('Template shape:', template.shape)
n_meanbrain_z = meanbrain_red.shape[2]
n_template_z = template.shape[2]

# %% SHOW TEMPLATE BRAIN
n_disp = n_template_z//3
n_cols = 5
n_rows = n_disp // n_cols + 1
fh, ax = plt.subplots(n_rows, n_cols, figsize=(16, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax.ravel()]
for z in range(n_template_z//3):
    ax[z].imshow(template[:, :, z*3].numpy().T)

# %% SHOW MEAN BRAIN - GREEN. Align on this.
n_disp = n_meanbrain_z//3
n_cols = 5
n_rows = n_disp // n_cols + 1

fh, ax = plt.subplots(n_rows, n_cols, figsize=(16, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax.ravel()]
for z in range(n_meanbrain_z//3):
    ax[z].imshow(meanbrain_1ch[:, :, z*3].numpy().T)

# %% COMPUTE ALIGNMENT IN TWO STAGES
# 1) Mannual affine + SyN
# 2) SyN
# Breaking it up helps troubleshoot. Make sure affine is close first, then mess with SyN warp as needed

# (1) Affine alignment. ~30 sec
reg_aff = ants.registration(fixed=ants.n4_bias_field_correction(meanbrain_1ch),
                            moving=ants.n4_bias_field_correction(template),
                            type_of_transform='Affine',
                            mask=None,  # mask includes only gloms, excludes lobe
                            flow_sigma=6,
                            total_sigma=0,
                            random_seed=0,
                            aff_sampling=32,
                            grad_step=0.05,
                            reg_iterations=[250, 100, 50])

# # (1) Initialize with manual affine transform + SyN

# transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_template')
# manual_affine_path = os.path.join(transform_dir, 'affine_by_hand_20240824_fwd.txt') # generated with ITK snap

# reg_syn = ants.registration(fixed=ants.n4_bias_field_correction(meanbrain_1ch),
#                             moving=ants.n4_bias_field_correction(template),
#                             type_of_transform='SyNOnly',
#                             # type_of_transform='ElasticSyN',
#                             mask=lobe_mask,  # mask includes only gloms, excludes lobe
#                             random_seed=0,
#                             initial_transform=[manual_affine_path]
# )

# Show overlay
disp_meanbrain = meanbrain_1ch[30:230, 10:175, :].numpy()
disp_male = reg_aff['warpedmovout'][30:230, 10:175, :].numpy()

male_tmp = np.ma.masked_where(disp_male == 0, disp_male)  # mask at 0

slices = [10, 20, 30, 40, 50, 60, 65]
fh, ax = plt.subplots(len(slices), 3, figsize=(12, len(slices)*3))
[x.set_axis_off() for x in ax.ravel()]
for s_ind, slice in enumerate(slices):
    ax[s_ind, 0].imshow(disp_meanbrain[:, :, slice].T, alpha=0.5, cmap='Purples', vmax=np.quantile(disp_meanbrain, 0.99))

    ax[s_ind, 1].imshow(male_tmp[:, :, slice].T, alpha=0.5, cmap='Greens', interpolation='nearest')

    ax[s_ind, 2].imshow(disp_meanbrain[:, :, slice].T, alpha=0.5, cmap='Purples', vmax=np.quantile(disp_meanbrain, 0.99))
    ax[s_ind, 2].imshow(male_tmp[:, :, slice].T, alpha=0.5, cmap='Greens', interpolation='nearest')

# %%
# (2) Then compute a small SyN warp on top of this. MI metric. ~60 sec
# output reg_syn transform contains both this new syn warp and the initial affine transform
reg_syn = ants.registration(fixed=ants.n4_bias_field_correction(meanbrain_1ch),
                            moving=ants.n4_bias_field_correction(template),
                            type_of_transform='SyNOnly',
                            mask=None,  # mask includes only gloms, excludes lobe
                            initial_transform=reg_aff['fwdtransforms'][0],
                            random_seed=0)

# APPLY ALIGNMENT TO MASK & TEMPLATE
template_warped = ants.apply_transforms(fixed=meanbrain_1ch,
                                        moving=template,
                                        transformlist=reg_aff['fwdtransforms'],
                                        interpolator='nearestNeighbor')

glom_mask_warped = ants.apply_transforms(fixed=meanbrain_1ch,
                                         moving=glom_mask,
                                         transformlist=reg_aff['fwdtransforms'],
                                         interpolator='genericLabel')

# Save transform
transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_template')
os.makedirs(transform_dir, exist_ok=True)
dataio.save_transforms(reg_syn, transform_dir)

# Save transformed images
ants.image_write(template_warped, os.path.join(transform_dir, 'JRC2018_resliced_reg2meanbrain.nii'))
ants.image_write(glom_mask_warped, os.path.join(transform_dir, 'glom_mask_resliced_reg2meanbrain.nii'))

# Save overlay multichannel image
merged = ants.merge_channels([meanbrain_red, meanbrain_1ch, template_warped, glom_mask_warped])
save_path = os.path.join(transform_dir, 'overlay_meanbrain.nii')
ants.image_write(merged, save_path)

# %% CHECK OVERALL ALIGNMENT
mi_orig = ants.image_mutual_information(meanbrain_1ch, template)
mi_aff = ants.image_mutual_information(meanbrain_1ch, reg_aff['warpedmovout'])
mi_syn = ants.image_mutual_information(meanbrain_1ch, template_warped)

fh, ax = plt.subplots(1, 3, figsize=(16, 6))
ax[0].imshow(meanbrain_1ch.max(axis=2).T, vmax=np.quantile(meanbrain_1ch.max(axis=2).ravel(), 0.9))
ax[0].set_title('Original, MI = {:.2f}'.format(mi_orig))

ax[1].imshow(reg_aff['warpedmovout'].max(axis=2).T)
ax[1].set_title('Affine, MI = {:.2f}'.format(mi_aff))

ax[2].imshow(template_warped.max(axis=2).T)
ax[2].set_title('+ SyN Warp, MI = {:.2f}'.format(mi_syn))

for x in ax.ravel():
    x.grid(which='major', axis='both', linestyle='-', color='r')

# %% CHECK ALIGNMENT IN SLICES
disp_meanbrain = meanbrain_1ch[30:230, 10:175, :].numpy()
disp_glom = glom_mask_warped[30:230, 10:175, :].numpy()

glom_tmp = np.ma.masked_where(disp_glom == 0, disp_glom)  # mask at 0

slices = [10, 20, 30, 40, 50, 60, 65]
fh, ax = plt.subplots(len(slices), 3, figsize=(12, len(slices)*3))
[x.set_axis_off() for x in ax.ravel()]
for s_ind, slice in enumerate(slices):
    ax[s_ind, 0].imshow(disp_meanbrain[:, :, slice].T, alpha=0.5, cmap='Greens', vmax=np.quantile(disp_meanbrain, 0.99))

    ax[s_ind, 1].imshow(glom_tmp[:, :, slice].T, alpha=0.5, cmap=cc.cm.glasbey, vmin=1, vmax=45, interpolation='nearest')

    ax[s_ind, 2].imshow(disp_meanbrain[:, :, slice].T, alpha=0.5, cmap='Greens', vmax=np.quantile(disp_meanbrain, 0.99))
    ax[s_ind, 2].imshow(glom_tmp[:, :, slice].T, alpha=0.5, cmap=cc.cm.glasbey, vmin=1, vmax=45, interpolation='nearest')


# %% Re-apply saved transform to remade glom map
transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_template')

# Load brains
meanbrain = ants.image_read(os.path.join(base_dir, 'mean_brain', meanbrain_fn))
[meanbrain_red, meanbrain_green] = ants.split_channels(meanbrain)

# Load glom mask
glom_mask = ants.image_read(os.path.join(base_dir, 'template_brain', 'vpn_glom_mask_closed.nii'))

# load pre-registered (above) template
template_warped = ants.image_read(os.path.join(transform_dir, 'JRC2018_reg2meanbrain.nii'))

# Load transform
transform_list = dataio.get_transform_list(transform_dir, direction='forward')

glom_mask_warped = ants.apply_transforms(fixed=meanbrain_green,
                                         moving=glom_mask,
                                         transformlist=transform_list,
                                         interpolator='genericLabel')

# Save transformed images
ants.image_write(glom_mask_warped, os.path.join(transform_dir, 'glom_mask_reg2meanbrain.nii'))

# Save overlay multichannel image
merged = ants.merge_channels([meanbrain_red, meanbrain_green, template_warped, glom_mask_warped])
save_path = os.path.join(transform_dir, 'overlay_meanbrain.nii')
ants.image_write(merged, save_path)

# %%
