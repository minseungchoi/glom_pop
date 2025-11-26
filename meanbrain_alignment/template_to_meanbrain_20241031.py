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

# Input parameters
meanbrain_tag = '20241028'
meanbrain_tag = '20241028_pad20'
template_fn = 'JRC2018_FEMALE_38um_iso_32f_z_inverted_resliced_20241028_pad20.nii'
glom_mask_fn = 'glom_mask_4_r_z_inverted_resliced_20241028_pad20.nii'
use_lobe_mask = False
use_red_meanbrain = True
disp_z_indices = [10, 20, 30, 40, 50, 60, 65]
disp_meanbrain_xy_slices = (slice(20, 230), slice(10, 185))
disp_template_xy_slices = (slice(20, 230), slice(10, 185))
random_seed = 0

# Output filenames
template_warped_fn = f'JRC2018_resliced_{meanbrain_tag}_reg2meanbrain_{"red" if use_red_meanbrain else "green"}.nii'
glom_mask_warped_fn = f'glom_mask_resliced_{meanbrain_tag}_reg2meanbrain_{"red" if use_red_meanbrain else "green"}.nii'
overlay_fn = f'overlay_meanbrain_{meanbrain_tag}_{"red" if use_red_meanbrain else "green"}.nii'
reg_fig_fn = f'registration_{meanbrain_tag}_{"red" if use_red_meanbrain else "green"}.png'

# %% FUNCTIONS

def plot_alignment(reg_list, meanbrain, template, glom_mask, disp_z_indices, disp_meanbrain_xy_slices=None, disp_template_xy_slices=None):
    n_cols = 2 + 2*len(reg_list) # original + template + each reg step (template, meanbrain)
    n_rows = len(disp_z_indices)

    fh, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*2.5))
    [x.set_axis_off() for x in ax.ravel()]

    if disp_meanbrain_xy_slices is None:
        disp_meanbrain_xy_slices = (slice(None), slice(None))
    if disp_template_xy_slices is None:
        disp_template_xy_slices = (slice(None), slice(None))
    disp_meanbrain = meanbrain[*disp_meanbrain_xy_slices, :].numpy()
    disp_template = template[*disp_template_xy_slices, :].numpy()

    for zii, z_ind in enumerate(disp_z_indices):
        ax[zii, 0].set_axis_on()
        ax[zii, 0].spines['top'].set_visible(False)
        ax[zii, 0].spines['right'].set_visible(False)
        ax[zii, 0].spines['left'].set_visible(False)
        ax[zii, 0].spines['bottom'].set_visible(False)
        ax[zii, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[zii, 0].set_ylabel(f'{z_ind} \u03bcm')

        ax[zii, 0].imshow(disp_template[:, :, z_ind].T, alpha=0.5, cmap='Greens', vmax=np.quantile(disp_template, 0.99))
        ax[0, 0].set_title('Template (Moving)')

        ax[zii, 1].imshow(disp_meanbrain[:, :, z_ind].T, alpha=0.5, cmap='Purples', vmax=np.quantile(disp_meanbrain, 0.99))
        ax[0, 1].set_title('Meanbrain (Fixed)')

    for i,reg in enumerate(reg_list):
        template_warped = reg['warpedmovout']

        glom_mask_warped = ants.apply_transforms(fixed=meanbrain,
                                                moving=glom_mask,
                                                transformlist=reg['fwdtransforms'],
                                                interpolator='genericLabel')

        disp_glom = glom_mask_warped[*disp_template_xy_slices, :].numpy()
        glom_tmp = np.ma.masked_where(disp_glom == 0, disp_glom)  # mask at 0

        disp_template = template_warped[*disp_template_xy_slices, :].numpy()
        disp_template = np.ma.masked_where(disp_template == 0, disp_template)  # mask at 0

        for zii, z_ind in enumerate(disp_z_indices):
            ax[zii, 2+2*i].imshow(disp_template[:, :, z_ind].T, alpha=0.5, cmap='Greens', vmax=np.quantile(disp_template, 0.99))
            # ax[s_ind, 1+2*i].imshow(glom_tmp[:, :, z_ind].T, alpha=0.5, cmap=cc.cm.glasbey, vmin=1, vmax=45, interpolation='nearest')

            ax[zii, 3+2*i].imshow(disp_meanbrain[:, :, z_ind].T, alpha=0.5, cmap='Purples', vmax=np.quantile(disp_meanbrain, 0.99))
            ax[zii, 3+2*i].imshow(glom_tmp[:, :, z_ind].T, alpha=0.5, cmap=cc.cm.glasbey, vmin=1, vmax=45, interpolation='nearest')
        
        ax[0, 2+2*i].set_title(f'{i} ({reg["type"]}): Template')
        ax[0, 3+2*i].set_title(f'{i} ({reg["type"]}): Glom mask')
    return fh, ax

def plot_MI(reg_list, meanbrain, template):
    n_panels = len(reg_list) + 1 # original + each reg step

    fh, ax = plt.subplots(n_panels, 1, figsize=(6, 4*n_panels))

    mi_orig = ants.image_mutual_information(meanbrain, template)
    ax[0].imshow(meanbrain.max(axis=2).T, vmax=np.quantile(meanbrain.max(axis=2).ravel(), 0.9))
    ax[0].set_title(f'Original, MI = {mi_orig:.3f}')

    for i,reg in enumerate(reg_list):
        mi_reg = ants.image_mutual_information(meanbrain, reg['warpedmovout'])
        ax[i+1].imshow(reg['warpedmovout'].max(axis=2).T)
        ax[i+1].set_title(f'{reg["type"]}, MI = {mi_reg:.3f}')

    for x in ax.ravel():
        x.grid(which='major', axis='both', linestyle='-', color='r')
    return fh, ax

# %% LOAD
base_dir = dataio.get_config_file()['base_dir']

meanbrain_dir = os.path.join(base_dir, 'mean_brain', meanbrain_tag)
meanbrain_fn = 'chat_meanbrain_{}.nii'.format(meanbrain_tag)
lobe_mask_fn = 'lobe_mask_chat_meanbrain_{}.nii'.format(meanbrain_tag)

# Load meanbrain
meanbrain = ants.image_read(os.path.join(meanbrain_dir, meanbrain_fn))
[meanbrain_red, meanbrain_green] = ants.split_channels(meanbrain)
meanbrain_1ch = meanbrain_red if use_red_meanbrain else meanbrain_green

# Load lobe mask
if use_lobe_mask:
    lobe_mask = np.asanyarray(nib.load(os.path.join(meanbrain_dir, lobe_mask_fn)).dataobj).astype('uint32')
    lobe_mask = ants.from_numpy(np.squeeze(lobe_mask), spacing=meanbrain_red.spacing)
else:
    lobe_mask = None

# Load template & glom mask
template = ants.image_read(os.path.join(base_dir, 'template_brain', template_fn))
glom_mask = ants.image_read(os.path.join(base_dir, 'template_brain', glom_mask_fn))

# Load mask key for VPN types
vpn_types = pd.read_csv(os.path.join(base_dir, 'template_brain', 'vpn_types.csv'))

print('Meanbrain shape:', meanbrain_1ch.shape)
print('Template shape:', template.shape)

n_meanbrain_z = meanbrain_1ch.shape[2]
n_template_z = template.shape[2]

# Initialize reg list
reg_list = []

# %% SHOW TEMPLATE BRAIN and MEAN BRAIN (align on the latter)
n_disp = n_template_z//3
n_cols = 5
n_rows = n_disp // n_cols + 1

fh, ax = plt.subplots(n_rows, n_cols, figsize=(16, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax.ravel()]
for z in range(n_template_z//3):
    ax[z].imshow(template[:, :, z*3].numpy().T)

fh, ax = plt.subplots(n_rows, n_cols, figsize=(16, 8))
ax = ax.ravel()
[x.set_axis_off() for x in ax.ravel()]
for z in range(n_meanbrain_z//3):
    ax[z].imshow(meanbrain_1ch[:, :, z*3].numpy().T)

_ = plot_alignment(reg_list, meanbrain_1ch, template, glom_mask, 
                    disp_z_indices, disp_meanbrain_xy_slices, disp_template_xy_slices)

# %% COMPUTE ALIGNMENT IN TWO STAGES
# 1) Affine
# 2) SyN
# Breaking it up helps troubleshoot. Make sure affine is close first, then mess with SyN warp as needed

# (1) Affine alignment. ~30 sec
reg = ants.registration(fixed=ants.n4_bias_field_correction(meanbrain_1ch),
                        moving=ants.n4_bias_field_correction(template),
                        type_of_transform='Affine',
                        mask=lobe_mask, 
                        flow_sigma=6,
                        total_sigma=0,
                        random_seed=random_seed,
                        aff_sampling=32,
                        grad_step=0.05,
                        reg_iterations=[250, 100, 50])
reg['type'] = 'Affine'
reg_list.append(reg)

_ = plot_alignment(reg_list, meanbrain_1ch, template, glom_mask,
                    disp_z_indices, disp_meanbrain_xy_slices, disp_template_xy_slices)

# %%
# (2) Then compute a small SyN warp on top of this. MI metric. ~60 sec
# output reg_syn transform contains both this new syn warp and the initial affine transform

reg = ants.registration(fixed=ants.n4_bias_field_correction(meanbrain_1ch),
                        moving=ants.n4_bias_field_correction(template),
                        type_of_transform='SyN',
                        mask=lobe_mask,
                        initial_transform=reg_list[-1]['fwdtransforms'],
                        random_seed=random_seed)
reg['type'] = 'SyN'
reg_list.append(reg)

_ = plot_alignment(reg_list, meanbrain_1ch, template, glom_mask, 
                    disp_z_indices, disp_meanbrain_xy_slices, disp_template_xy_slices)

# %% ASSESS OVERALL ALIGNMENT
_ = plot_MI(reg_list, meanbrain_1ch, template)
fh,_ = plot_alignment(reg_list, meanbrain_1ch, template, glom_mask, 
                        disp_z_indices, disp_meanbrain_xy_slices, disp_template_xy_slices)

fh.savefig(os.path.join(base_dir, 'template_brain', reg_fig_fn))

# %%
# APPLY ALIGNMENT TO MASK & TEMPLATE
template_warped = ants.apply_transforms(fixed=meanbrain_1ch,
                                        moving=template,
                                        transformlist=reg_list[-1]['fwdtransforms'],
                                        interpolator='nearestNeighbor')

glom_mask_warped = ants.apply_transforms(fixed=meanbrain_1ch,
                                         moving=glom_mask,
                                         transformlist=reg_list[-1]['fwdtransforms'],
                                         interpolator='genericLabel')

# Save transform
transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_template')
os.makedirs(transform_dir, exist_ok=True)
dataio.save_transforms(reg_list[-1], transform_dir)

# Save transformed images
ants.image_write(template_warped, os.path.join(transform_dir, template_warped_fn))
ants.image_write(glom_mask_warped, os.path.join(transform_dir, glom_mask_warped_fn))

# Save overlay multichannel image
merged = ants.merge_channels([meanbrain_red, meanbrain_green, template_warped, glom_mask_warped])
save_path = os.path.join(transform_dir, overlay_fn)
ants.image_write(merged, save_path)



# %% Re-apply saved transform to remade glom map
transform_dir = os.path.join(base_dir, 'transforms', 'meanbrain_template')

# Load brains
meanbrain = ants.image_read(os.path.join(meanbrain_dir, meanbrain_fn))
[meanbrain_red, meanbrain_green] = ants.split_channels(meanbrain)
meanbrain_1ch = meanbrain_red if use_red_meanbrain else meanbrain_green

# Load glom mask
glom_mask = ants.image_read(os.path.join(base_dir, 'template_brain', glom_mask_fn))

# load pre-registered (above) template
template_warped = ants.image_read(os.path.join(transform_dir, template_warped_fn))

# Load transform
transform_list = dataio.get_transform_list(transform_dir, direction='forward')

glom_mask_warped = ants.apply_transforms(fixed=meanbrain_1ch,
                                         moving=glom_mask,
                                         transformlist=transform_list,
                                         interpolator='genericLabel')

# Save transformed images
ants.image_write(glom_mask_warped, os.path.join(transform_dir, glom_mask_warped_fn))

# Save overlay multichannel image
merged = ants.merge_channels([meanbrain_red, meanbrain_green, template_warped, glom_mask_warped])
save_path = os.path.join(transform_dir, overlay_fn)
ants.image_write(merged, save_path)

# %%
