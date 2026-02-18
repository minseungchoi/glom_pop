"""
Make Meanbrain from anatomical scans

maxwellholteturner@gmail.com
https://github.com/mhturner/glom_pop
"""
#%%
import os
import csv
import re
import numpy as np
import ants
import matplotlib.pyplot as plt
import time
import glob

from glom_pop import dataio
from panglom_suite import plotting


meanbrain_tag = '20260218_weighted'
brain_input_dir = '/run/media/minseung/analysis/glom_project_2/mean_brain/brains'

base_dir = dataio.get_config_file()['base_dir']
brain_dir = os.path.join(base_dir, 'mean_brain', meanbrain_tag)
figure_dir = os.path.join(brain_dir, 'figures')
os.makedirs(figure_dir, exist_ok=True)

# Registration-quality weighting options
use_quality_weighting = True
outlier_reject_frac = 0.2  # Set to e.g. 0.1 to drop the worst 10% at each stage
weight_temperature = 0.75  # Lower values increase contrast between good and bad fits
quality_report_dir = os.path.join(brain_dir, 'registration_quality')
os.makedirs(quality_report_dir, exist_ok=True)


# %% REFERENCE BRAIN CANDIDATES

def show_reference_brain(reference_fn):
    reference_brain = ants.image_read(os.path.join(brain_input_dir, reference_fn))
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
reference_brain = ants.image_read(os.path.join(brain_input_dir, chosen_reference_fn))
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


def compute_registration_cc(fixed, moving, eps=1e-8):
    """
    Compute normalized cross-correlation on overlapping nonzero voxels.
    """
    fixed_np = fixed.numpy().astype(np.float64)
    moving_np = moving.numpy().astype(np.float64)

    valid = np.isfinite(fixed_np) & np.isfinite(moving_np) & (fixed_np > 0) & (moving_np > 0)
    n_valid = int(np.sum(valid))
    if n_valid < 100:
        return np.nan, n_valid

    fixed_vals = fixed_np[valid]
    moving_vals = moving_np[valid]
    fixed_vals = (fixed_vals - fixed_vals.mean()) / (fixed_vals.std() + eps)
    moving_vals = (moving_vals - moving_vals.mean()) / (moving_vals.std() + eps)
    score = float(np.mean(fixed_vals * moving_vals))
    return score, n_valid


def compute_quality_weights(scores, reject_frac=0.0, temperature=0.75):
    """
    Convert per-brain quality scores into nonnegative weights and optional outlier mask.
    """
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    keep = finite.copy()

    if not np.any(finite):
        weights = np.ones_like(scores, dtype=np.float64)
        return weights / np.sum(weights), np.ones_like(scores, dtype=bool)

    if reject_frac > 0 and np.sum(finite) > 1:
        cutoff = np.quantile(scores[finite], reject_frac)
        keep = keep & (scores >= cutoff)
        if not np.any(keep):
            keep = finite.copy()

    if np.sum(keep) == 1:
        weights = keep.astype(np.float64)
        return weights / np.sum(weights), keep

    keep_scores = scores[keep]
    median = np.median(keep_scores)
    mad = np.median(np.abs(keep_scores - median))
    scale = 1.4826 * mad if mad > 1e-8 else (np.std(keep_scores) + 1e-8)
    z = (scores - median) / (scale + 1e-8)
    z = np.clip(z, -4, 4)

    logits = np.full_like(scores, -np.inf, dtype=np.float64)
    logits[keep] = z[keep] / max(temperature, 1e-6)
    logits[keep] = logits[keep] - np.max(logits[keep])

    weights = np.zeros_like(scores, dtype=np.float64)
    weights[keep] = np.exp(logits[keep])
    if np.sum(weights) == 0:
        weights = keep.astype(np.float64)
    weights = weights / np.sum(weights)
    return weights, keep


def weighted_nanmean(stack, weights):
    """
    Weighted mean across last axis, ignoring NaNs in stack.
    """
    weights = np.asarray(weights, dtype=np.float64)
    reshape_dims = (1,) * (stack.ndim - 1) + (weights.size,)
    w = weights.reshape(reshape_dims)

    valid = np.isfinite(stack)
    weighted_sum = np.sum(np.where(valid, stack * w, 0.0), axis=-1)
    weight_sum = np.sum(np.where(valid, w, 0.0), axis=-1)

    out = np.full(stack.shape[:-1], np.nan, dtype=np.float64)
    np.divide(weighted_sum, weight_sum, out=out, where=weight_sum > 0)
    return out


def save_quality_report(path, file_paths, scores, weights, keep_mask, n_valid_voxels):
    """
    Save per-brain registration quality and weights to CSV.
    """
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['brain', 'score_cc', 'weight', 'kept', 'n_valid_voxels'])
        for fp, score, weight, keep, n_valid in zip(file_paths, scores, weights, keep_mask, n_valid_voxels):
            writer.writerow([os.path.basename(fp), float(score), float(weight), bool(keep), int(n_valid)])


def get_fly_label(file_path):
    """
    Return compact fly label like 20250424-001 from TSeries filename.
    """
    base = os.path.basename(file_path)
    if base.startswith('TSeries-'):
        parts = base.split('_')[0].split('-')
        if len(parts) >= 3:
            return '-'.join(parts[1:3])
    return base.split('_')[0]


def get_stage_prefix(stage_name):
    """
    Return zero-padded stage prefix for sorting, e.g. '00' from 'stage0_rigid'.
    """
    match = re.search(r'stage(\d+)', stage_name)
    if match:
        return f'{int(match.group(1)):02d}'
    return '99'


def save_stage_nifti_outputs(brain, stage_name, output_tag, output_dir):
    """
    Save merged and single-channel NIfTI outputs for a stage using sortable names.
    """
    os.makedirs(output_dir, exist_ok=True)
    stage_prefix = get_stage_prefix(stage_name)
    base_name = f'{stage_prefix}_{stage_name}_{output_tag}'

    merged_path = os.path.join(output_dir, f'{base_name}.nii.gz')
    ants.image_write(brain, merged_path)

    red_ch, green_ch = ants.split_channels(brain)
    red_path = os.path.join(output_dir, f'{base_name}_ch1.nii.gz')
    green_path = os.path.join(output_dir, f'{base_name}_ch2.nii.gz')
    ants.image_write(red_ch, red_path)
    ants.image_write(green_ch, green_path)

    print(f'Saved stage NIfTI outputs: {merged_path}')


def computeMeanbrain(reference_brain,
                     brains,
                     file_paths,
                     initial_transforms=None,
                     type_of_transform='SyN',
                     smooth_reference=False,
                     smooth_moving=False,
                     use_quality_weighting=False,
                     outlier_reject_frac=0.0,
                     weight_temperature=0.75,
                     stage_name='stage',
                     quality_report_dir=None,
                     fig_output_dir=None,
                     plot_alignment_multipanel=True, **reg_kwargs):
    """
    Generate a meanbrain from a list of anatomical scans.

    :reference_brain: two-channel ANTs image to register each brain to
    :brains: dictionary mapping file paths to (red_channel, green_channel) tuples
    :file_paths: list of file paths to process (keys in the brains dictionary)
    :initial_transforms: optional dictionary mapping file paths to transform
        list used to initialize per-brain registration in this stage
    :type_of_transform: for ants.registration()
    :smooth_reference: whether to smooth the reference brain
    :smooth_moving: whether to smooth the moving brain
    :use_quality_weighting: if True, use quality-weighted averaging across brains
    :outlier_reject_frac: fraction of lowest-scoring brains to exclude per stage
    :weight_temperature: softness for converting quality scores to weights
    :stage_name: name for quality report output
    :quality_report_dir: optional directory where stage quality CSV is saved
    :fig_output_dir: optional directory where per-stage figures are saved
    :plot_alignment_multipanel: whether to plot the alignment of brains
    """
    t0 = time.time()
    corrected_red = []
    corrected_green = []
    quality_scores = []
    n_valid_voxels = []

    if plot_alignment_multipanel:
        stride = 8
        num_slices = int(reference_brain.shape[2]/stride)
        fh, ax = plt.subplots(num_slices, len(file_paths)+1, figsize=((len(file_paths)+1)*2, num_slices*3.5))
        [plotting.clean_axes(x) for x in ax.ravel()]

        z_indices = [z_ind * stride for z_ind in range(num_slices)]
        z_depth_um = [zi * reference_brain.spacing[2] for zi in z_indices]

        for z_ind in range(num_slices):
            ax[z_ind, 0].imshow(ants.split_channels(reference_brain)[0][:, :, z_indices[z_ind]].numpy().T, cmap='Purples')
            ax[z_ind, 0].set_ylabel(f'z={z_depth_um[z_ind]:.1f} um')
        ax[0, 0].set_title('Reference')

    if initial_transforms is None:
        initial_transforms = {}

    next_initial_transforms = {}

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

        prior_transform = initial_transforms.get(fp, None)
        if isinstance(prior_transform, tuple):
            prior_transform = list(prior_transform)
        if prior_transform == []:
            prior_transform = None
        if prior_transform is not None:
            print(f'Using prior transform for {fp}: {prior_transform}')

        reg = ants.registration(fixed=fixed_red,
                                moving=moving_red,
                                type_of_transform=type_of_transform,
                                initial_transform=prior_transform,
                                **reg_kwargs_local)

        # Build a full chain from original moving image to the current fixed image.
        full_fwdtransforms = list(reg['fwdtransforms'])
        if prior_transform is not None:
            prior_list = prior_transform if isinstance(prior_transform, list) else [prior_transform]
            for tx in prior_list:
                if tx not in full_fwdtransforms:
                    full_fwdtransforms.append(tx)
        next_initial_transforms[fp] = full_fwdtransforms
        
        red_reg = ants.apply_transforms(fixed=fixed_red,
                                        moving=individual_red,
                                        transformlist=full_fwdtransforms,
                                        interpolator='bSpline',
                                        spline_order=3,
                                        defaultvalue=0,
                                        )

        green_reg = ants.apply_transforms(fixed=fixed_green,
                                          moving=individual_green,
                                          transformlist=full_fwdtransforms,
                                          interpolator='bSpline',
                                          spline_order=3,
                                          defaultvalue=0)

        red_reg_np = red_reg.clone().numpy()
        red_reg_np[red_reg_np == 0] = np.nan

        green_reg_np = green_reg.clone().numpy()
        green_reg_np[green_reg_np == 0] = np.nan

        quality_score, n_valid = compute_registration_cc(fixed_red, red_reg)
        quality_scores.append(quality_score)
        n_valid_voxels.append(n_valid)

        corrected_red.append(red_reg_np)
        corrected_green.append(green_reg_np)

        if plot_alignment_multipanel:
            for z_ind in range(num_slices):
                ax[z_ind, fp_ind+1].imshow(red_reg_np[:, :, z_indices[z_ind]].T, cmap='Greens')
            ax[0, fp_ind+1].set_title(get_fly_label(fp))

        print('Done with brain {}:{} ({:.2f} sec)'.format(fp_ind, fp.split('/')[-1], time.time()-t0_fp))

    corrected_red_stack = np.stack(corrected_red, -1)
    corrected_green_stack = np.stack(corrected_green, -1)
    quality_scores = np.asarray(quality_scores, dtype=np.float64)

    if use_quality_weighting:
        weights, keep_mask = compute_quality_weights(
            quality_scores,
            reject_frac=outlier_reject_frac,
            temperature=weight_temperature,
        )
    else:
        finite = np.isfinite(quality_scores)
        keep_mask = finite.copy()
        if outlier_reject_frac > 0 and np.sum(finite) > 1:
            cutoff = np.quantile(quality_scores[finite], outlier_reject_frac)
            keep_mask = keep_mask & (quality_scores >= cutoff)
        if not np.any(keep_mask):
            keep_mask = np.ones_like(quality_scores, dtype=bool)
        weights = keep_mask.astype(np.float64)
        weights = weights / np.sum(weights)

    meanbrain_red = weighted_nanmean(corrected_red_stack, weights)
    meanbrain_green = weighted_nanmean(corrected_green_stack, weights)

    if quality_report_dir is not None:
        os.makedirs(quality_report_dir, exist_ok=True)
        quality_report_path = os.path.join(quality_report_dir, f'{stage_name}_quality.csv')
        save_quality_report(quality_report_path, file_paths, quality_scores, weights, keep_mask, n_valid_voxels)
        print(f'Saved quality report: {quality_report_path}')

    if plot_alignment_multipanel:
        for fp_ind, fp in enumerate(file_paths):
            fly_label = get_fly_label(fp)
            title = fly_label
            title_color = 'black'
            if not keep_mask[fp_ind]:
                title = f'{fly_label} EXCLUDED'
                title_color = 'red'
            ax[0, fp_ind+1].set_title(title, color=title_color)
        fh.suptitle(f'({meanbrain_tag}) {stage_name}')
        fh.tight_layout(rect=[0, 0, 1, 0.96])
        if fig_output_dir is not None:
            os.makedirs(fig_output_dir, exist_ok=True)
            stage_prefix = get_stage_prefix(stage_name)
            multipanel_path = os.path.join(fig_output_dir, f'{stage_prefix}_{stage_name}_multipanel.png')
            fh.savefig(multipanel_path, dpi=150)
            print(f'Saved multipanel fig: {multipanel_path}')

    finite_scores = quality_scores[np.isfinite(quality_scores)]
    if finite_scores.size > 0:
        print(
            f'Quality summary ({stage_name}) | '
            f'CC median={np.median(finite_scores):.4f}, '
            f'min={np.min(finite_scores):.4f}, max={np.max(finite_scores):.4f}, '
            f'kept={int(np.sum(keep_mask))}/{len(file_paths)}'
        )
    else:
        print(
            f'Quality summary ({stage_name}) | '
            f'no finite CC scores, kept={int(np.sum(keep_mask))}/{len(file_paths)}'
        )

    # occluded back to 0
    meanbrain_red[np.isnan(meanbrain_red)] = 0
    meanbrain_green[np.isnan(meanbrain_green)] = 0

    # Convert to ANTs image and merge channels
    meanbrain_red = ants.from_numpy(meanbrain_red, spacing=reference_brain.spacing, origin=reference_brain.origin, direction=reference_brain.direction)
    meanbrain_green = ants.from_numpy(meanbrain_green, spacing=reference_brain.spacing, origin=reference_brain.origin, direction=reference_brain.direction)
    meanbrain = ants.merge_channels([meanbrain_red, meanbrain_green])

    print('Computed meanbrain ({} sec)'.format(time.time()-t0))
    return meanbrain, next_initial_transforms


def showBrain(brain, stride, stage_name='stage', output_tag=None, fig_output_dir=None):
    """Quick display z slices of 2 channel brain."""
    num_slices = int(brain.shape[2]/stride)
    fh, ax = plt.subplots(2, num_slices, figsize=(num_slices*3, 4))
    [x.set_axis_off() for x in ax.ravel()]
    if output_tag is None:
        output_tag = meanbrain_tag
    for z_ind in range(num_slices):
        z_idx = z_ind * stride
        z_um = z_idx * brain.spacing[2]
        ax[0, z_ind].imshow(ants.split_channels(brain)[0][:, :, z_idx].numpy().T, cmap='Reds')
        ax[1, z_ind].imshow(ants.split_channels(brain)[1][:, :, z_idx].numpy().T, cmap='Greens')
        ax[0, z_ind].set_title(f'z={z_um:.1f} um')
    fh.suptitle(f'({output_tag}) {stage_name}')
    fh.tight_layout(rect=[0, 0, 1, 0.92])
    if fig_output_dir is not None:
        os.makedirs(fig_output_dir, exist_ok=True)
        stage_prefix = get_stage_prefix(stage_name)
        showbrain_path = os.path.join(fig_output_dir, f'{stage_prefix}_{stage_name}_showbrain.png')
        fh.savefig(showbrain_path, dpi=150)
        print(f'Saved showBrain fig: {showbrain_path}')


# %% Preload all brain images
brains, file_paths = preload_brain_images(brain_input_dir, do_bias_correction=False)
cumulative_transforms = None

# %% Compute meanbrain 0:
# Affine smoothed brains, to get things roughly aligned
stage_name = 'stage0_rigid'
meanbrain_0, _ = computeMeanbrain(
    reference_brain=reference_brain,
    brains=brains,
    file_paths=file_paths,
    initial_transforms=cumulative_transforms,
    type_of_transform='Rigid',
    smooth_reference=True,
    smooth_moving=True,
    use_quality_weighting=use_quality_weighting,
    outlier_reject_frac=outlier_reject_frac,
    weight_temperature=weight_temperature,
    stage_name=stage_name,
    quality_report_dir=quality_report_dir,
    fig_output_dir=figure_dir,
    plot_alignment_multipanel=True
)
showBrain(meanbrain_0, stride=8, stage_name=stage_name, output_tag=meanbrain_tag, fig_output_dir=figure_dir)
save_stage_nifti_outputs(meanbrain_0, stage_name=stage_name, output_tag=meanbrain_tag, output_dir=brain_dir)

# %% Compute meanbrain 1:
# Affine
stage_name = 'stage1_affine'
meanbrain_1, _ = computeMeanbrain(
    reference_brain=meanbrain_0,
    brains=brains,
    file_paths=file_paths,
    initial_transforms=cumulative_transforms,
    type_of_transform='Affine',
    smooth_reference=True,
    smooth_moving=True,
    use_quality_weighting=use_quality_weighting,
    outlier_reject_frac=outlier_reject_frac,
    weight_temperature=weight_temperature,
    stage_name=stage_name,
    quality_report_dir=quality_report_dir,
    fig_output_dir=figure_dir,
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
showBrain(meanbrain_1, stride=8, stage_name=stage_name, output_tag=meanbrain_tag, fig_output_dir=figure_dir)
save_stage_nifti_outputs(meanbrain_1, stage_name=stage_name, output_tag=meanbrain_tag, output_dir=brain_dir)

# %% Compute meanbrain 2:
# Elastic SyN
stage_name = 'stage2_elasticsyn_coarse'
meanbrain_2, _ = computeMeanbrain(
    reference_brain=meanbrain_1,
    brains=brains,
    file_paths=file_paths,
    initial_transforms=cumulative_transforms,
    type_of_transform='ElasticSyN',
    smooth_reference=True,
    smooth_moving=True,
    use_quality_weighting=use_quality_weighting,
    outlier_reject_frac=outlier_reject_frac,
    weight_temperature=weight_temperature,
    stage_name=stage_name,
    quality_report_dir=quality_report_dir,
    fig_output_dir=figure_dir,
    plot_alignment_multipanel=True, 
    grad_step=0.20,                 # ElasticSyN[0.2,6,0]
    flow_sigma=6,
    total_sigma=0.75,
    syn_metric='CC',                # Cross‑Correlation
    syn_sampling=4,                 # CC radius
    reg_iterations=(20, 10, 0),
    histogram_matching=False,
)
showBrain(meanbrain_2, stride=8, stage_name=stage_name, output_tag=meanbrain_tag, fig_output_dir=figure_dir)
save_stage_nifti_outputs(meanbrain_2, stage_name=stage_name, output_tag=meanbrain_tag, output_dir=brain_dir)

# %% Compute meanbrain 3:
# ElasticSyN Fine
stage_name = 'stage3_elasticsyn_fine'
meanbrain_3, _ = computeMeanbrain(
    reference_brain=meanbrain_2,
    brains=brains,
    file_paths=file_paths,
    initial_transforms=cumulative_transforms,
    type_of_transform='ElasticSyN',
    smooth_reference=False,
    smooth_moving=False,
    use_quality_weighting=use_quality_weighting,
    outlier_reject_frac=outlier_reject_frac,
    weight_temperature=weight_temperature,
    stage_name=stage_name,
    quality_report_dir=quality_report_dir,
    fig_output_dir=figure_dir,
    plot_alignment_multipanel=True, 
    grad_step=0.10,                 # ElasticSyN[0.1,3,0]
    flow_sigma=3,
    total_sigma=0.5,
    syn_metric='CC',                # Cross‑Correlation
    syn_sampling=4,                 # CC radius
    reg_iterations=(40, 20, 0),
    histogram_matching=False,
)
showBrain(meanbrain_3, stride=8, stage_name=stage_name, output_tag=meanbrain_tag, fig_output_dir=figure_dir)
save_stage_nifti_outputs(meanbrain_3, stage_name=stage_name, output_tag=meanbrain_tag, output_dir=brain_dir)

# %% Compute final meanbrain:
# SyN
stage_name = 'stage4_syn_final'
meanbrain, _ = computeMeanbrain(
    reference_brain=meanbrain_3,
    brains=brains,
    file_paths=file_paths,
    initial_transforms=cumulative_transforms,
    type_of_transform='SyN',
    smooth_reference=False,
    smooth_moving=False,
    use_quality_weighting=use_quality_weighting,
    outlier_reject_frac=outlier_reject_frac,
    weight_temperature=weight_temperature,
    stage_name=stage_name,
    quality_report_dir=quality_report_dir,
    fig_output_dir=figure_dir,
    plot_alignment_multipanel=True, 
    grad_step=0.10,                 # SyN[0.1,6,0]
    flow_sigma=3,
    total_sigma=0,
    syn_metric='CC',                # Cross‑Correlation
    syn_sampling=4,                 # CC radius
    reg_iterations=(40, 20, 1),
    histogram_matching=False,
)
showBrain(meanbrain, stride=8, stage_name=stage_name, output_tag=meanbrain_tag, fig_output_dir=figure_dir)
save_stage_nifti_outputs(meanbrain, stage_name=stage_name, output_tag=meanbrain_tag, output_dir=brain_dir)

# %% Save final meanbrain
save_path = os.path.join(brain_dir, 'chat_meanbrain_{}.nii'.format(meanbrain_tag))
ants.image_write(meanbrain, save_path)

# save individual channels
ants.image_write(ants.split_channels(meanbrain)[0], os.path.join(brain_dir, 'chat_meanbrain_{}_ch1.nii'.format(meanbrain_tag)))
ants.image_write(ants.split_channels(meanbrain)[1], os.path.join(brain_dir, 'chat_meanbrain_{}_ch2.nii'.format(meanbrain_tag)))

# %%
