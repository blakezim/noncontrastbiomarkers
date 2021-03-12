import os
import glob
import yaml
import torch
import shutil
import numpy as np
import CAMP.Core as core
import CAMP.FileIO as io
import CAMP.StructuredGridOperators as so

from scipy.io import loadmat
from scipy.ndimage.morphology import binary_erosion, binary_dilation


def process_mrti_data(rabbit, base_dir, out_path):
    rerun = False
    if os.path.exists(f'{out_path}/{rabbit}_thermal_vol.nii.gz') and not rerun:
        print(f'Processing MRTI for {rabbit} ... done')
        return
    print(f'Processing MRTI for {rabbit} ... ', end='')
    data_dir = f'/hdscratch/ucair/AcuteBiomarker/Data/mrti/'
    files = sorted(glob.glob(f'{data_dir}/*'))
    mat_file = [x for x in files if rabbit in x][0]

    mrti_dict = loadmat(mat_file)

    cs_data = np.transpose(mrti_dict['PosDCS'][:, :, :, 0:3], (2, 1, 0, 3))
    mrti_data = np.transpose(mrti_dict['temps'], (2, 1, 0, 3))
    ctd_map = np.nan_to_num(np.transpose(mrti_dict['CTD_Map'], (2, 1, 0)), nan=1.0)
    ctd_map = np.log(ctd_map)
    origin = np.ascontiguousarray(cs_data[0, 0, 0, ::-1])

    thermal_vol = core.StructuredGrid(
        size=mrti_data.shape[0:3],
        origin=origin,
        spacing=torch.tensor([1.0, 1.0, 1.0]),
        tensor=torch.tensor(mrti_data).permute(3, 0, 1, 2),
        channels=mrti_data.shape[-1]
    )

    log_ctd_map = core.StructuredGrid(
        size=mrti_data.shape[0:3],
        origin=origin,
        spacing=torch.tensor([1.0, 1.0, 1.0]),
        tensor=torch.tensor(ctd_map).unsqueeze(0),
        channels=1
    )

    io.SaveITKFile(thermal_vol, f'{out_path}/{rabbit}_thermal_vol.nii.gz')
    io.SaveITKFile(log_ctd_map, f'{out_path}/{rabbit}_log_ctd_map.nii.gz')
    print('done')


def process_masks(rabbit, out_path):
    rerun = False
    if os.path.exists(f'{out_path}/{rabbit}_tissue_seg.nii.gz') and not rerun:
        print(f'Processing Mask for {rabbit} ... done')
        return
    print(f'Processing Mask for {rabbit} ... ', end='')
    data_dir = f'/hdscratch/ucair/AcuteBiomarker/Data/extras/'
    files = sorted(glob.glob(f'{data_dir}/*'))
    mat_file = [x for x in files if rabbit in x][0]

    roi_dict = loadmat(mat_file)

    grid = io.LoadITKFile(f'{out_path}/{rabbit}_log_ctd_map.nii.gz')

    nonv = np.transpose(roi_dict['ROInonV'], (2, 1, 0))
    nonv0 = np.transpose(roi_dict['ROInonV0'], (2, 1, 0))
    tissue = np.transpose(roi_dict['ROItissue'], (2, 1, 0))
    tumor = np.transpose(roi_dict['ROItumr'], (2, 1, 0))
    # mrti_data = np.transpose(mrti_dict['temps'], (2, 1, 0, 3))
    # ctd_map = np.nan_to_num(np.transpose(mrti_dict['CTD_Map'], (2, 1, 0)), nan=1.0)
    # ctd_map = np.log(ctd_map)
    # origin = np.ascontiguousarray(cs_data[0, 0, 0, ::-1])

    nonv0_vol = core.StructuredGrid(
        size=grid.size,
        origin=grid.origin,
        spacing=grid.spacing,
        tensor=torch.tensor(nonv0).unsqueeze(0),
        channels=1
    )

    tumor_vol = core.StructuredGrid(
        size=grid.size,
        origin=grid.origin,
        spacing=grid.spacing,
        tensor=torch.tensor(tumor).unsqueeze(0),
        channels=1
    )

    tissue_vol = core.StructuredGrid(
        size=grid.size,
        origin=grid.origin,
        spacing=grid.spacing,
        tensor=torch.tensor(tissue).unsqueeze(0),
        channels=1
    )

    nonv_vol = core.StructuredGrid(
        size=grid.size,
        origin=grid.origin,
        spacing=grid.spacing,
        tensor=torch.tensor(nonv).unsqueeze(0),
        channels=1
    )

    io.SaveITKFile(nonv_vol, f'{out_path}/{rabbit}_nonv_seg.nii.gz')
    io.SaveITKFile(nonv0_vol, f'{out_path}/{rabbit}_nonv0_seg.nii.gz')
    io.SaveITKFile(tumor_vol, f'{out_path}/{rabbit}_tumor_seg.nii.gz')
    io.SaveITKFile(tissue_vol, f'{out_path}/{rabbit}_tissue_seg.nii.gz')
    print('done')


def get_param_file_dict(rabbit):
    def check_motion(file, r):
        # Check for motion corrected volumes
        file_num = file.split('/')[-1].split('_')[0]
        motion_list = sorted(glob.glob(f'/scratch/rabbit_data/{r}/elastVolumes/day0_motion/*'))
        motion_file = [x for x in motion_list if file_num in x and '.nii.gz' in x]
        if motion_file:
            return motion_file[0]
        else:
            return file

    # Get a list of the raw ablation files
    file_list = sorted(glob.glob(f'/scratch/rabbit_data/{rabbit}/rawVolumes/Ablation*/*'))

    # Get the t2 file list
    t2_w_file_list = [x for x in file_list if 't2_spc_1mm' in x]
    t2_map_file_list = [x for x in file_list if 'T2Map' in x]
    t1_w_file_list = [x for x in file_list if '3D_VIBE_1mmIso' in x and 'Post' not in x and 'POST' not in x]
    t1_wc_file_list = [x for x in file_list if '3D_VIBE_1mmIso' in x and 'Post' in x or 'POST' in x]
    if not t1_wc_file_list:
        print(' Wrong Labels ... ', end='')
        t1_w_file_list = [x for x in file_list if '3D_VIBE_1mmIso' in x and ('cor.nii' in x or 'cor_pre' in x)]
        t1_wc_file_list = [x for x in file_list if '3D_VIBE_1mmIso' in x and 'cor.nii' not in x and 'cor_p' not in x]

    t1_map_file_list = [x for x in file_list if 'T1Map' in x]
    adc_file_list = [x for x in file_list if 'ADC' in x]
    day0_npv_file_list = [x for x in file_list if 'npv' in x or 'NPV' in x and '.nrrd' in x]

    # Assume the first and last t2 are the ones we want
    param_dict = {
        't2_w_pre_file': check_motion(t2_w_file_list[0], rabbit),
        't2_w_post_file': check_motion(t2_w_file_list[-1], rabbit),
        't2_map_pre_file': check_motion(t2_map_file_list[0], rabbit),
        't2_map_post_file': check_motion(t2_map_file_list[-1], rabbit),
        't1_w_pre_file': check_motion(t1_w_file_list[0], rabbit),
        't1_w_post_file': check_motion(t1_w_file_list[-1], rabbit),
        't1_wc_post_file': check_motion(t1_wc_file_list[-1], rabbit),
        't1_map_pre_file': check_motion(t1_map_file_list[0], rabbit),
        't1_map_post_file': check_motion(t1_map_file_list[-1], rabbit),
        'adc_pre_file': check_motion(adc_file_list[0], rabbit),
        'adc_post_file': check_motion(adc_file_list[-1], rabbit),
        'day0_npv_file': day0_npv_file_list[0]
    }

    return param_dict

    
def copy_contrast_files(rabbit, base_dir, out_path):

    rerun = False
    print(f'Copying files for {rabbit} ... ', end='')
    
    if not os.path.exists(f'{out_path}/{rabbit}_param_file_dict.yaml'):
        param_file_dict = get_param_file_dict(rabbit)
        with open(f'{out_path}/{rabbit}_param_file_dict.yaml', 'w') as f:
            yaml.dump(param_file_dict, f)
    else:
        with open(f'{out_path}/{rabbit}_param_file_dict.yaml', 'r') as f:
             param_file_dict = yaml.load(f, Loader=yaml.FullLoader)

    for key in param_file_dict.keys():

        if os.path.exists(f'{out_path}/{key}.nii.gz') and not rerun:
            continue

        file = param_file_dict[key]
        shutil.copy(file, f'{out_path}/{rabbit}_{key}.nii.gz')

    print('done')


def generate_hist_label(rabbit, base_dir, out_path):

    rerun = True
    device = 'cuda:1'

    # if os.path.exists(f'{out_path}/{rabbit}_hist_label.nii.gz') and not rerun:
    #     print(f'Processing label for {rabbit} ... done')
    #     return

    print(f'Processing label for {rabbit} ... ', end='')
    # Get the path for the
    try:
        deform = io.LoadITKFile(f'{out_path}/{rabbit}_day3_to_day0_phi_inv.nii.gz', device=device)
    except:
        def_file = f'/home/sci/blakez/ucair/longitudinal/{rabbit}/'
        def_file += 'deformation_fields/Day3_non_contrast_VIBE_interday_deformation_incomp.nii.gz'
        deform = io.LoadITKFile(def_file, device=device)
        io.SaveITKFile(deform, f'{out_path}/{rabbit}_day3_to_day0_phi_inv.nii.gz')

    # Load the ablation to day3 segmentation
    hist_file = f'{base_dir}/{rabbit}/microscopic/recons/all_ablation_segs_to_invivo.nrrd'
    if not os.path.exists(hist_file):
        hist_file = f'{base_dir}/{rabbit}/microscopic/recons/all_ablation_segs_to_invivo.mhd'

    hist_seg = io.LoadITKFile(hist_file, device=device)
    def_hist = so.ApplyGrid.Create(deform, device=device)(hist_seg, deform)
    def_hist.to_('cpu')

    def_hist.data = (def_hist.data >= 0.5).float()

    def_hist_np = binary_erosion(binary_dilation(def_hist.data.squeeze(), iterations=4), iterations=4)
    def_hist.data = torch.tensor(def_hist_np).unsqueeze(0).float()

    io.SaveITKFile(def_hist, f'{out_path}/{rabbit}_hist_label.nii.gz')

    print('done')


def compile():
    rabbit_list = ['18_047', '18_060', '18_061', '18_062']

    for rabbit in rabbit_list:
        if rabbit == '18_047':
            base_dir = '/hdscratch2/'
        else:
            base_dir = '/hdscratch/ucair/'
        out_path = f'/hdscratch/ucair/AcuteBiomarker/Data/{rabbit}/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # process_mrti_data(rabbit, base_dir, out_path)

        # Get a list of the ablation files
        # copy_contrast_files(rabbit, base_dir, out_path)

        generate_hist_label(rabbit, base_dir, out_path)

        # process_masks(rabbit, out_path)


if __name__ == '__main__':
    compile()
