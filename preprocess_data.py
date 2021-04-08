import os
import glob
import torch
import CAMP.camp.FileIO as io
import CAMP.camp.StructuredGridOperators as so

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:0'


def process_data(data_dir, rabbit, quad_mask, mask_region='train'):

    thermal_vol = io.LoadITKFile(f'/hdscratch2/NoncontrastBiomarker/Data/{rabbit}/{rabbit}_thermal_vol.nii.gz')
    post_t2 = io.LoadITKFile(f'/hdscratch2/NoncontrastBiomarker/Data/{rabbit}/{rabbit}_t2_w_post_file.nii.gz')
    pre_t2 = io.LoadITKFile(
        f'/hdscratch2/NoncontrastBiomarker/Data/{rabbit}/deform_files/{rabbit}_t2_w_pre_def.nii.gz')
    pre_t1 = io.LoadITKFile(f'/hdscratch2/NoncontrastBiomarker/Data/{rabbit}/{rabbit}_t1_w_pre_file.nii.gz')
    post_t1 = io.LoadITKFile(f'/hdscratch2/NoncontrastBiomarker/Data/{rabbit}/{rabbit}_t1_w_post_file.nii.gz')
    post_adc = io.LoadITKFile(f'/hdscratch2/NoncontrastBiomarker/Data/{rabbit}/{rabbit}_adc_post_file.nii.gz')
    pre_adc = io.LoadITKFile(f'/hdscratch2/NoncontrastBiomarker/Data/{rabbit}/{rabbit}_adc_pre_file.nii.gz')
    hist_label = io.LoadITKFile(f'/hdscratch2/NoncontrastBiomarker/Data/{rabbit}/{rabbit}_hist_label.nii.gz')
    npv_label = io.LoadITKFile(f'/hdscratch2/NoncontrastBiomarker/Data/{rabbit}/{rabbit}_day0_npv_file.nrrd')

    npv_label = so.ResampleWorld.Create(hist_label)(npv_label)
    npv_label.data[npv_label.data >= 0.5] = 1.0

    hist_label = so.ResampleWorld.Create(quad_mask)(hist_label)
    hist_label.data = (hist_label.data >= 0.5).long()

    post_t2 = so.ResampleWorld.Create(quad_mask)(post_t2)
    pre_t2 = so.ResampleWorld.Create(quad_mask)(pre_t2)

    post_adc = so.ResampleWorld.Create(quad_mask)(post_adc)
    pre_adc = so.ResampleWorld.Create(quad_mask)(pre_adc)

    post_t1 = so.ResampleWorld.Create(quad_mask)(post_t1)
    pre_t1 = so.ResampleWorld.Create(quad_mask)(pre_t1)

    quad_mask = quad_mask.data.squeeze().bool()
    thermal_vol.data[torch.isnan(thermal_vol.data)] = 0.0
    hist_mask = hist_label.data[:, quad_mask].squeeze()

    # Pad and unfold the data
    p = 3
    pad_vec = tuple([p // 2] * 2 + [0] * 2 + [p // 2] * 2 + [0] * 2)
    thermal_pad = torch.nn.functional.pad(thermal_vol.data.squeeze(), pad_vec)
    thermal_unfold = thermal_pad.unfold(1, p, 1).unfold(3, p, 1).contiguous()
    # thermal_view = thermal_unfold.reshape(list(thermal_vol.data.shape) + [-1]).permute(0, -1, 1, 2, 3).contiguous()

    thermal = thermal_unfold.clone()[:, quad_mask, :, :].permute(1, 0, 2, 3)

    tdiff_sum = torch.abs((thermal[:, 1:, :, :] - thermal[:, :-1, :, :])).sum(1, keepdim=True)
    tdiff_max = (thermal[:, 1:, :, :] - thermal[:, :-1, :, :]).max(1, keepdim=True)[0]

    t2_diff = (post_t2.data.squeeze() - pre_t2.data.squeeze())
    t1_diff = (post_t1.data.squeeze() - pre_t1.data.squeeze())
    adc_diff = (post_adc.data.squeeze() - pre_adc.data.squeeze())

    # recon_prediction(t2_diff, rabbit, 'max_vec', region=typeL)

    t1_pad = torch.nn.functional.pad(t1_diff, pad_vec[0:6])
    t1_unfold = t1_pad.unfold(0, p, 1).unfold(2, p, 1).contiguous()
    # t2_view = t2_unfold.reshape(list(t2_diff.shape) + [-1]).permute(-1, 0, 1, 2).contiguous()
    t1_view = t1_unfold.clone()[quad_mask, :, :].squeeze().unsqueeze(1)

    t2_pad = torch.nn.functional.pad(t2_diff, pad_vec[0:6])
    t2_unfold = t2_pad.unfold(0, p, 1).unfold(2, p, 1).contiguous()
    # t2_view = t2_unfold.reshape(list(t2_diff.shape) + [-1]).permute(-1, 0, 1, 2).contiguous()
    t2_view = t2_unfold.clone()[quad_mask, :, :].squeeze().unsqueeze(1)

    adc_pad = torch.nn.functional.pad(adc_diff, pad_vec[0:6])
    adc_unfold = adc_pad.unfold(0, p, 1).unfold(2, p, 1).contiguous()
    # adc_view = adc_unfold.reshape(list(adc_diff.shape) + [-1]).permute(-1, 0, 1, 2).contiguous()
    adc_view = adc_unfold.clone()[quad_mask, :, :].squeeze().unsqueeze(1)

    # taf = (thermal >= 43.0).sum(1, keepdim=True)

    rcem = torch.ones_like(thermal.squeeze())
    rcem[thermal >= 43.0] *= 0.5
    rcem[thermal < 43.0] *= 0.25
    ctd = (rcem.pow((43.0 - torch.clamp(thermal, thermal.min(), 95.0))) * 4.5).sum(1, keepdim=True)
    ctd = torch.clamp(ctd, ctd.min(), 1000000)

    # mod_rcem = torch.ones_like(thermal.squeeze())
    # mod_rcem[thermal >= 43.0] *= 0.95
    # mod_rcem[thermal < 43.0] *= 0.5
    # mod_ctd = (mod_rcem.pow((43.0 - thermal)) * 4.5).sum(1, keepdim=True)

    max_vec = thermal.max(1, keepdim=True)[0]

    # ctd_norm = (ctd - ctd.min()) / (ctd.max() - ctd.min())
    # taf_norm = (taf - taf.min()) / float((taf.max() - taf.min()))
    # mod_norm = ((mod_ctd - mod_ctd.min()) / (mod_ctd.max() - mod_ctd.min()))
    # max_vec_norm = ((max_vec - max_vec.min()) / (max_vec.max() - max_vec.min()))
    # t2_vec_norm = ((t2_view - t2_view.min()) / (t2_view.max() - t2_view.min()))
    # adc_vec_norm = ((adc_view - adc_view.min()) / (adc_view.max() - adc_view.min()))
    # tdiff_sum = ((tdiff_sum - tdiff_sum.min()) / (tdiff_sum.max() - tdiff_sum.min()))'ctd', 'max', 't2', 'adc', 'tdiff_sum', 'tdiff_max', 'taf'
    # tdiff_max = ((tdiff_max - tdiff_max.min()) / (tdiff_max.max() - tdiff_max.min()))

    # norm_features = torch.cat([ctd_norm, max_vec_norm, t2_vec_norm, adc_vec_norm, tdiff_sum, tdiff_max, mod_norm], -1)
    feats = torch.cat([ctd, max_vec, t2_view, adc_view, t1_view], 1)

    out_path = '/'.join(data_dir.split('/')[:-2] + ['']) + 'ProcessedData/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    torch.save(feats, f'{out_path}/{rabbit}_{mask_region}_features.pt')
    torch.save(hist_mask, f'{out_path}/{rabbit}_{mask_region}_labels.pt')


if __name__ == '__main__':
    data_dir = '/hdscratch2/NoncontrastBiomarker/Data/'
    rabbit_list = [x.split('/')[-1] for x in sorted(glob.glob(f'{data_dir}/18_*'))]

    norm_features = []
    labels = []

    for rabbit in rabbit_list:
        train_mask = io.LoadITKFile(f'{data_dir}{rabbit}/{rabbit}_train_mask.nrrd')
        eval_mask = io.LoadITKFile(f'{data_dir}{rabbit}/{rabbit}_thermal_tissue.nrrd')
        process_data(data_dir, rabbit, train_mask, mask_region='train')
        process_data(data_dir, rabbit, eval_mask, mask_region='test')

