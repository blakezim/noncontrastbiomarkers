import glob
import torch
import numpy as np
from skimage import measure
import CAMP.camp.FileIO as io
import sklearn.metrics as metrics
import CAMP.camp.StructuredGridOperators as so
from matplotlib.colors import LinearSegmentedColormap

import tools as tls
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()


def recon_prediction(pred, rabbit, region='test'):
    # Load the target volume
    data_dir = f'/hdscratch2/NoncontrastBiomarker/Data/'
    if region == 'train':
        mask = io.LoadITKFile(f'{data_dir}{rabbit}/{rabbit}_train_mask.nrrd')
    else:
        mask = io.LoadITKFile(f'{data_dir}{rabbit}/{rabbit}_thermal_tissue.nrrd')

    out_vol = mask.clone()
    if hasattr(pred, 'float'):
        out_vol.data[mask.data.bool()] = pred.float()
    else:
        out_vol.data[mask.data.bool()] = torch.tensor(pred).float()

    return out_vol


def load_data(rabbit_list, rabbit_dict):
    data_dir = '/hdscratch2/NoncontrastBiomarker/Data/'
    print('===> Loading volumes for each subject ... ')

    # Load the necessary volumes
    for i, r in enumerate(rabbit_list):
        print(f'=> Suject {r}: Loading raw volumes ... ', end='')
        hist_label = io.LoadITKFile(f'{data_dir}{r}/{r}_hist_label.nii.gz')
        # delay_hist_label = io.LoadITKFile(f'{data_dir}{r}/{r}_day3_hist_label.nii.gz')
        acute_npv = io.LoadITKFile(f'{data_dir}{r}/{r}_day0_npv_file.nrrd')
        # delay_npv = io.LoadITKFile(f'{data_dir}{r}/{r}_day3_npv.nii.gz')
        rabbit_dict[r]['ctd_tmp'] = io.LoadITKFile(f'{data_dir}{r}/{r}_log_ctd_map.nii.gz')

        # thermal_vol = io.LoadITKFile(f'{data_dir}{r}/{r}_thermal_vol.nii.gz')
        post_t2 = io.LoadITKFile(f'{data_dir}{r}/{r}_t2_w_post_file.nii.gz')
        pre_t2 = io.LoadITKFile(f'{data_dir}{r}/deform_files/{r}_t2_w_pre_def.nii.gz')
        post_adc = io.LoadITKFile(f'{data_dir}{r}/{r}_adc_post_file.nii.gz')
        pre_adc = io.LoadITKFile(f'{data_dir}{r}/{r}_adc_pre_file.nii.gz')
        eval_mask = io.LoadITKFile(f'{data_dir}{r}/{r}_thermal_tissue.nrrd')
        t1_volume = io.LoadITKFile(f'{data_dir}{r}/{r}_t1_w_post_file.nii.gz')
        train_mask = io.LoadITKFile(f'{data_dir}{r}/{r}_train_mask.nrrd')
        pre_t1_volume = io.LoadITKFile(f'{data_dir}{r}/{r}_t1_w_pre_file.nii.gz')

        rabbit_dict[r]['train_mask'] = train_mask.clone()
        rabbit_dict[r]['eval_mask'] = eval_mask.clone()
        rabbit_dict[r]['t1_vol'] = t1_volume.clone()
        rabbit_dict[r]['acute_raw_histology'] = hist_label.clone()
        rabbit_dict[r]['pre_t1_vol'] = pre_t1_volume.clone()
        print('done')

        print(f'=> Suject {r}: Resampling raw volumes ... ', end='')
        acute_npv_eval = so.ResampleWorld.Create(eval_mask)(acute_npv)
        acute_npv_eval.data = (acute_npv_eval.data >= 0.5).float()
        acute_hist_eval = so.ResampleWorld.Create(eval_mask)(hist_label)
        acute_hist_eval.data = (acute_hist_eval.data >= 0.5).long()
        resamp_size = acute_npv.size * acute_npv.spacing
        acute_npv.set_size(resamp_size)
        acute_hist_label = so.ResampleWorld.Create(acute_npv)(hist_label)
        acute_t1 = so.ResampleWorld.Create(rabbit_dict[r]['ctd_tmp'])(t1_volume)
        acute_eval_mask = so.ResampleWorld.Create(acute_npv)(eval_mask)
        acute_hist_label.data = (acute_hist_label.data >= 0.5).float()
        acute_npv.data = (acute_npv.data >= 0.5).float()
        acute_eval_mask.data = (acute_eval_mask.data >= 0.5)
        acute_npv_blur = so.Gaussian.Create(1, 3, 1, 3)(acute_npv_eval)
        acute_npv_blur_wfov = so.Gaussian.Create(1, 5, 1, 3)(acute_npv)
        rabbit_dict[r]['acute_histology'] = acute_hist_label.clone()
        rabbit_dict[r]['acute_npv'] = acute_npv.clone()
        rabbit_dict[r]['acute_npv_blur'] = acute_npv_blur.clone()
        rabbit_dict[r]['acute_eval_mask'] = eval_mask.clone()
        rabbit_dict[r]['acute_eval_hist'] = acute_hist_eval.clone()
        rabbit_dict[r]['acute_eval_npv'] = acute_npv_eval.clone()
        rabbit_dict[r]['acute_npv_wfov'] = acute_npv_blur_wfov.clone()
        rabbit_dict[r]['acute_t1'] = acute_t1.clone()
        del acute_hist_label, acute_npv

        # resamp_size = delay_npv.size * delay_npv.spacing
        # delay_npv.set_size(resamp_size)
        # delay_hist_label = so.ResampleWorld.Create(delay_npv)(delay_hist_label)
        # delay_hist_label.data = (delay_hist_label.data >= 0.5).float()
        # delay_npv.data = (delay_npv.data >= 0.5).float()
        # rabbit_dict[r]['delay_histology'] = delay_hist_label.clone()
        # rabbit_dict[r]['delay_npv'] = delay_npv.clone()
        # del delay_hist_label, delay_npv

        hist_label = so.ResampleWorld.Create(rabbit_dict[r]['ctd_tmp'])(hist_label)
        hist_label.data = (hist_label.data >= 0.5).float()
        rabbit_dict[r]['therm_histology'] = hist_label.clone()
        del hist_label

        t2_diff = post_t2.clone()
        post_t2 = so.ResampleWorld.Create(rabbit_dict[r]['ctd_tmp'])(post_t2)
        pre_t2 = so.ResampleWorld.Create(rabbit_dict[r]['ctd_tmp'])(pre_t2)
        t2_diff.data = (post_t2.data - pre_t2.data)
        t2_diff = so.ResampleWorld.Create(rabbit_dict[r]['ctd_tmp'])(t2_diff)
        rabbit_dict[r]['t2w_vol'] = t2_diff.clone()
        del t2_diff

        adc_diff = post_adc.clone()
        post_adc = so.ResampleWorld.Create(rabbit_dict[r]['ctd_tmp'])(post_adc)
        pre_adc = so.ResampleWorld.Create(rabbit_dict[r]['ctd_tmp'])(pre_adc)
        adc_diff.data = (post_adc.data - pre_adc.data)
        adc_diff = so.ResampleWorld.Create(rabbit_dict[r]['ctd_tmp'])(adc_diff)
        rabbit_dict[r]['adc_vol'] = adc_diff.clone()
        del adc_diff

        # thermal_vol.data[torch.isnan(thermal_vol.data)] = 0.0
        # # thermal_vol.data = thermal_vol.data.max(dim=0, keepdim=True)[0]
        # rabbit_dict[r]['thermal_vol'] = thermal_vol.clone()
        # del thermal_vol
        print('done')
    print('===> Done loading volumes for each subject.')


def run_classifiers(rabbit_list, rabbit_dict):

    print('===> Training and evaluating predictive models ... ')
    for i, r in enumerate(rabbit_list):
        print(f'=> Suject {r}: Generating training data ... ', end='')
        #### Generate the data for training and testing the classifier models ####
        train_feats, train_labels, test_feats, test_labels = tls.get_data(['ctd', 'max', 't2', 'adc'])

        temp_label = train_labels.copy()
        temp_feats = train_feats.copy()

        rabbit_dict[r]['test_labels'] = test_labels[i].clone()
        test_feats = test_feats[i].clone()

        _ = temp_label.pop(i)
        _ = temp_feats.pop(i)

        rabbit_dict[r]['train_labels'] = torch.cat(temp_label, 0)
        train_feats = torch.cat(temp_feats, 0)

        # Get the mean of the training features
        train_mean_feats = train_feats.mean(0, keepdims=True)
        train_std_feats = train_feats.std(0, keepdim=True)

        # Normalize the test and train features by the training mean
        # train_feats = train_feats / train_mean_feats
        # test_feats = test_feats / train_mean_feats
        train_feats = (train_feats - train_mean_feats) / train_std_feats
        test_feats = (test_feats - train_mean_feats) / train_std_feats

        # Store the data in the dictionary for later use
        rabbit_dict[r]['train_features'] = train_feats.reshape(train_feats.shape[0], -1).squeeze()
        rabbit_dict[r]['test_features'] = test_feats.reshape(test_feats.shape[0], -1).squeeze()
        print('done')

        print(f'=> Suject {r}: Training and evaluating logistic regression classifier ... ', end='')
        # Train and eval the logistic regression classifier
        rabbit_dict[r]['logistic_model'] = {}
        temp_train_proba, temp_test_proba = tls.logistic_regression(rabbit_dict[r]['train_features'],
                                                                   rabbit_dict[r]['train_labels'],
                                                                   rabbit_dict[r]['test_features'])
        rabbit_dict[r]['logistic_model']['train_proba'] = temp_train_proba
        rabbit_dict[r]['logistic_model']['test_proba'] = temp_test_proba
        rabbit_dict[r]['logistic_model']['test_proba_vol'] = recon_prediction(temp_test_proba, rabbit=r)
        print('done')

        print(f'=> Suject {r}: Training and evaluating random forest classifier ... ', end='')
        # Train and eval the logistic regression classifier
        rabbit_dict[r]['forest_model'] = {}
        temp_train_proba, temp_test_proba = tls.random_forest(rabbit_dict[r]['train_features'],
                                                             rabbit_dict[r]['train_labels'],
                                                             rabbit_dict[r]['test_features'])
        rabbit_dict[r]['forest_model']['train_proba'] = temp_train_proba
        rabbit_dict[r]['forest_model']['test_proba'] = temp_test_proba
        rabbit_dict[r]['forest_model']['test_proba_vol'] = recon_prediction(temp_test_proba, rabbit=r)
        print('done')
    print('===> Done training and evaluating predictive models.')


def eval_classifiers(rabbit_list, rabbit_dict):
    print('===> Evaluating clinical models ... ', end='')
    # classifiers = ['logistic_model', 'forest_model', 'network_model']
    # labels_class = ['LRC', 'RFC', 'CNN']
    classifiers = ['logistic_model', 'forest_model']
    labels_class = ['LRC', 'RFC']
    colors_class = [matplotlib._cm._tab10_data[9], matplotlib._cm._tab10_data[1]]

    test_fmeasure_thold = np.zeros((5, 4))
    train_fmeasure_thold = np.zeros((5, 4))
    test_dice_thold = np.zeros((5, 4))
    train_dice_thold = np.zeros((5, 4))
    ms = 60
    mark = 'o'
    # font = {'color': 'black', 'size': 14}

    prec_train_table = np.zeros((6, 6))
    prec_eval_table = np.zeros((6, 6))
    recall_train_table = np.zeros((6, 6))
    recall_eval_table = np.zeros((6, 6))
    dice_train_table = np.zeros((6, 6))
    dice_eval_table = np.zeros((6, 6))
    mda_train_table = np.zeros((6, 6))
    mda_eval_table = np.zeros((6, 6))
    thold_train_table = np.zeros((6, 6))
    thold_eval_table = np.zeros((6, 6))

    for i, r in enumerate(rabbit_list):
        font = {'family': 'calibri',
                'size': 12}
        matplotlib.rc('font', **font)
        plt.rcParams['font.family'] = 'monospace'
        fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [4, 1]})
        fig.set_figheight(8)

        # These are masks with the original shape
        eval_mask = rabbit_dict[r]['eval_mask'].data.bool().clone()
        train_mask = rabbit_dict[r]['train_mask'].data.bool().clone()

        # These masks are linear to index into linear feature and label vectors
        eval_lin_mask = rabbit_dict[r]['eval_mask'].data[rabbit_dict[r]['eval_mask'].data.bool()].bool()
        train_lin_mask = rabbit_dict[r]['train_mask'].data[rabbit_dict[r]['eval_mask'].data.bool()].bool()

        #### ROC analysis using training mask ####
        # Plot the ROC curves first
        blur_train_data = rabbit_dict[r]['acute_npv_blur'].data[train_mask]
        blur_train_label = rabbit_dict[r]['acute_eval_hist'].data[train_mask]
        blur_tpr, blur_fpr, blur_tholds = metrics.roc_curve(blur_train_label, blur_train_data)
        npv_legend = f'NPV  AUC:{metrics.auc(blur_tpr, blur_fpr):.03f}'
        axs[0].plot(blur_tpr, blur_fpr, label=npv_legend, color='blue')

        # ctd_tpr, ctd_fpr, _ = metrics.roc_curve(rabbit_dict[r]['test_labels'][train_lin_mask],
        #                                         rabbit_dict[r]['test_features'][:, 4][train_lin_mask])
        ctd_data = rabbit_dict[r]['ctd_tmp'].data[train_mask].clone()
        ctd_label = rabbit_dict[r]['therm_histology'].data[train_mask]
        ctd_fpr, ctd_tpr, ctd_tholds = metrics.roc_curve(ctd_label, ctd_data)
        ctd_legend = f'CTD  AUC:{metrics.auc(ctd_fpr, ctd_tpr):.03f}'
        axs[0].plot(ctd_fpr, ctd_tpr, label=ctd_legend, color=matplotlib._cm._tab10_data[3])

        for j, c in enumerate(classifiers):
            tpr, fpr, tholds = metrics.roc_curve(rabbit_dict[r]['test_labels'][train_lin_mask],
                                                 rabbit_dict[r][c]['test_proba'][train_lin_mask])
            class_legend = f'{labels_class[j]}  AUC:{metrics.auc(tpr, fpr):.03f}'
            axs[0].plot(tpr, fpr, label=class_legend, color=colors_class[j])

        plt.gca().tick_params(labelsize=font['size'])
        axs[0].legend(prop={'size': font['size']})
        axs[0].legend(loc='lower right')

        ## Now calculate the thresholds for the ROC analysis on the train mask

        # Original NPV
        npv_train_data = rabbit_dict[r]['acute_eval_npv'].data[train_mask]
        npv_train_label = rabbit_dict[r]['therm_histology'].data[train_mask]
        npv_train_fpr, npv_train_tpr, npv_train_tholds = metrics.roc_curve(npv_train_label, npv_train_data)
        npv_train_dt = 0.5  # We already know what this should be because it is binary
        npv_train_dt_fpr = npv_train_fpr[np.argmin(np.abs(npv_train_tholds - npv_train_dt))]
        npv_train_dt_tpr = npv_train_tpr[np.argmin(np.abs(npv_train_tholds - npv_train_dt))]
        axs[0].scatter(npv_train_dt_fpr, npv_train_dt_tpr, color='black', marker='^')

        # Calculate the stats over the EVAL region
        npv_train_data = rabbit_dict[r]['acute_eval_npv'].data[eval_mask]
        npv_train_label = rabbit_dict[r]['therm_histology'].data[eval_mask]
        prec, recall, dice = tls.calc_prec_rec_dice(npv_train_data >= npv_train_dt, npv_train_label)
        temp = rabbit_dict[r]['acute_eval_npv'] * rabbit_dict[r]['eval_mask']
        temp.data = (temp.data >= npv_train_dt).float()
        mda = tls.calc_mean_dist_agreement(temp, rabbit_dict[r]['therm_histology'])

        prec_train_table[0, i] = prec
        recall_train_table[0, i] = recall
        dice_train_table[0, i] = dice
        mda_train_table[0, i] = mda[0].item()
        thold_train_table[0, i] = npv_train_dt
        del prec, recall, dice, mda

        # Blur NPV
        blur_train_data = rabbit_dict[r]['acute_npv_blur'].data[train_mask]
        blur_train_label = rabbit_dict[r]['therm_histology'].data[train_mask]
        blur_tpr, blur_fpr, blur_tholds = metrics.roc_curve(blur_train_label, blur_train_data)
        blur_train_dt = tls.get_dice_thold(blur_train_data, blur_train_label)
        blur_train_dt_fpr = blur_tpr[np.argmin(np.abs(blur_tholds - blur_train_dt))]
        blur_train_dt_tpr = blur_fpr[np.argmin(np.abs(blur_tholds - blur_train_dt))]
        axs[0].scatter(blur_train_dt_fpr, blur_train_dt_tpr, color='blue', marker=mark, s=ms)

        # Calculate the stats over the EVAL region
        blur_train_data = rabbit_dict[r]['acute_npv_blur'].data[eval_mask]
        blur_train_label = rabbit_dict[r]['therm_histology'].data[eval_mask]
        prec, recall, dice = tls.calc_prec_rec_dice(blur_train_data >= blur_train_dt, blur_train_label)
        temp = rabbit_dict[r]['acute_npv_blur'] * rabbit_dict[r]['eval_mask']
        temp.data = (temp.data >= blur_train_dt).float()
        mda = tls.calc_mean_dist_agreement(temp, rabbit_dict[r]['therm_histology'])

        prec_train_table[1, i] = prec
        recall_train_table[1, i] = recall
        dice_train_table[1, i] = dice
        mda_train_table[1, i] = mda[0].item()
        thold_train_table[1, i] = blur_train_dt
        del prec, recall, dice, mda

        # CTD 240
        ctd_data = rabbit_dict[r]['ctd_tmp'].data[train_mask].clone()
        ctd_label = rabbit_dict[r]['therm_histology'].data[train_mask]
        ctd_fpr, ctd_tpr, ctd_tholds = metrics.roc_curve(ctd_label, ctd_data)
        ctd_train_dt = np.log(240.0)
        ctd_240_fpr = ctd_fpr[np.argmin(np.abs(ctd_tholds - ctd_train_dt))]
        ctd_240_tpr = ctd_tpr[np.argmin(np.abs(ctd_tholds - ctd_train_dt))]
        axs[0].scatter(ctd_240_fpr, ctd_240_tpr, color='black', marker='s')

        # Calculate the stats over the EVAL region
        ctd_data = rabbit_dict[r]['ctd_tmp'].data[eval_mask].clone()
        ctd_label = rabbit_dict[r]['therm_histology'].data[eval_mask]
        prec, recall, dice = tls.calc_prec_rec_dice(ctd_data >= ctd_train_dt, ctd_label)
        temp = rabbit_dict[r]['ctd_tmp'] * rabbit_dict[r]['eval_mask']
        temp.data = (temp.data >= ctd_train_dt).float()
        mda = tls.calc_mean_dist_agreement(temp, rabbit_dict[r]['therm_histology'])

        prec_train_table[2, i] = prec
        recall_train_table[2, i] = recall
        dice_train_table[2, i] = dice
        mda_train_table[2, i] = mda[0].item()
        thold_train_table[2, i] = np.exp(ctd_train_dt)
        del prec, recall, dice, mda

        # CTD model
        ctd_data = rabbit_dict[r]['ctd_tmp'].data[train_mask].clone()
        ctd_label = rabbit_dict[r]['therm_histology'].data[train_mask]
        ctd_fpr, ctd_tpr, ctd_tholds = metrics.roc_curve(ctd_label, ctd_data)
        ctd_temp = (ctd_data.clone() - ctd_data.min()) / (ctd_data.max() - ctd_data.min())  # must be between 0 and 1
        ctd_dt_temp = tls.get_dice_thold(ctd_temp, ctd_label)
        ctd_train_dt = ((ctd_dt_temp * (ctd_data.max() - ctd_data.min())) + ctd_data.min()).item()
        ctd_dt_fpr = ctd_fpr[np.argmin(np.abs(ctd_tholds - ctd_train_dt))]
        ctd_dt_tpr = ctd_tpr[np.argmin(np.abs(ctd_tholds - ctd_train_dt))]
        axs[0].scatter(ctd_dt_fpr, ctd_dt_tpr, color=matplotlib._cm._tab10_data[3], marker=mark, s=ms)

        # Calculate the stats over the EVAL region
        ctd_data = rabbit_dict[r]['ctd_tmp'].data[eval_mask].clone()
        ctd_label = rabbit_dict[r]['therm_histology'].data[eval_mask]
        prec, recall, dice = tls.calc_prec_rec_dice(ctd_data >= ctd_train_dt, ctd_label)
        temp = rabbit_dict[r]['ctd_tmp'] * rabbit_dict[r]['eval_mask']
        temp.data = (temp.data >= ctd_train_dt).float()
        mda = tls.calc_mean_dist_agreement(temp, rabbit_dict[r]['therm_histology'])

        prec_train_table[3, i] = prec
        recall_train_table[3, i] = recall
        dice_train_table[3, i] = dice
        mda_train_table[3, i] = mda[0].item()
        thold_train_table[3, i] = np.exp(ctd_train_dt)
        del prec, recall, dice, mda

        # LRC
        lrc_data = rabbit_dict[r]['logistic_model']['test_proba'][train_lin_mask]
        lrc_label = rabbit_dict[r]['test_labels'][train_lin_mask].float().numpy()
        lrc_train_dt = tls.get_dice_thold(lrc_data, lrc_label)
        lcr_fpr, lcr_tpr, lcr_tholds = metrics.roc_curve(lrc_label, lrc_data)
        lcr_dt_fpr = lcr_fpr[np.argmin(np.abs(lcr_tholds - lrc_train_dt))]
        lcr_dt_tpr = lcr_tpr[np.argmin(np.abs(lcr_tholds - lrc_train_dt))]
        axs[0].scatter(lcr_dt_fpr, lcr_dt_tpr, color=matplotlib._cm._tab10_data[9], marker=mark, s=ms)

        # Calculate the stats over the EVAL region
        lrc_data = rabbit_dict[r]['logistic_model']['test_proba'][eval_lin_mask]
        lrc_label = rabbit_dict[r]['test_labels'][eval_lin_mask].float().numpy()
        prec, recall, dice = tls.calc_prec_rec_dice(lrc_data >= lrc_train_dt, lrc_label)
        pred_vol = rabbit_dict[r]['logistic_model']['test_proba_vol'].clone() * rabbit_dict[r]['eval_mask']
        pred_vol.data = (pred_vol.data >= lrc_train_dt).float()
        mda = tls.calc_mean_dist_agreement(pred_vol, rabbit_dict[r]['therm_histology'])

        prec_train_table[4, i] = prec
        recall_train_table[4, i] = recall
        dice_train_table[4, i] = dice
        mda_train_table[4, i] = mda[0].item()
        thold_train_table[4, i] = lrc_train_dt
        del prec, recall, dice, mda

        # RFC
        rfc_data = rabbit_dict[r]['forest_model']['test_proba'][train_lin_mask]
        rfc_label = rabbit_dict[r]['test_labels'][train_lin_mask].float().numpy()
        rfc_train_dt = tls.get_dice_thold(rfc_data, rfc_label)
        rfc_fpr, rfc_tpr, rfc_tholds = metrics.roc_curve(rfc_label, rfc_data)
        rfc_dt_fpr = rfc_fpr[np.argmin(np.abs(rfc_tholds - rfc_train_dt))]
        rfc_dt_tpr = rfc_tpr[np.argmin(np.abs(rfc_tholds - rfc_train_dt))]
        axs[0].scatter(rfc_dt_fpr, rfc_dt_tpr, color=matplotlib._cm._tab10_data[1], marker=mark, s=ms)

        # Calculate the stats over the EVAL region
        rfc_data = rabbit_dict[r]['forest_model']['test_proba'][eval_lin_mask]
        rfc_label = rabbit_dict[r]['test_labels'][eval_lin_mask].float().numpy()
        prec, recall, dice = tls.calc_prec_rec_dice(rfc_data >= rfc_train_dt, rfc_label)
        pred_vol = rabbit_dict[r]['forest_model']['test_proba_vol'].clone() * rabbit_dict[r]['eval_mask']
        pred_vol.data = (pred_vol.data >= rfc_train_dt).float()
        mda = tls.calc_mean_dist_agreement(pred_vol, rabbit_dict[r]['therm_histology'])

        prec_train_table[5, i] = prec
        recall_train_table[5, i] = recall
        dice_train_table[5, i] = dice
        mda_train_table[5, i] = mda[0].item()
        thold_train_table[5, i] = rfc_train_dt
        del prec, recall, dice, mda

        ##### Now determine the optimal threshold based on the 'eval' region
        # Original NPV
        npv_eval_dt = 0.5  # We already know what this should be because it is binary

        # Calculate the stats over the EVAL region
        npv_eval_data = rabbit_dict[r]['acute_eval_npv'].data[eval_mask]
        npv_eval_label = rabbit_dict[r]['therm_histology'].data[eval_mask]
        prec, recall, dice = tls.calc_prec_rec_dice(npv_eval_data >= npv_eval_dt, npv_eval_label)
        temp = rabbit_dict[r]['acute_eval_npv'] * rabbit_dict[r]['eval_mask']
        temp.data = (temp.data >= npv_eval_dt).float()
        mda = tls.calc_mean_dist_agreement(temp, rabbit_dict[r]['therm_histology'])

        prec_eval_table[0, i] = prec
        recall_eval_table[0, i] = recall
        dice_eval_table[0, i] = dice
        mda_eval_table[0, i] = mda[0].item()
        thold_eval_table[0, i] = npv_eval_dt
        del prec, recall, dice, mda

        # Blur NPV
        blur_eval_data = rabbit_dict[r]['acute_npv_blur'].data[eval_mask]
        blur_eval_label = rabbit_dict[r]['therm_histology'].data[eval_mask]
        blur_eval_dt = tls.get_dice_thold(blur_eval_data, blur_eval_label)

        # Calculate the stats over the EVAL region
        blur_eval_data = rabbit_dict[r]['acute_npv_blur'].data[eval_mask]
        blur_eval_label = rabbit_dict[r]['therm_histology'].data[eval_mask]
        prec, recall, dice = tls.calc_prec_rec_dice(blur_eval_data >= blur_eval_dt, blur_eval_label)
        temp = rabbit_dict[r]['acute_npv_blur'] * rabbit_dict[r]['eval_mask']
        temp.data = (temp.data >= blur_eval_dt).float()
        mda = tls.calc_mean_dist_agreement(temp, rabbit_dict[r]['therm_histology'])

        prec_eval_table[1, i] = prec
        recall_eval_table[1, i] = recall
        dice_eval_table[1, i] = dice
        mda_eval_table[1, i] = mda[0].item()
        thold_eval_table[1, i] = blur_eval_dt
        del prec, recall, dice, mda

        # CTD 240
        ctd_eval_dt = np.log(240.0)

        # Calculate the stats over the EVAL region
        ctd_data = rabbit_dict[r]['ctd_tmp'].data[eval_mask].clone()
        ctd_label = rabbit_dict[r]['therm_histology'].data[eval_mask]
        prec, recall, dice = tls.calc_prec_rec_dice(ctd_data >= ctd_eval_dt, ctd_label)
        temp = rabbit_dict[r]['ctd_tmp'] * rabbit_dict[r]['eval_mask']
        temp.data = (temp.data >= ctd_eval_dt).float()
        mda = tls.calc_mean_dist_agreement(temp, rabbit_dict[r]['therm_histology'])

        prec_eval_table[2, i] = prec
        recall_eval_table[2, i] = recall
        dice_eval_table[2, i] = dice
        mda_eval_table[2, i] = mda[0].item()
        thold_eval_table[2, i] = np.exp(ctd_eval_dt)
        del prec, recall, dice, mda

        # CTD model
        ctd_data = rabbit_dict[r]['ctd_tmp'].data[eval_mask].clone()
        ctd_label = rabbit_dict[r]['therm_histology'].data[eval_mask]
        ctd_temp = (ctd_data.clone() - ctd_data.min()) / (ctd_data.max() - ctd_data.min())  # must be between 0 and 1
        ctd_dt_temp = tls.get_dice_thold(ctd_temp, ctd_label)
        ctd_eval_dt = ((ctd_dt_temp * (ctd_data.max() - ctd_data.min())) + ctd_data.min()).item()

        # Calculate the stats over the EVAL region
        ctd_data = rabbit_dict[r]['ctd_tmp'].data[eval_mask].clone()
        ctd_label = rabbit_dict[r]['therm_histology'].data[eval_mask]
        prec, recall, dice = tls.calc_prec_rec_dice(ctd_data >= ctd_eval_dt, ctd_label)
        temp = rabbit_dict[r]['ctd_tmp'] * rabbit_dict[r]['eval_mask']
        temp.data = (temp.data >= ctd_eval_dt).float()
        mda = tls.calc_mean_dist_agreement(temp, rabbit_dict[r]['therm_histology'])

        prec_eval_table[3, i] = prec
        recall_eval_table[3, i] = recall
        dice_eval_table[3, i] = dice
        mda_eval_table[3, i] = mda[0].item()
        thold_eval_table[3, i] = np.exp(ctd_eval_dt)
        del prec, recall, dice, mda

        # LRC
        lrc_data = rabbit_dict[r]['logistic_model']['test_proba'][eval_lin_mask]
        lrc_label = rabbit_dict[r]['test_labels'][eval_lin_mask].float().numpy()
        lrc_eval_dt = tls.get_dice_thold(lrc_data, lrc_label)

        # Calculate the stats over the EVAL region
        lrc_data = rabbit_dict[r]['logistic_model']['test_proba'][eval_lin_mask]
        lrc_label = rabbit_dict[r]['test_labels'][eval_lin_mask].float().numpy()
        prec, recall, dice = tls.calc_prec_rec_dice(lrc_data >= lrc_eval_dt, lrc_label)
        pred_vol = rabbit_dict[r]['logistic_model']['test_proba_vol'].clone() * rabbit_dict[r]['eval_mask']
        pred_vol.data = (pred_vol.data >= lrc_eval_dt).float()
        mda = tls.calc_mean_dist_agreement(pred_vol, rabbit_dict[r]['therm_histology'])

        prec_eval_table[4, i] = prec
        recall_eval_table[4, i] = recall
        dice_eval_table[4, i] = dice
        mda_eval_table[4, i] = mda[0].item()
        thold_eval_table[4, i] = lrc_eval_dt
        del prec, recall, dice, mda

        # RFC
        rfc_data = rabbit_dict[r]['forest_model']['test_proba'][eval_lin_mask]
        rfc_label = rabbit_dict[r]['test_labels'][eval_lin_mask].float().numpy()
        rfc_eval_dt = tls.get_dice_thold(rfc_data, rfc_label)

        # Calculate the stats over the EVAL region
        rfc_data = rabbit_dict[r]['forest_model']['test_proba'][eval_lin_mask]
        rfc_label = rabbit_dict[r]['test_labels'][eval_lin_mask].float().numpy()
        prec, recall, dice = tls.calc_prec_rec_dice(rfc_data >= rfc_eval_dt, rfc_label)
        pred_vol = rabbit_dict[r]['forest_model']['test_proba_vol'].clone() * rabbit_dict[r]['eval_mask']
        pred_vol.data = (pred_vol.data >= rfc_eval_dt).float()
        mda = tls.calc_mean_dist_agreement(pred_vol, rabbit_dict[r]['therm_histology'])

        prec_eval_table[5, i] = prec
        recall_eval_table[5, i] = recall
        dice_eval_table[5, i] = dice
        mda_eval_table[5, i] = mda[0].item()
        thold_eval_table[5, i] = rfc_eval_dt
        del prec, recall, dice, mda

        plt.gca()
        axs[1].axis('tight')
        axs[1].axis('off')

        axs[0].set_xlabel('FPR')
        axs[0].set_ylabel('TPR')
        axs[0].set_title(f'{r}: ROC Threshold Analysis')
        axs[0].tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
        axs[0].xaxis.set_label_position('top')
        plt.savefig(f'/hdscratch2/NoncontrastBiomarker/Output/{r}/{r}_roc_fig.png',
                    dpi=150, bbox_inches='tight', pad_inches=0)

        # npv_data = rabbit_dict[r]['acute_eval_npv'].data[rabbit_dict[r]['acute_eval_mask'].data.bool()][train_lin_mask]
        # npv_label = rabbit_dict[r]['therm_histology'].data[rabbit_dict[r]['acute_eval_mask'].data.bool()][train_lin_mask]
        #
        # ctd_240 = (ctd_data >= np.log(240)).float()
        #
        # ctd_240_tpr = float(torch.logical_and(ctd_240 == 1, npv_label == 1).sum()) / float(npv_label.sum())
        # ctd_240_fpr = float(torch.logical_and(ctd_240 == 1, npv_label == 0).sum()) / float((npv_label - 1).abs().sum())
        #
        # npv_tpr = float(torch.logical_and(npv_data == 1, npv_label == 1).sum()) / float(npv_label.sum())
        # npv_fpr = float(torch.logical_and(npv_data == 1, npv_label == 0).sum()) / float((npv_label - 1).abs().sum())
        # # axs[0].scatter(npv_fpr, npv_tpr, color='black', marker='s')
        #
        # # Generate the table
        # test_table = np.zeros((6, 5))
        #
        # # Calculate everything for the NPV
        # npv_data = rabbit_dict[r]['acute_eval_npv'].data[rabbit_dict[r]['acute_eval_mask'].data.bool()]
        # npv_label = rabbit_dict[r]['therm_histology'].data[rabbit_dict[r]['acute_eval_mask'].data.bool()]
        # ctd_data = rabbit_dict[r]['ctd_tmp'].data[rabbit_dict[r]['acute_eval_mask'].data.bool()]
        # ctd_240 = (ctd_data >= np.log(240)).float()
        # blur_data = rabbit_dict[r]['acute_npv_blur'].data[rabbit_dict[r]['acute_eval_mask'].data.bool()]
        # blur_label = rabbit_dict[r]['acute_eval_hist'].data[rabbit_dict[r]['acute_eval_mask'].data.bool()]
        # npv_dt = tls.get_dice_thold(blur_data, blur_label)
        #
        # blur_tpr, blur_fpr, blur_tholds = metrics.roc_curve(blur_label[train_mask], blur_data[train_mask])
        # npv_dt_fpr = blur_fpr[np.argmin(np.abs(blur_tholds - npv_dt))]
        # npv_dt_tpr = blur_tpr[np.argmin(np.abs(blur_tholds - npv_dt))]
        #
        # # npv_legend = f'NPV  AUC:{metrics.auc(blur_tpr, blur_fpr):.03f}'
        # # # axs[0].plot(blur_tpr, blur_fpr, label=npv_legend, color=matplotlib._cm._tab10_data[0])
        # # axs[0].plot(blur_tpr, blur_fpr, label=npv_legend, color='blue')
        # # plt.gca().tick_params(labelsize=font['size'])
        # # axs[0].legend(prop={'size': font['size']})
        #
        # axs[0].scatter(npv_fpr, npv_tpr, color='black', marker='^')
        # # axs[0].scatter(npv_dt_tpr, npv_dt_fpr, color=matplotlib._cm._tab10_data[0], marker=mark, s=ms)
        # axs[0].scatter(npv_dt_tpr, npv_dt_fpr, color='blue', marker=mark, s=ms)
        # prec, recall, dice = tls.calc_prec_rec_dice(npv_data, npv_label)
        # mda = tls.calc_mean_dist_agreement(rabbit_dict[r]['acute_eval_npv'] * rabbit_dict[r]['acute_eval_mask'], rabbit_dict[r]['therm_histology'])
        # # Save out some volume for checking
        # io.SaveITKFile(rabbit_dict[r]['acute_eval_npv'], f'/home/sci/blakez/test_acuteBM/{r}_NPV_no_blur.nii.gz')
        # io.SaveITKFile(rabbit_dict[r]['therm_histology'], f'/home/sci/blakez/test_acuteBM/{r}_thermal_histology.nii.gz')
        #
        # test_table[0, 0] = f'{prec:0.2f}'
        # test_table[0, 1] = f'{recall:0.2f}'
        # test_table[0, 2] = f'{dice:0.2f}'
        # test_table[0, 3] = f'{0.50:0.2f}'
        # test_table[0, 4] = f'{mda[0].item():0.2f}'
        #
        # prec, recall, dice = tls.calc_prec_rec_dice(blur_data >= npv_dt, blur_label)
        # temp = rabbit_dict[r]['acute_npv_blur'].clone() * rabbit_dict[r]['acute_eval_mask']
        # temp.data = (temp.data >= npv_dt).float()
        # mda = tls.calc_mean_dist_agreement(temp, rabbit_dict[r]['therm_histology'])
        # io.SaveITKFile(rabbit_dict[r]['acute_npv_blur'], f'/home/sci/blakez/test_acuteBM/{r}_NPV_blur.nii.gz')
        # io.SaveITKFile(temp, f'/home/sci/blakez/test_acuteBM/{r}_NPV_blur_pred.nrrd')
        # test_table[1, 0] = f'{prec:0.2f}'
        # test_table[1, 1] = f'{recall:0.2f}'
        # test_table[1, 2] = f'{dice:0.2f}'
        # test_table[1, 3] = f'{npv_dt:0.2f}'
        # test_table[1, 4] = f'{mda[0].item():0.2f}'
        #
        # # Calculate the numbers for the LCR
        # lcr_dt = tls.get_dice_thold(rabbit_dict[r]['logistic_model']['test_proba'], rabbit_dict[r]['test_labels'])
        # lcr_fpr, lcr_tpr, lcr_tholds = metrics.roc_curve(rabbit_dict[r]['test_labels'][train_mask],
        #                                                  rabbit_dict[r]['logistic_model']['test_proba'][train_mask])
        # rabbit_dict[r]['logistic_model']['lcr_dt'] = lcr_dt
        # lcr_dt_fpr = lcr_fpr[np.argmin(np.abs(lcr_tholds - lcr_dt))]
        # lcr_dt_tpr = lcr_tpr[np.argmin(np.abs(lcr_tholds - lcr_dt))]
        #
        # axs[0].scatter(lcr_dt_fpr, lcr_dt_tpr, color=matplotlib._cm._tab10_data[9], marker=mark, s=ms)
        # prec, recall, dice = tls.calc_prec_rec_dice(rabbit_dict[r]['logistic_model']['test_proba'] >= lcr_dt,
        #                                            rabbit_dict[r]['test_labels'])
        # pred_vol = rabbit_dict[r]['logistic_model']['test_proba_vol'].clone() * rabbit_dict[r]['acute_eval_mask']
        # io.SaveITKFile(pred_vol, f'/home/sci/blakez/test_acuteBM/{r}_logistic_proba.nii.gz')
        # pred_vol.data = (pred_vol.data >= lcr_dt).float()
        # mda = tls.calc_mean_dist_agreement(pred_vol, rabbit_dict[r]['therm_histology'])
        # io.SaveITKFile(pred_vol, f'/home/sci/blakez/test_acuteBM/{r}_logistic_pred.nrrd')
        # test_table[2, 0] = f'{prec:0.2f}'
        # test_table[2, 1] = f'{recall:0.2f}'
        # test_table[2, 2] = f'{dice:0.2f}'
        # test_table[2, 3] = f'{lcr_dt:0.2f}'
        # test_table[2, 4] = f'{mda[0].item():0.2f}'
        #
        # # Calculate the numbers for the RFC
        # rfc_dt = tls.get_dice_thold(rabbit_dict[r]['forest_model']['test_proba'], rabbit_dict[r]['test_labels'])
        # rfc_fpr, rfc_tpr, rfc_tholds = metrics.roc_curve(rabbit_dict[r]['test_labels'][train_mask],
        #                                                  rabbit_dict[r]['forest_model']['test_proba'][train_mask])
        # rabbit_dict[r]['forest_model']['rfc_dt'] = rfc_dt
        # rfc_dt_fpr = rfc_fpr[np.argmin(np.abs(rfc_tholds - rfc_dt))]
        # rfc_dt_tpr = rfc_tpr[np.argmin(np.abs(rfc_tholds - rfc_dt))]
        # axs[0].scatter(rfc_dt_fpr, rfc_dt_tpr, color=matplotlib._cm._tab10_data[1], marker=mark, s=ms)
        # prec, recall, dice = tls.calc_prec_rec_dice(rabbit_dict[r]['forest_model']['test_proba'] >= rfc_dt,
        #                                            rabbit_dict[r]['test_labels'])
        # pred_vol = rabbit_dict[r]['forest_model']['test_proba_vol'].clone() * rabbit_dict[r]['acute_eval_mask']
        # io.SaveITKFile(pred_vol, f'/home/sci/blakez/test_acuteBM/{r}_forest_proba.nii.gz')
        # pred_vol.data = (pred_vol.data >= rfc_dt).float()
        # mda = tls.calc_mean_dist_agreement(pred_vol, rabbit_dict[r]['therm_histology'])
        # io.SaveITKFile(pred_vol, f'/home/sci/blakez/test_acuteBM/{r}_forest_pred.nrrd')
        # test_table[3, 0] = f'{prec:0.2f}'
        # test_table[3, 1] = f'{recall:0.2f}'
        # test_table[3, 2] = f'{dice:0.2f}'
        # test_table[3, 3] = f'{rfc_dt:0.2f}'
        # test_table[3, 4] = f'{mda[0].item():0.2f}'
        #
        # # Calculate the numbers for the CTD
        # axs[0].scatter(ctd_dt_fpr, ctd_dt_tpr, color=matplotlib._cm._tab10_data[3], marker=mark, s=ms)
        # prec, recall, dice = tls.calc_prec_rec_dice(ctd_data >= ctd_dt,
        #                                            rabbit_dict[r]['test_labels'])
        # pred_vol = rabbit_dict[r]['ctd_tmp'].clone() * rabbit_dict[r]['acute_eval_mask']
        # pred_vol.data[torch.isnan(pred_vol.data)] = 0.0
        # io.SaveITKFile(pred_vol, f'/home/sci/blakez/test_acuteBM/{r}_CTD_eval_vol.nii.gz')
        # # max_val = pred_vol.data.max()
        # # min_val = pred_vol.data.min()
        # # pred_vol.data = (pred_vol.data - min_val) / (max_val - min_val)
        # pred_vol.data = (pred_vol.data >= ctd_dt).float()
        # mda = tls.calc_mean_dist_agreement(pred_vol, rabbit_dict[r]['therm_histology'])
        # io.SaveITKFile(pred_vol, f'/home/sci/blakez/test_acuteBM/{r}_CTD_dice_pred.nrrd')
        # # thold = np.exp((ctd_dt * (max_val - min_val)) + min_val)
        # test_table[4, 0] = f'{prec:0.2f}'
        # test_table[4, 1] = f'{recall:0.2f}'
        # test_table[4, 2] = f'{dice:0.2f}'
        # test_table[4, 3] = f'{np.exp(ctd_dt):0.2f}'
        # test_table[4, 4] = f'{mda[0].item():0.2f}'
        #
        # # Calculate the numbers for the NPV
        # axs[0].scatter(ctd_240_fpr, ctd_240_tpr, color='black', marker='s')
        # prec, recall, dice = tls.calc_prec_rec_dice(ctd_240, rabbit_dict[r]['test_labels'])
        # ctd_240_pred = rabbit_dict[r]['ctd_tmp'].clone() * rabbit_dict[r]['acute_eval_mask']
        # ctd_240_pred.data[torch.isnan(ctd_240_pred.data)] = 0.0
        # ctd_240_pred.data = (ctd_240_pred.data >= np.log(240)).float()
        # mda = tls.calc_mean_dist_agreement(ctd_240_pred, rabbit_dict[r]['therm_histology'])
        # io.SaveITKFile(ctd_240_pred, f'/home/sci/blakez/test_acuteBM/{r}_CTD_240_pred.nrrd')
        # test_table[5, 0] = f'{prec:0.2f}'
        # test_table[5, 1] = f'{recall:0.2f}'
        # test_table[5, 2] = f'{dice:0.2f}'
        # test_table[5, 3] = f'{240.0:0.2f}'
        # test_table[5, 4] = f'{mda[0].item():0.2f}'
        #
        # axs[0].legend(loc='lower right')
        #
        # col_labels = ['prec', 'recall', 'dice', 't-hold', 'mda']
        # colors = [["white"] * 5,
        #           [matplotlib._cm._tab10_data[0]] * 5,
        #           [matplotlib._cm._tab10_data[9]] * 5,
        #           [matplotlib._cm._tab10_data[3]] * 5,
        #           [matplotlib._cm._tab10_data[1]] * 5,
        #           ["white"] * 5]
        #
        # the_table = axs[1].table(test_table, cellColours=colors, colLabels=col_labels, loc='center')
        # axs[1].axis('tight')
        # axs[1].axis('off')
        #
        # the_table.auto_set_font_size(False)
        # the_table.set_fontsize(10)
        # the_table.scale(1, 1.2)
        # axs[0].set_xlabel('FPR')
        # axs[0].set_ylabel('TPR')
        # axs[0].set_title(f'{r}: ROC Threshold Analysis')
        # axs[0].tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
        # axs[0].xaxis.set_label_position('top')
        # # plt.savefig(f'./Figures/new_eval/{r}_roc_analysis.png', dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.savefig(f'/hdscratch2/NoncontrastBiomarker/Output/{r}/{r}_roc_fig.png',
        #             dpi=150, bbox_inches='tight', pad_inches=0)
        #
        # np.savetxt(f'/hdscratch2/NoncontrastBiomarker/Output/{r}/metrics.csv', test_table, delimiter=',')
        #
        # # Set some things for the new figure
        # colors = [(0.0, 0.0, 0.0), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)]
        # cm = LinearSegmentedColormap.from_list('hist_color', colors, N=1)
        # contour_width = 2

        # for s in range(rabbit_dict[r]['acute_eval_mask'].shape()[-2]):
        #     print(f'Processing {r} slice {s}: ', end='')
        #     if rabbit_dict[r]['therm_histology'][0, :, s, :].sum() == 0.0:
        #         print('No Histology.')
        #         continue
        #     if rabbit_dict[r]['acute_eval_mask'][0, :, s, :].sum() == 0.0:
        #         print('No Eval Mask.')
        #         continue
        #     plt.figure()
        #     mask = rabbit_dict[r]['acute_eval_mask'][0, :, s, :]
        #     rows = np.any(mask.numpy(), axis=1)
        #     cols = np.any(mask.numpy(), axis=0)
        #     ymin, ymax = np.where(rows)[0][[0, -1]]
        #     xmin, xmax = np.where(cols)[0][[0, -1]]
        #     ymin -= 1
        #     ymax += 1
        #     xmin -= 1
        #     xmax += 1
        #     plt.imshow(rabbit_dict[r]['acute_t1'][0, :, s, :][ymin:ymax + 1, xmin:xmax + 1], cmap='gray')
        #
        #     hist_label = rabbit_dict[r]['therm_histology'][0, :, s, :][ymin:ymax + 1, xmin:xmax + 1]
        #     masked = np.ma.masked_where(hist_label.data.squeeze() == 0, hist_label.data.squeeze())
        #     plt.imshow(masked, cmap=cm, alpha=0.5)
        #
        #     tissue_cont = measure.find_contours(mask[ymin:ymax + 1, xmin:xmax + 1], 0.5)
        #     for contour in tissue_cont:
        #         plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[2], linewidth=contour_width)
        #
        #     rf_pred = rabbit_dict[r]['forest_model']['test_proba_vol'].clone() * rabbit_dict[r]['acute_eval_mask']
        #     rf_pred.data = (rf_pred.data >= rfc_dt).float()
        #     rf_slice = rf_pred.data[0, :, s, :][ymin:ymax + 1, xmin:xmax + 1]
        #     rf_cont = measure.find_contours(rf_slice.cpu().numpy(), 0.5)
        #     try:
        #         for contour in rf_cont:
        #             # Prune some of the smaller contours
        #             if len(contour) <= 9:
        #                 continue
        #             plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[3], linewidth=contour_width)
        #     except IndexError:
        #         print('No Forest Prediction, ', end='')
        #
        #     npv_pred = rabbit_dict[r]['acute_eval_npv'] * rabbit_dict[r]['acute_eval_mask']
        #     npv_slice = npv_pred.data[0, :, s, :][ymin:ymax + 1, xmin:xmax + 1]
        #     npv_cont = measure.find_contours(npv_slice.cpu().numpy(), 0.5)
        #     try:
        #         for contour in npv_cont:
        #             # Prune some of the smaller contours
        #             if len(contour) <= 9:
        #                 continue
        #             plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[0], linewidth=contour_width)
        #     except IndexError:
        #         print('No NPV Prediction, ', end='')
        #
        #     lc_pred = rabbit_dict[r]['logistic_model']['test_proba_vol'].clone() * rabbit_dict[r]['acute_eval_mask']
        #     lc_pred.data = (lc_pred.data >= lcr_dt).float()
        #     lc_slice = lc_pred.data[0, :, s, :][ymin:ymax + 1, xmin:xmax + 1]
        #     lc_cont = measure.find_contours(lc_slice.cpu().numpy(), 0.5)
        #     try:
        #         for contour in lc_cont:
        #             # Prune some of the smaller contours
        #             if len(contour) <= 9:
        #                 continue
        #             plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[9], linewidth=contour_width)
        #     except IndexError:
        #         print('No Logistic Prediction, ', end='')
        #
        #     ctd_pred = rabbit_dict[r]['ctd_tmp'].clone() * rabbit_dict[r]['acute_eval_mask']
        #     ctd_pred.data = (ctd_pred.data >= np.log(240)).float()
        #     ctd_slice = ctd_pred.data[0, :, s, :][ymin:ymax + 1, xmin:xmax + 1]
        #     ctd_cont = measure.find_contours(ctd_slice.cpu().numpy(), 0.5)
        #     try:
        #         for contour in ctd_cont:
        #             # Prune some of the smaller contours
        #             if len(contour) <= 9:
        #                 continue
        #             plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[1], linewidth=contour_width)
        #     except IndexError:
        #         print('No CTD Prediction, ', end='')
        #     print('done. ')
        #     plt.savefig(f'/hdscratch2/NoncontrastBiomarker/Output/{r}/slice_{s}_pred.png',
        #                 dpi=150, bbox_inches='tight', pad_inches=0)
        #     plt.close('all')
    plt.close('all')

    prec_train_table[:, 4] = prec_train_table[:, 0:4].mean(1)
    prec_train_table[:, 5] = prec_train_table[:, 0:4].std(1)
    recall_train_table[:, 4] = recall_train_table[:, 0:4].mean(1)
    recall_train_table[:, 5] = recall_train_table[:, 0:4].std(1)
    dice_train_table[:, 4] = dice_train_table[:, 0:4].mean(1)
    dice_train_table[:, 5] = dice_train_table[:, 0:4].std(1)
    mda_train_table[:, 4] = mda_train_table[:, 0:4].mean(1)
    mda_train_table[:, 5] = mda_train_table[:, 0:4].std(1)
    thold_train_table[:, 4] = thold_train_table[:, 0:4].mean(1)
    thold_train_table[:, 5] = thold_train_table[:, 0:4].std(1)

    prec_eval_table[:, 4] = prec_eval_table[:, 0:4].mean(1)
    prec_eval_table[:, 5] = prec_eval_table[:, 0:4].std(1)
    recall_eval_table[:, 4] = recall_eval_table[:, 0:4].mean(1)
    recall_eval_table[:, 5] = recall_eval_table[:, 0:4].std(1)
    dice_eval_table[:, 4] = dice_eval_table[:, 0:4].mean(1)
    dice_eval_table[:, 5] = dice_eval_table[:, 0:4].std(1)
    mda_eval_table[:, 4] = mda_eval_table[:, 0:4].mean(1)
    mda_eval_table[:, 5] = mda_eval_table[:, 0:4].std(1)
    thold_eval_table[:, 4] = thold_eval_table[:, 0:4].mean(1)
    thold_eval_table[:, 5] = thold_eval_table[:, 0:4].std(1)

    np.savetxt(f'/hdscratch2/NoncontrastBiomarker/Output/metrics/prec_train_metrics.csv', prec_train_table, delimiter=',')
    np.savetxt(f'/hdscratch2/NoncontrastBiomarker/Output/metrics/recall_train_metrics.csv', recall_train_table, delimiter=',')
    np.savetxt(f'/hdscratch2/NoncontrastBiomarker/Output/metrics/dice_train_metrics.csv', dice_train_table, delimiter=',')
    np.savetxt(f'/hdscratch2/NoncontrastBiomarker/Output/metrics/mda_train_metrics.csv', mda_train_table, delimiter=',')
    np.savetxt(f'/hdscratch2/NoncontrastBiomarker/Output/metrics/thold_train_metrics.csv', thold_train_table, delimiter=',')

    np.savetxt(f'/hdscratch2/NoncontrastBiomarker/Output/metrics/prec_eval_metrics.csv', prec_eval_table, delimiter=',')
    np.savetxt(f'/hdscratch2/NoncontrastBiomarker/Output/metrics/recall_eval_metrics.csv', recall_eval_table, delimiter=',')
    np.savetxt(f'/hdscratch2/NoncontrastBiomarker/Output/metrics/dice_eval_metrics.csv', dice_eval_table, delimiter=',')
    np.savetxt(f'/hdscratch2/NoncontrastBiomarker/Output/metrics/mda_eval_metrics.csv', mda_eval_table, delimiter=',')
    np.savetxt(f'/hdscratch2/NoncontrastBiomarker/Output/metrics/thold_eval_metrics.csv', thold_eval_table, delimiter=',')

    print('done')


def generate_figures(rabbit_list, rabbit_dict, rabbit_slices):

    # Set some things for the new figure
    colors = [(0.0, 0.0, 0.0), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)]
    cm = LinearSegmentedColormap.from_list('hist_color', colors, N=1)
    contour_width = 1.0

    for i, r in enumerate(rabbit_list):
        font = {'family': 'calibri',
                'size': 12}
        matplotlib.rc('font', **font)
        plt.rcParams['font.family'] = 'monospace'
        s = rabbit_slices[i]

        print(f'Processing {r} slice {s}: ', end='')
        plt.figure()
        mask = rabbit_dict[r]['acute_eval_mask'][0, :, s, :]
        # rows = np.any(mask.numpy(), axis=1)
        # cols = np.any(mask.numpy(), axis=0)
        # ymin, ymax = np.where(rows)[0][[0, -1]]
        # xmin, xmax = np.where(cols)[0][[0, -1]]
        # ymin -= 10
        # ymax += 10
        # xmin -= 10
        # xmax += 50
        plt.imshow(rabbit_dict[r]['acute_t1'][0, :, s, :], cmap='gray')
        plt.axis('off')

        hist_label = rabbit_dict[r]['therm_histology'][0, :, s, :]
        masked = np.ma.masked_where(hist_label.data.squeeze() == 0, hist_label.data.squeeze())
        plt.imshow(masked, cmap=cm, alpha=0.8)

        rf_pred = rabbit_dict[r]['forest_model']['test_proba_vol'].clone() * rabbit_dict[r]['acute_eval_mask']
        rfc_dt = rabbit_dict[r]['forest_model']['rfc_dt']
        rf_pred.data = (rf_pred.data >= rfc_dt).float()
        rf_slice = rf_pred.data[0, :, s, :]
        rf_cont = measure.find_contours(rf_slice.cpu().numpy(), 0.5)
        try:
            for contour in rf_cont:
                # Prune some of the smaller contours
                if len(contour) <= 9:
                    continue
                plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[1], linewidth=contour_width)
        except IndexError:
            print('No Forest Prediction, ', end='')

        npv_pred = rabbit_dict[r]['acute_eval_npv'] * rabbit_dict[r]['acute_eval_mask']
        npv_slice = npv_pred.data[0, :, s, :]
        npv_cont = measure.find_contours(npv_slice.cpu().numpy(), 0.5)
        try:
            for contour in npv_cont:
                # Prune some of the smaller contours
                if len(contour) <= 9:
                    continue
                # plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[0], linewidth=contour_width)
                plt.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=contour_width)
        except IndexError:
            print('No NPV Prediction, ', end='')

        lc_pred = rabbit_dict[r]['logistic_model']['test_proba_vol'].clone() * rabbit_dict[r]['acute_eval_mask']
        lcr_dt = rabbit_dict[r]['logistic_model']['lcr_dt']
        lc_pred.data = (lc_pred.data >= lcr_dt).float()
        lc_slice = lc_pred.data[0, :, s, :]
        lc_cont = measure.find_contours(lc_slice.cpu().numpy(), 0.5)
        try:
            for contour in lc_cont:
                # Prune some of the smaller contours
                if len(contour) <= 9:
                    continue
                plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[9], linewidth=contour_width)
        except IndexError:
            print('No Logistic Prediction, ', end='')

        ctd_pred = rabbit_dict[r]['ctd_tmp'].clone() * rabbit_dict[r]['acute_eval_mask']
        ctd_pred.data = (ctd_pred.data >= np.log(240)).float()
        ctd_slice = ctd_pred.data[0, :, s, :]
        ctd_cont = measure.find_contours(ctd_slice.cpu().numpy(), 0.5)
        try:
            for contour in ctd_cont:
                # Prune some of the smaller contours
                if len(contour) <= 9:
                    continue
                plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[3], linewidth=contour_width)
        except IndexError:
            print('No CTD Prediction, ', end='')

        plt.savefig(f'/hdscratch2/NoncontrastBiomarker/Output/PaperFigs/{r}_slice_{s}_pred_no_boundary.png',
                    dpi=600, bbox_inches='tight', pad_inches=0)

        # tissue_cont = measure.find_contours(mask, 0.5)
        # for contour in tissue_cont:
        #     plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[2], linewidth=contour_width)
        #
        # plt.savefig(f'/hdscratch2/NoncontrastBiomarker/Output/PaperFigs/{r}_slice_{s}_pred.png',
        #             dpi=600, bbox_inches='tight', pad_inches=0)

        plt.close('all')
        print('done. ')

    print('done')


def process_rabbits():
    data_dir = '/hdscratch2/NoncontrastBiomarker/Data/'

    rabbit_list = [x.split('/')[-1] for x in sorted(glob.glob(f'{data_dir}/18_*'))]

    # Make a dictionary of each of the rabbits
    rabbit_dict = {f'{x}': {} for x in rabbit_list}

    rabbit_slices = [12, 7, 15, 15]

    load_data(rabbit_list, rabbit_dict)
    run_classifiers(rabbit_list, rabbit_dict)
    eval_classifiers(rabbit_list, rabbit_dict)
    generate_figures(rabbit_list, rabbit_dict, rabbit_slices)


if __name__ == '__main__':
    process_rabbits()
