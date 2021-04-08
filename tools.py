import glob
import torch
import numpy as np

import sklearn.metrics as metrics
import CAMP.camp.StructuredGridOperators as so
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

np.random.seed(31415)


def get_fmeasure_thold(data, label):

    fpr, tpr, thold = metrics.roc_curve(label.squeeze(), data.squeeze(), drop_intermediate=False)
    return fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], thold[np.argmax(tpr - fpr)]


def get_dice_thold(data, label):
    test_dce = []
    for t in range(1000):
        _, _, dice = calc_prec_rec_dice((data >= t / 1000.0), label)
        test_dce.append(dice)

    return np.argmax(test_dce) / 1000.0


def get_j_thold(data, label):
    test_j = []
    for t in range(1000):
        j = calc_j_score((data >= t / 1000.0), label)
        test_j.append(j)

    return np.argmax(test_j) / 1000.0


def calc_prec_rec_dice(pred, label):

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.logical_and(pred == 1, label == 1).sum()

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.logical_and(pred == 0, label == 0).sum()

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.logical_and(pred == 1, label == 0).sum()

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.logical_and(pred == 0, label == 1).sum()

    dice = (2 * TP) / float(((2 * TP) + FP + FN))
    prec = TP / float((TP + FP))
    reca = TP / float((TP + FN))

    return prec, reca, dice


def calc_j_score(pred, label):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.float(np.logical_and(pred == 1, label == 1).sum())

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.float(np.logical_and(pred == 0, label == 0).sum())

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.float(np.logical_and(pred == 1, label == 0).sum())

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.float(np.logical_and(pred == 0, label == 1).sum())

    if TP == 0 and FN == 0:
        ft = 0.0
    else:
        ft = (TP / (TP + FN))

    if TN == 0 and FP == 0:
        st = 0.0
    else:
        st = (TN / (TN + FP))

    acc = (ft + st) - 1

    return acc


def get_segmentation_boundaries(seg):
    # Get the boundary indices
    grad_vol = so.Gradient.Create(dim=3)(seg)
    grad_vol.data = (grad_vol.data != 0.0).float()

    # Intersect the gradient with the segmentation because we are using central difference gradient
    grad_vol = grad_vol * seg
    boundary_vol = seg.clone()
    boundary_vol.data = grad_vol.data.max(dim=0, keepdim=True)[0]

    return boundary_vol


def calc_mean_dist_agreement(pred_label, target_label):
    # Get the boundaries of the labels
    pred_bounds = get_segmentation_boundaries(pred_label)
    target_bounds = get_segmentation_boundaries(target_label)

    if pred_bounds.max() == 0.0:
        return torch.tensor(0.0), torch.tensor(0.0)

    # Get the indicies of label
    pred_inds = torch.nonzero(pred_bounds.data.squeeze(), as_tuple=False).float()
    target_inds = torch.nonzero(target_bounds.data.squeeze(), as_tuple=False).float()

    distance_mat = ((pred_inds.permute(1, 0).unsqueeze(0) - target_inds.unsqueeze(2)) ** 2).sum(1).sqrt()
    min_dist_vector = torch.cat([distance_mat.min(dim=0)[0], distance_mat.min(dim=1)[0]], dim=0)
    mean_dist_agreement = min_dist_vector.mean()
    std_dist_agreement = min_dist_vector.std()

    return mean_dist_agreement, std_dist_agreement


def calc_signed_distance(pred_label, target_label):
    # Get the boundaries of the labels
    pred_bounds = get_segmentation_boundaries(pred_label)
    target_bounds = get_segmentation_boundaries(target_label)

    # Get the boundary pixels that are within the target segmentation
    internal = torch.logical_and(pred_bounds.data, target_label.data)
    external = torch.logical_and(torch.logical_xor(pred_bounds.data, target_label.data), pred_bounds.data)

    sign_vol = pred_bounds.clone()
    sign_vol.data[internal] = -1.0
    sign_vol.data[external] = 1.0

    # Get the indicies of label
    signs = sign_vol.data[pred_bounds.data != 0.0]
    pred_inds = torch.nonzero(pred_bounds.data.squeeze(), as_tuple=False).float()
    target_inds = torch.nonzero(target_bounds.data.squeeze(), as_tuple=False).float()

    distance_mat = ((pred_inds.permute(1, 0).unsqueeze(0) - target_inds.unsqueeze(2)) ** 2).sum(1).sqrt()
    min_dist_vector = distance_mat.min(dim=0)[0]
    signed_distance_vector = signs * min_dist_vector

    return signed_distance_vector


def _get_old_data(feats=None, patch=True):
    data_dir = '/home/sci/blakez/ucair/AcuteBiomarker/ClassifierData/Patch3X3/'
    rabbit_list = [x.split('/')[-1].split('_te')[0] for x in sorted(glob.glob(f'{data_dir}18_*_test_label*'))]

    train_feats = []
    train_labels = []

    test_feats = []
    test_labels = []

    if not feats:
        feats = ['ctd', 'max', 't2', 'adc']

    full_feat_list = ['ctd', 'max', 't2', 'adc', 't1']

    for rabbit in rabbit_list:
        train_labels.append(torch.load(f'{data_dir}{rabbit}_train_labels.pt').long())
        test_labels.append(torch.load(f'{data_dir}{rabbit}_test_labels.pt').long())

        train_feat = torch.load(f'{data_dir}{rabbit}_train_features.pt')
        test_feat = torch.load(f'{data_dir}{rabbit}_test_features.pt')

        temp_train_feats = []
        temp_test_feats = []
        # if train_feat.shape[1] != len(full_feat_list):
        #     exit('Wrong number of features in the data...')
        for i, f in enumerate(full_feat_list):
            if f in feats:
                if patch:
                    temp_train_feats.append(train_feat[:, i, None, :, :])
                    temp_test_feats.append(test_feat[:, i, None, :, :])
                else:
                    temp_train_feats.append(train_feat[:, i, None, 1, 1])
                    temp_test_feats.append(test_feat[:, i, None, 1, 1])
        train_feats.append(torch.cat(temp_train_feats, 1))
        test_feats.append(torch.cat(temp_test_feats, 1))

    return train_feats, train_labels, test_feats, test_labels


def get_data(feats=None):
    data_dir = '/hdscratch2/NoncontrastBiomarker/ProcessedData/'
    rabbit_list = [x.split('/')[-1].split('_te')[0] for x in sorted(glob.glob(f'{data_dir}18_*_test_label*'))]

    train_feats = []
    train_labels = []

    test_feats = []
    test_labels = []

    if not feats:
        feats = ['ctd', 'max', 't2', 'adc']

    full_feat_list = ['ctd', 'max', 't2', 'adc', 't1']

    for rabbit in rabbit_list:
        train_labels.append(torch.load(f'{data_dir}{rabbit}_train_labels.pt').long())
        test_labels.append(torch.load(f'{data_dir}{rabbit}_test_labels.pt').long())

        train_feat = torch.load(f'{data_dir}{rabbit}_train_features.pt')
        test_feat = torch.load(f'{data_dir}{rabbit}_test_features.pt')

        temp_train_feats = []
        temp_test_feats = []
        # if train_feat.shape[1] != len(full_feat_list):
        #     exit('Wrong number of features in the data...')
        for i, f in enumerate(full_feat_list):
            if f in feats:
                temp_train_feats.append(train_feat[:, i, None, :, :])
                temp_test_feats.append(test_feat[:, i, None, :, :])

        train_feats.append(torch.cat(temp_train_feats, 1))
        test_feats.append(torch.cat(temp_test_feats, 1))

    # old_train_f, old_train_l, old_test_f, old_test_l = _get_old_data(['ctd', 'max', 't2', 'adc'])
    #
    # if old_train_f != train_feats:
    #     print('Diff Train Feats...', end='')
    # if old_train_l != train_labels:
    #     print('Diff Train Feats...', end='')
    # if old_test_f != test_feats:
    #     print('Diff Train Feats...', end='')
    # if old_test_l != test_labels:
    #     print('Diff Train Feats...', end='')

    return train_feats, train_labels, test_feats, test_labels


def logistic_regression(train_data, train_labels, test_data):

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_data, train_labels)
    train_proba = clf.predict_proba(train_data)[:, 1]
    test_proba = clf.predict_proba(test_data)[:, 1]

    return train_proba, test_proba


def random_forest(train_data, train_labels, test_data):

    clf = RandomForestClassifier(max_depth=21, n_estimators=700, n_jobs=16)
    clf.fit(train_data, train_labels)
    train_proba = clf.predict_proba(train_data)[:, 1]
    test_proba = clf.predict_proba(test_data)[:, 1]

    return train_proba, test_proba
