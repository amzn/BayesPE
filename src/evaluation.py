import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torchmetrics.classification import MulticlassCalibrationError
from bpe import replace_nans_with_uniform


SMALL_CONSTANT = 0.00001
METRICS = ['nll', 'auc', 'ece', 'mce', 'brier']
DETERMINISTIC_METRICS = ['f1', 'acc']


def labels_to_probs(labels, max_samples=None, n_classes=None):
    """
    Converts a list or 1D array of integer labels to a 2D array of probabilities.

    :param labels: 2D array of integer labels [n_inputs, n_samples]
    :param max_samples: maximum number of samples to use to make distribution (default None means use all available)
    :param n_classes: number of classes (by default it takes the max of the labels +1)

    :return: 2D array containing class probabilities [n_inputs, n_classes].
    """
    if max_samples is None:
        n_samples = np.shape(labels)[1]
    else:
        n_samples = np.minimum(max_samples, np.shape(labels)[1])
    if n_classes is None:
        n_classes = int(np.amax(labels)) + 1
    probs = np.zeros((np.shape(labels)[0], n_classes))
    for i in range(np.shape(labels)[0]):
        probs_i = np.zeros(n_classes)
        for j in range(n_samples):
            if labels[i,j]==-1:
                probs_ij = np.divide(1.0, n_classes)*np.ones(n_classes)
            else:
                probs_ij = SMALL_CONSTANT*np.ones(n_classes)
                probs_ij[int(labels[i,j])] = 1.0
            probs_i = probs_i + probs_ij
        probs[i,:] = np.divide(probs_i, n_samples)
    return probs


def negative_log_likelihood(gt_labels, inferred_probs):
    """
    Computes the mean negative log likelihood of gt_labels given inferred class probabilities over a given set.

    :param gt_labels: list or 1D array of integer ground-truth labels [n_examples]
    :param inferred_probs: 2D array of class probabilities [n_examples, n_classes]

    :return: a single float giving the mean negative log likelihood.
    """
    nll = 0.0
    for i in range(len(gt_labels)):
        nll = nll - np.log(inferred_probs[i, int(gt_labels[i])])
    return np.divide(nll, len(gt_labels))


def roc_auc(gt_labels, inferred_probs):
    """
    Computes the ROC-AUC score given ground-truth labels and inferred class probabilities.

    :param gt_labels: list or 1D array of integer ground-truth labels [n_examples]
    :param inferred_probs: 2D array of class probabilities [n_examples, n_classes]

    :return: a single float giving the ROC-AUC score.
    """
    for i in range(np.shape(inferred_probs)[0]):
        inferred_probs[i] = np.divide(inferred_probs[i], np.sum(inferred_probs[i]))
    if np.shape(inferred_probs)[1]>2:
        return roc_auc_score(gt_labels, inferred_probs, multi_class='ovr', labels=np.arange(np.shape(inferred_probs)[1]))
    else:
        return roc_auc_score(gt_labels, inferred_probs[:,1])


def reliability_histograms(gt_labels, probs, n_bars=10):
    """
    Computes the reliability histograms for the given ground-truth labels and classification probabilities.

    :param gt_labels: list or 1D array of integer ground-truth labels [n_examples]
    :param probs: 2D array of class probabilities [n_examples, n_classes]
    :param n_bars: number of hystograms

    :return: a 1D array containing the ratio of correct predictions for each confidence bin.
    :return: a 1D array containing the average confidence for each confidence bin.
    :return: a 1D array containing the number of probabilities added to the bin.
    """
    all_count = np.zeros(n_bars+1)
    sum_probs = np.zeros(n_bars+1)
    correct_count = np.zeros(n_bars+1)
    for i in range(len(gt_labels)):
        for j in range(np.shape(probs)[1]):
            if probs[i,j]>=0.0 and probs[i,j]<=1.0:
                probs_ij = probs[i,j]
            else:
                probs_ij = np.divide(1.0, np.shape(probs)[1])
            rounded_prob = int(np.round(probs_ij * n_bars))
            all_count[rounded_prob] = all_count[rounded_prob] + 1.0
            sum_probs[rounded_prob] = sum_probs[rounded_prob] + probs_ij
            if gt_labels[i] == j:
                correct_count[rounded_prob] = correct_count[rounded_prob] + 1.0
    return np.divide(correct_count, all_count+SMALL_CONSTANT), np.divide(sum_probs, all_count+SMALL_CONSTANT), all_count


def expected_calibration_error(gt_labels, probs, n_bars=10):
    """
    Computes the expected calibration error (ECE).

    :param gt_labels: list or 1D array of integer ground-truth labels [n_examples]
    :param probs: 2D array of class probabilities [n_examples, n_classes]
    :param n_bars: number of histograms to compute ECE over

    :return: the expected calibration error
    """
    metric = MulticlassCalibrationError(np.shape(probs)[1], n_bins=n_bars)
    return metric.forward(torch.from_numpy(probs), torch.from_numpy(gt_labels)).numpy()


def maximum_calibration_error(gt_labels, probs, n_bars=10):
    """
    Computes the maximum calibration error (MCE).

    :param gt_labels: list or 1D array of integer ground-truth labels [n_examples]
    :param probs: 2D array of class probabilities [n_examples, n_classes]
    :param n_bars: number of histograms to compute ECE over

    :return: the maximum calibration error
    """
    acc_histos, conf_histos, counts = reliability_histograms(gt_labels, probs, n_bars=n_bars)
    valid_histos = counts > 0.0
    return np.amax(np.multiply(valid_histos,np.abs(acc_histos-conf_histos)))


def brier_score(gt_labels, probs):
    """
    Computes the brier score.

    :param gt_labels: list or 1D array of integer ground-truth labels [n_examples]
    :param probs: 2D array of class probabilities [n_examples, n_classes]

    :return: the brier score
    """
    gt_array = labels_to_probs(np.expand_dims(gt_labels, axis=1))
    return np.mean(np.square(gt_array - probs))


def compute_probabilistic_metric(gt_labels, inferred_probs, metric):
    """
    Computes a selected probabilistic classification metric.

    :param gt_labels: list or 1D array of integer ground-truth labels [n_examples]
    :param inferred_probs: 2D array of class probabilities [n_examples, n_classes]
    :param metric: the name of the metric to compute (see METRICS above)

    :return: a single float giving the metric computed.
    """
    inferred_probs = replace_nans_with_uniform(inferred_probs)
    if metric==METRICS[0]:
        return negative_log_likelihood(gt_labels, inferred_probs)
    elif metric==METRICS[1]:
        return roc_auc(gt_labels, inferred_probs)
    elif metric==METRICS[2]:
        return expected_calibration_error(gt_labels, inferred_probs, n_bars=10)
    elif metric==METRICS[3]:
        return maximum_calibration_error(gt_labels, inferred_probs, n_bars=10)
    elif metric==METRICS[4]:
        return brier_score(gt_labels, inferred_probs)
    else:
        print('metric {} not yet implemented.'.format(metric))
        print('available probabilistic metrics:')
        for p_metric in METRICS:
            print(p_metric)


def compute_deterministic_metric(gt_labels, inferred_labels_or_probs, metric):
    """
    Computes a selected probabilistic classification metric.

    :param gt_labels: list or 1D array of integer ground-truth labels [n_examples]
    :param inferred_labels_or_probs: 1D array of inferred labels [n_examples] or 2 array of class probabilities [n_examples, n_classes]
    :param metric: the name of the metric to compute (see DETERMINISTIC_METRICS above)

    :return: a single float giving the metric computed.
    """
    if len(np.shape(inferred_labels_or_probs))>1:
        inferred_labels = np.argmax(inferred_labels_or_probs, axis=1)
    else:
        inferred_labels = inferred_labels_or_probs

    if metric==DETERMINISTIC_METRICS[0]:
        return f1_score(gt_labels, inferred_labels, average='macro')
    elif metric==DETERMINISTIC_METRICS[1]:
        return accuracy_score(gt_labels, inferred_labels)
    else:
        print('metric {} not yet implemented.'.format(metric))
        print('available probabilistic metrics:')
        for p_metric in DETERMINISTIC_METRICS:
            print(p_metric)

def compute_metric(gt_labels, inferred_labels_or_probs, metric):
    if metric in METRICS:
        return compute_probabilistic_metric(gt_labels, inferred_labels_or_probs, metric)
    elif metric in DETERMINISTIC_METRICS:
        return compute_deterministic_metric(gt_labels, inferred_labels_or_probs, metric)
    else:
        print('metric {} not yet implemented.'.format(metric))
        print('available probabilistic metrics:')
        for p_metric in DETERMINISTIC_METRICS:
            print(p_metric)
        for p_metric in METRICS:
            print(p_metric)
