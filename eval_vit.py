import typing
import io
import os
import argparse

from tqdm import tqdm
import torch
import numpy as np
!pip install opencv-python
import cv2
import matplotlib.pyplot as plt

from urllib.request import urlretrieve

from PIL import Image
from torchvision import transforms

from models.modeling import VisionTransformer, CONFIGS
from utils.cifar10_ood import CIFAR10_OOD_ViT


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ViT-B_16', help='Bleh')

if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'

args = parser.parse_args()
print (args)


class cargs:
    img_size = 224
    pretrained_dir = "output_models/cifar10/ViT-L_32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "cifar10"
    local_rank = -1
    train_batch_size = 128
    eval_batch_size = 128


config = CONFIGS[args.model]

models = []

model_dict = {
	'ViT-B_16': 'vit_b_16',
	'ViT-B_32': 'vit_b_32',
	'ViT-L_16': 'vit_l_16',
	'ViT-L_32': 'vit_l_32'
}

for i in range(1, 4):
	model = VisionTransformer(config, cargs.img_size, zero_head=True, num_classes=10, vis=True)
	model.load_state_dict(torch.load(args.pretrained_dir + f'/Run{i}/{args.dataset}_{model_dict[args.model]}_{i}_checkpoint.bin'))
	model.to(cargs.device)
	models.append(model)


from utils.data_utils import get_loader
from ipywidgets import IntProgress

_, cifar10_test_loader = get_loader(cargs)
cargs.dataset = "cifar100"
_, cifar100_test_loader = get_loader(cargs)
cargs.dataset = "svhn"
_, svhn_test_loader = get_loader(cargs)

cifar10_ood = CIFAR10_OOD_ViT(path='./data/ood_generated')
cifar10_ood_loader = torch.utils.data.DataLoader(
                         cifar10_ood,
                         batch_size=eval_batch_size,
                         shuffle=False,
                         num_workers=4,
                         pin_memory=True
                     )


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

def get_logits_labels(model, data_loader, device):
    """
    Utility function to get logits and labels.
    """
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            logit, _ = model(data)
            logits.append(logit)
            labels.append(label)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    return logits, labels

def test_classification_net_softmax(softmax_prob, labels):
    """
    This function reports classification accuracy and confusion matrix given softmax vectors and
    labels from a model.
    """
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    confidence_vals, predictions = torch.max(softmax_prob, dim=1)
    labels_list.extend(labels.cpu().numpy())
    predictions_list.extend(predictions.cpu().numpy())
    confidence_vals_list.extend(confidence_vals.cpu().numpy())
    accuracy = accuracy_score(labels_list, predictions_list)
    return (
        confusion_matrix(labels_list, predictions_list),
        accuracy,
        labels_list,
        predictions_list,
        confidence_vals_list,
    )

def test_classification_net_logits(logits, labels):
    """
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.
    """
    softmax_prob = F.softmax(logits, dim=1)
    return test_classification_net_softmax(softmax_prob, labels)


def test_classification_net(model, data_loader, device):
    """
    This function reports classification accuracy and confusion matrix over a dataset.
    """
    logits, labels = get_logits_labels(model, data_loader, device)
    return test_classification_net_logits(logits, labels)

device = "cuda" if torch.cuda.is_available() else "cpu"

accs = []
for model in models:
    _, accuracy, _, _, _ = test_classification_net(model, cifar10_test_loader, device)
    accs.append(accuracy)

accs = torch.tensor(accs)

# RESULT HERE!
mean_acc = torch.mean(accs).item()
std_acc = torch.std(accs).item()


import torch
import torch.nn.functional as F


def entropy(logits):
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=1)
    return entropy


def logsumexp(logits):
    return torch.logsumexp(logits, dim=1, keepdim=False)


def confidence(logits):
    p = F.softmax(logits, dim=1)
    confidence, _ = torch.max(p, dim=1)
    return confidence


def entropy_prob(probs):
    p = probs
    eps = 1e-12
    logp = torch.log(p + eps)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=1)
    return entropy


def mutual_information_prob(probs):
    mean_output = torch.mean(probs, dim=0)
    predictive_entropy = entropy_prob(mean_output)

    # Computing expectation of entropies
    p = probs
    eps = 1e-12
    logp = torch.log(p + eps)
    plogp = p * logp
    exp_entropies = torch.mean(-torch.sum(plogp, dim=2), dim=0)

    # Computing mutual information
    mi = predictive_entropy - exp_entropies
    return mi



import torch
import torch.nn.functional as F
from sklearn import metrics


def get_roc_auc(net, test_loader, ood_test_loader, uncertainty, device, confidence=False):
    logits, _ = get_logits_labels(net, test_loader, device)
    ood_logits, _ = get_logits_labels(net, ood_test_loader, device)

    return get_roc_auc_logits(logits, ood_logits, uncertainty, device, confidence=confidence)


def get_roc_auc_logits(logits, ood_logits, uncertainty, device, confidence=False):
    uncertainties = uncertainty(logits)
    ood_uncertainties = uncertainty(ood_logits)

    # In-distribution
    bin_labels = torch.zeros(uncertainties.shape[0]).to(device)
    in_scores = uncertainties

    # OOD
    bin_labels = torch.cat((bin_labels, torch.ones(ood_uncertainties.shape[0]).to(device)))

    if confidence:
        bin_labels = 1 - bin_labels
    ood_scores = ood_uncertainties  # entropy(ood_logits)
    scores = torch.cat((in_scores, ood_scores))

    fpr, tpr, thresholds = metrics.roc_curve(bin_labels.cpu().numpy(), scores.cpu().numpy())
    precision, recall, prc_thresholds = metrics.precision_recall_curve(bin_labels.cpu().numpy(), scores.cpu().numpy())
    auroc = metrics.roc_auc_score(bin_labels.cpu().numpy(), scores.cpu().numpy())
    auprc = metrics.average_precision_score(bin_labels.cpu().numpy(), scores.cpu().numpy())

    return (fpr, tpr, thresholds), (precision, recall, prc_thresholds), auroc, auprc


aurocs_entropy_svhn = []
aurocs_confidence_svhn = []
aurocs_entropy_cifar100 = []
aurocs_confidence_cifar100 = []
aurocs_entropy_cifar10_m = []
aurocs_confidence_cifar10_m = []

for model in models:
    (_, _, _), (_, _, _), auroc_e, _ = get_roc_auc(model, cifar10_test_loader, svhn_test_loader, entropy, device)
    (_, _, _), (_, _, _), auroc_c, _ = get_roc_auc(model, cifar10_test_loader, svhn_test_loader, confidence, device, confidence=True)
    aurocs_entropy_svhn.append(auroc_e)
    aurocs_confidence_svhn.append(auroc_c)

    (_, _, _), (_, _, _), auroc_e, _ = get_roc_auc(model, cifar10_test_loader, cifar100_test_loader, entropy, device)
    (_, _, _), (_, _, _), auroc_c, _ = get_roc_auc(model, cifar10_test_loader, cifar100_test_loader, confidence, device, confidence=True)
    aurocs_entropy_cifar100.append(auroc_e)
    aurocs_confidence_cifar100.append(auroc_c)

    (_, _, _), (_, _, _), auroc_e, _ = get_roc_auc(model, cifar10_test_loader, cifar10_ood_loader, entropy, device)
    (_, _, _), (_, _, _), auroc_c, _ = get_roc_auc(model, cifar10_test_loader, cifar10_ood_loader, confidence, device, confidence=True)
    aurocs_entropy_cifar10_m.append(auroc_e)
    aurocs_confidence_cifar10_m.append(auroc_c)


aurocs_entropy_svhn_mean = torch.mean(aurocs_entropy_svhn).item()
aurocs_entropy_svhn_std = torch.std(aurocs_entropy_svhn).item()

aurocs_confidence_svhn_mean = torch.mean(aurocs_confidence_svhn).item()
aurocs_confidence_svhn_std = torch.std(aurocs_confidence_svhn).item()

aurocs_entropy_cifar100_mean = torch.mean(aurocs_entropy_cifar100).item()
aurocs_entropy_cifar100_std = torch.std(aurocs_entropy_cifar100).item()

aurocs_confidence_cifar100_mean = torch.mean(aurocs_confidence_cifar100).item()
aurocs_confidence_cifar100_std = torch.std(aurocs_confidence_cifar100).item()

aurocs_entropy_cifar10_m_mean = torch.mean(aurocs_entropy_cifar10_m).item()
aurocs_entropy_cifar10_m_std = torch.std(aurocs_entropy_cifar10_m).item()

aurocs_confidence_cifar10_m_mean = torch.mean(aurocs_confidence_cifar10_m).item()
aurocs_confidence_cifar10_m_std = torch.std(aurocs_confidence_cifar10_m).item()


res_dict = {
    'acc': {
        'mean': mean_acc,
        'std': std_acc
    },
    'auroc_svhn_entropy': {
        'mean': aurocs_entropy_svhn_mean,
        'std': aurocs_entropy_svhn_std
    },
    'auroc_svhn_confidence': {
        'mean': aurocs_confidence_svhn_mean,
        'std': aurocs_confidence_svhn_std
    },
    'auroc_cifar100_entropy': {
        'mean': aurocs_entropy_cifar100_mean,
        'std': aurocs_entropy_cifar100_std
    },
    'auroc_cifar100_confidence': {
        'mean': aurocs_confidence_cifar100_mean,
        'std': aurocs_confidence_cifar100_std
    },
    'auroc_cifar10_m_entropy': {
        'mean': aurocs_entropy_cifar10_m_mean,
        'std': aurocs_entropy_cifar10_m_std
    },
    'auroc_cifar10_m_confidence': {
        'mean': aurocs_confidence_cifar10_m_mean,
        'std': aurocs_confidence_cifar10_m_std
    },
}

import json

with open(f'res_{args.model}.json', 'w+') as fp:
    json.dump(res_dict, fp)
