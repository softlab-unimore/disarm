import torch
import random
import configparser
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    return device

def set_random_seeds(seed):
    """
    set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def output_metrics(labels, preds):
    """

    :param labels: ground truth labels
    :param preds: prediction labels
    :return: accuracy, precision, recall, f1
    """
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")

    print("{:15}{:<.6f}".format('accuracy:', accuracy))
    print("{:15}{:<.6f}".format('precision:', precision))
    print("{:15}{:<.6f}".format('recall:', recall))
    print("{:15}{:<.6f}".format('f1:', f1))

    return accuracy, precision, recall, f1