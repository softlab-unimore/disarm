import torch
import random
import configparser
import argparse
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ast import literal_eval


class_weights = {
    "student_essay": [1,10],
    "debate": [1,1],
    "m-arg": [9.375, 30, 1]
}

def arg_check(args):
    if args["grid_search"]:
        assert args["adversarial"], "Grid search can only be applied with adversarial training. Please run the program with adversarial training if you want to use grid_search"
    if args["adversarial"] and not args["grid_search"]:
        assert args["discovery_weight"] != -1 and args["adv_weight"] != -1, "You must set grid_search, or directly using discovery_weight and adv_weight to use adversarial training"
    if args["discovery_weight"] != -1 or args["adv_weight"] != -1:
        assert args["adversarial"], "You must choose adversarial training to use discovery_weight and adv_weight"
    if len(args["visualize"]) != 0:
        assert args["visualize"] == "discovery" or args["visualize"] == args["dataset"], "The argument --visualize must have the same value of --dataset or 'discovery'"

    assert args["dataset"] in ["student_essay", "debate", "m-arg"], "The dataset must be one of 'student_essay', 'debate' or 'm-arg'"
    assert len(args["class_weight"]) == args["num_classes"] or len(args["class_weight"]) == 0, "The class_weight must be of the same size as the number of targets inside the dataset"

def get_config():
    parser = argparse.ArgumentParser(description="Argument parser for model configuration")

    parser.add_argument('--model_name', type=str, default='roberta-base', help='Model name')
    parser.add_argument('--embed_size', type=int, default=768, help='Embedding size')
    parser.add_argument('--first_last_avg', type=int, default=1, help='Use first and last average')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_sent_len', type=int, default=150, help='Maximum sentence length')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--discovery_weight', type=float, default=-1, help='Discovery weight')
    parser.add_argument('--adv_weight', type=float, default=-1, help='Adversarial weight')
    parser.add_argument('--adversarial', action="store_true", default=0, help='Use adversarial training')
    parser.add_argument('--dataset_from_saved', action="store_true", default=0, help='Load dataset from saved checkpoint')
    parser.add_argument('--injection', action="store_true", default=0, help='Use injection method')
    parser.add_argument('--grid_search', action="store_true", default=0, help='Perform grid search')
    parser.add_argument('--visualize', type=str, default="", help='Visualize results')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')

    parser.add_argument('--class_weight', type=float, nargs='+', default=[], help='Class weights')

    args = vars(parser.parse_args())

    args["num_classes_adv"] = 3
    args["num_classes"] = 2 if args["dataset"] in ["student_essay", "debate"] else 3

    arg_check(args)

    if len(args["class_weight"]) == 0:
        args["class_weight"] = class_weights[args["dataset"]]

    return args



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
