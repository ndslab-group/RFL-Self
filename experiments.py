#!/usr/bin/env python
import torch
import numpy as np
import copy
from CNN import CNN
from utils import *
from selfishness import *

import create_datasets

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="mnist", help='The dataset to use')
parser.add_argument('--seed', type=int, default=0, help='The seed to use')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--rounds', type=int, default=30, help='Number of rounds')
parser.add_argument('--clients', type=int, default=5, help='Number of clients')
parser.add_argument('--classes', type=int, default=10, help='Number of classes')
parser.add_argument('--selfish', type=int, default=0, help='Number of selfish clients')
parser.add_argument('--selfishness', type=float, default=1, help='Selfishness of selfish clients')
parser.add_argument('--size', type=int, default=200, help='Size of dataset')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
parser.add_argument('--distribution', type=str, default="non_iid", help='iid/non_iid')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument("--log", type=bool, default=False, action=argparse.BooleanOptionalAction, help='compute accuracy of the local models after each round')
parser.add_argument("--redirect", type=bool, default=False, action=argparse.BooleanOptionalAction, help='redirect standard output to file')
parser.add_argument('--aggregation', type=str, default="fedavg", help='fedavg/downscale/median/rfl-self')

args = parser.parse_args()
if args.redirect:
    params=str(vars(args)).replace("'","").replace(" ","").replace(":","#")
    import sys
    sys.stdout = open(params+'.dat', 'w')

DATASET             =  args.dataset
DISTRIBUTION        =  args.distribution
SEED                =  args.seed
CLASSES             =  args.classes
CLIENTS             =  args.clients
LOCAL_EPOCHS        =  args.epochs
LOG                 =  args.log
LOCAL_DATASET_SIZE  =  args.size
COMM_ROUNDS         =  args.rounds
BATCH_SIZE          =  min(args.batch_size, args.size)
LR                  =  args.lr
SELFISH             =  args.selfish
SELFISHNESS         =  args.selfishness
AGGREGATION         =  args.aggregation

create_datasets.set_dataset(DATASET)
set_seed(SEED)


# Initialize models
global_model = CNN(DATASET)
get_ravel_weights(global_model)
prev_global_model = copy.deepcopy(global_model)
models   = [ CNN(DATASET) for _ in range(CLIENTS) ]
prev_models = [ None for _ in models ]

# Download and load the training data
train_dls, test_dls = create_datasets.get_dataset(DISTRIBUTION, n_samples_train=LOCAL_DATASET_SIZE, n_samples_test=100, n_clients=CLIENTS, batch_size=BATCH_SIZE, shuffle=True)

# Train the network
for e in range(COMM_ROUNDS):
    decay = 1-e/COMM_ROUNDS
    # Send global model to clients
    for model in models:
        model.load_state_dict(global_model.state_dict())
    # distributed training
    prev_models = [ copy.deepcopy(model) for model in models ]

    loss = distributed_training(models, train_dls, LOCAL_EPOCHS, lr=LR, global_model=global_model)
    # compute honest vector
    if e > 0:
        for selfish, prev_selfish in zip(models[:SELFISH], prev_models):
            selfish_training(selfish, 
                            global_model=global_model,
                            previous_model=prev_selfish,
                            previous_global_model=prev_global_model, 
                            clients=CLIENTS,
                            selfishness=SELFISHNESS)
    # aggregation
    if AGGREGATION == "fedavg":
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack(
                    [model.state_dict()[k].float() for _, model in enumerate(models)],
                    0).sum(0)/CLIENTS
        prev_global_model = copy.deepcopy(global_model)
        global_model.load_state_dict(global_dict)
    elif AGGREGATION == "downscale":
        new_weights = downscale_aggregation(models, global_model)
        prev_global_model = copy.deepcopy(global_model)
        model_set_weights(global_model, new_weights)
    elif AGGREGATION == "rfl-self":
        new_weights = rotation_aggregation(models, global_model, prev_global_model)
        prev_global_model = copy.deepcopy(global_model)
        model_set_weights(global_model, new_weights)
    elif AGGREGATION == "median":
        new_weights = np.median([get_ravel_weights(model) for model in models], axis=0)
        prev_global_model = copy.deepcopy(global_model)
        model_set_weights(global_model, new_weights)

    if (LOG):
        accuracies = model_accuracy(global_model, test_dls)
        print(f"{e:2d} {loss:.4f}\t", *[f"{a:2.2f}" for a in accuracies],  f"{np.std(accuracies):.2f}", flush=True)

if not LOG:
    accuracies = model_accuracy(global_model, test_dls)
