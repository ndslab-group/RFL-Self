from CNN import CNN
import torch.optim  as optim
import torch.nn  as nn
import torch
import os
import sys
import random
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_accuracy(model, datasets):
    # Compute global model accuracy for each client's test dataset
    model.to(device=device)
    model.eval()
    accuracies = []
    for i in range(len(datasets)):
        num_correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in datasets[i]:
                data = data.to(device=device)
                labels = labels.to(device=device)
                predictions = torch.max(model(data),1)[1].data.squeeze()
                num_correct += (predictions == labels).sum()
                total += labels.size(0)
        if total != 0:
            accuracies.append(float(num_correct)/float(total)*100)
    return accuracies

def model_loss(model, dataset):
    with torch.no_grad():
        model.to(device=device)
        criterion = nn.CrossEntropyLoss()
        epoch_loss_collector = []
        running_loss = 0
        for images, labels in dataset:
                images = images.to(device=device)
                labels = labels.to(device=device)
                output = model.forward(images)
                loss = criterion(output, labels)
                running_loss += loss.item()
                epoch_loss_collector.append(loss.item())
    return sum(epoch_loss_collector)/len(epoch_loss_collector)


def log_accuracies(models, global_model, test_dls, verbose=False, volatile=False):
    # Compute the accuracy of both the global model and the local models
    l_acc = [ model_accuracy(model, [test_dls[i]])        for i, model in enumerate(models)  ]
    g_acc = [ model_accuracy(global_model, [test_dls[i]]) for i        in range(len(models)) ]
    if verbose:
        print(f"local accuracy:  {np.mean(l_acc):.2f} ±{np.std(l_acc):.2f}", end="\t")
        print(f"global accuracy: {np.mean(g_acc):.2f} ±{np.std(g_acc):.2f}", end="\r" if volatile else "\n")
        sys.stdout.flush()
    return [np.mean(g_acc), np.std(g_acc)]

def distributed_training(models, datasets, epochs, lr, global_model=None):
    # Define loss function
    criterion = nn.CrossEntropyLoss()

    epoch_loss_collector = []
    #Train each client on its local dataset
    for i in range(len(models)):
        models[i].train()
        models[i].to(device=device)
        running_loss = 0
        for _ in range(epochs):
            for images, labels in datasets[i]:
                images = images.to(device=device)
                labels = labels.to(device=device)
                optimizer = optim.SGD(models[i].parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                # Training pass
                optimizer.zero_grad()
                output = models[i].forward(images)
                loss = criterion(output, labels)
                loss.backward()
                running_loss += loss.item()
                optimizer.step()
                epoch_loss_collector.append(loss.item())
    return sum(epoch_loss_collector)/(len(epoch_loss_collector)+0.000000001)

# Weights manipulation
def get_ravel_weights(model):
    ww = []
    for par in model.parameters():
        ww.append(par.detach().cpu().data.numpy().ravel())
    return np.concatenate(ww)


def model_set_weights(xmodel, xweights):
    offset = 0
    for name, parameter in xmodel.named_parameters():
        size = np.prod(parameter.size())
        value = xweights[offset:offset+size].reshape(parameter.size())
        parameter.data.copy_(torch.from_numpy(value))
        offset += size