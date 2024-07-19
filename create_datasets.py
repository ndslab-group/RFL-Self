import torch
from torchvision import datasets
from torchvision import transforms

DATASET=datasets.CIFAR10
DOWNLOAD=False

def set_dataset(dataset):
    global DATASET
    if dataset == "mnist":
        DATASET=datasets.MNIST
    elif dataset == "cifar10":
        DATASET=datasets.CIFAR10


def non_iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle, shuffle_digits=True):
    assert(nb_nodes>0)
    splits= max(10, 2)

    digits=(torch.arange(splits) if shuffle_digits==False else torch.randperm(splits, generator=torch.Generator().manual_seed(0))) % 10

    # split the digits in a fair way
    digits_split=list()
    i=0
    for n in range(nb_nodes, 0, -1):
        inc=int((splits-i)/n)
        digits_split.append(digits[i:i+inc])
        i+=inc

    # load and shuffle nb_nodes*n_samples_per_node from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=int(nb_nodes*n_samples_per_node),
                                        shuffle=shuffle, drop_last=False)
    dataiter = iter(loader)
    images_train_mnist, labels_train_mnist = dataiter._next_data()

    data_splitted=list()
    for i in range(nb_nodes):
        idx=torch.stack([y_ == labels_train_mnist for y_ in digits_split[i]]).sum(0).bool() # get indices for the digits
        to_remove = idx.nonzero()[:n_samples_per_node]
        data_splitted.append(
                torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(images_train_mnist[idx][:n_samples_per_node],
                                                   labels_train_mnist[idx][:n_samples_per_node]
                    ),
                    batch_size=batch_size, 
                    shuffle=shuffle, 
                    drop_last=False
                )
        )
    return data_splitted



def iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle):
    # load and shuffle n_samples_per_node from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=n_samples_per_node,
                                        shuffle=shuffle,
                                        drop_last=False)
    dataiter = iter(loader)
    
    data_splitted=list()
    for _ in range(nb_nodes):
        data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(dataiter._next_data())), batch_size=batch_size, shuffle=shuffle, drop_last=False))

    return data_splitted

def  get_dataset(type="iid", n_samples_train=200, n_samples_test=100, n_clients=3, batch_size=25, shuffle=True):
    dataset_loaded_train = DATASET(
            root="./data",
            train=True,
            download=DOWNLOAD,
            transform=transforms.Compose(
                ([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010)
                        )
                    if DATASET==datasets.CIFAR10 else
                        transforms.Normalize(mean = (0.1307,), std = (0.3081,))
                ] 
                 )
            )
    )
    dataset_loaded_test = DATASET(
            root="./data",
            train=False,
            download=DOWNLOAD,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010)
                        )
                    if DATASET==datasets.CIFAR10 else
                        transforms.Normalize(mean = (0.1307,), std = (0.3081,))
                ])
    )

    if type=="iid":
        train=iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
        test=iid_split(dataset_loaded_test, n_clients, n_samples_test, n_samples_test, shuffle)
    elif type=="non_iid":
        train=non_iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
        test=non_iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)
    else:
        train=[]
        test=[]

    return train, test
