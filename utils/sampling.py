#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users, num_items):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_imbalanced_datasize(dataset, num_users, num_totalexsamples):
    """Sample I.I.D. client data from MNIST dataset
    allocate different clients with datasets of different sizes"""

    dict_users, all_data = {}, [i for i in range(len(dataset))]
    all_idxs = list(np.random.choice(all_data, num_totalexsamples, replace=False))
    num_foreach = [100,300,500,700,900,1100,1300,1500,1700,1900]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_foreach[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_imbalanced_classnumbers(dataset, num_users):
    dict_users, all_data = {}, [i for i in range(len(dataset))]
    data_classified = [[],[],[],[],[],[],[],[],[],[]]
    labels = dataset.train_labels.numpy()
    for i in range(len(labels)):
        data_classified[labels[i]].append(i)
    dict_users[0] = set(np.random.choice(data_classified[0], 1000, replace=False))
    dict_users[1] = set(np.random.choice(data_classified[0], 500, replace=False))\
                    |set(np.random.choice(data_classified[1], 500, replace=False))
    dict_users[2] = set(np.random.choice(data_classified[0], 333, replace=False))\
                    |set(np.random.choice(data_classified[1], 333, replace=False))\
                    |set(np.random.choice(data_classified[2], 334, replace=False))
    dict_users[3] = set(np.random.choice(data_classified[0], 250, replace=False))\
                    |set(np.random.choice(data_classified[1], 250, replace=False))\
                    |set(np.random.choice(data_classified[2], 250, replace=False))\
                    |set(np.random.choice(data_classified[3], 250, replace=False))
    dict_users[4] = set(np.random.choice(data_classified[0], 200, replace=False))\
                    |set(np.random.choice(data_classified[1], 200, replace=False))\
                    |set(np.random.choice(data_classified[2], 200, replace=False))\
                    |set(np.random.choice(data_classified[3], 200, replace=False))\
                    |set(np.random.choice(data_classified[4], 200, replace=False))
    dict_users[5] = set(np.random.choice(data_classified[0], 167, replace=False)) \
                    | set(np.random.choice(data_classified[1], 167, replace=False)) \
                    | set(np.random.choice(data_classified[2], 167, replace=False)) \
                    | set(np.random.choice(data_classified[3], 167, replace=False)) \
                    | set(np.random.choice(data_classified[4], 166, replace=False))\
                    | set(np.random.choice(data_classified[5], 166, replace=False))
    dict_users[6] = set(np.random.choice(data_classified[0], 143, replace=False)) \
                    | set(np.random.choice(data_classified[1], 143, replace=False)) \
                    | set(np.random.choice(data_classified[2], 143, replace=False)) \
                    | set(np.random.choice(data_classified[3], 143, replace=False)) \
                    | set(np.random.choice(data_classified[4], 143, replace=False)) \
                    | set(np.random.choice(data_classified[5], 143, replace=False)) \
                    | set(np.random.choice(data_classified[6], 143, replace=False))
    dict_users[7] = set(np.random.choice(data_classified[0], 125, replace=False)) \
                    | set(np.random.choice(data_classified[1], 125, replace=False)) \
                    | set(np.random.choice(data_classified[2], 125, replace=False)) \
                    | set(np.random.choice(data_classified[3], 125, replace=False)) \
                    | set(np.random.choice(data_classified[4], 125, replace=False)) \
                    | set(np.random.choice(data_classified[5], 125, replace=False)) \
                    | set(np.random.choice(data_classified[6], 125, replace=False))\
                    | set(np.random.choice(data_classified[7], 125, replace=False))
    dict_users[8] = set(np.random.choice(data_classified[0], 112, replace=False)) \
                    | set(np.random.choice(data_classified[1], 111, replace=False)) \
                    | set(np.random.choice(data_classified[2], 111, replace=False)) \
                    | set(np.random.choice(data_classified[3], 111, replace=False)) \
                    | set(np.random.choice(data_classified[4], 111, replace=False)) \
                    | set(np.random.choice(data_classified[5], 111, replace=False)) \
                    | set(np.random.choice(data_classified[6], 111, replace=False)) \
                    | set(np.random.choice(data_classified[7], 111, replace=False))\
                    | set(np.random.choice(data_classified[8], 111, replace=False))
    dict_users[9] = set(np.random.choice(data_classified[0], 100, replace=False)) \
                    | set(np.random.choice(data_classified[1], 100, replace=False)) \
                    | set(np.random.choice(data_classified[2], 100, replace=False)) \
                    | set(np.random.choice(data_classified[3], 100, replace=False)) \
                    | set(np.random.choice(data_classified[4], 100, replace=False)) \
                    | set(np.random.choice(data_classified[5], 100, replace=False)) \
                    | set(np.random.choice(data_classified[6], 100, replace=False)) \
                    | set(np.random.choice(data_classified[7], 100, replace=False)) \
                    | set(np.random.choice(data_classified[8], 100, replace=False)) \
                    | set(np.random.choice(data_classified[9], 100, replace=False))
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/MNIST/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
