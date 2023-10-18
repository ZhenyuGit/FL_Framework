import torch
import numpy as np
from torchvision import datasets, transforms
import random
from PIL import Image


def noisy_label_change_client( dict_users, dataset, noisy_client, noisy_rate):
    """
    change correct label into noisy label
    dataName:'MNIST' or 'cifar'
    dict_users:The data assigned to each user
    dataset:All the data from the corresponding dataset
    noisy_client:The number of users with noisy labels
    noisy_rate:The proportion of incorrect data for each user

    Replace the labels of a portion of clients' data with incorrect labels, selecting the first k clients
    No return value; it directly modifies the original dataset. Therefore, if you need the original
    dataset after the call, you will need to read it again
    """
    originTargets = dataset.train_labels.numpy()

    allorigin_targets = set(originTargets)

    for i in np.arange(len(noisy_client)):
        if noisy_client[i] > len(dict_users):
            print('too many noisy client')
            raise NameError('noisy_client')
            exit()
    noisyDataList = [[]]*10

    noisy_client[0]=[0]
    noisy_client[1]=[1]
    noisy_client[2]=[2]
    noisy_client[3]=[3]
    noisy_client[4]=[4]
    noisy_client[5]=[5]
    noisy_client[6]=[6]
    noisy_client[7]=[7]
    noisy_client[8]=[8]
    noisy_client[9]=[9]


    for j in np.arange(len(noisy_client)):
        for userIndex in noisy_client[j]:

            noisyDataList[j].extend(list(
                np.random.choice(list(dict_users[userIndex]), int(len(dict_users[userIndex]) * noisy_rate[j]), replace=False)))

    for s in np.arange(len(noisyDataList)):
        for index in noisyDataList[s]:
            all_targets = allorigin_targets
            all_targets = all_targets - set([originTargets[index]])
            new_label = np.random.choice(list(all_targets), 1, replace=False)
            originTargets[index] = new_label[0]
    dataset.targets = torch.tensor(originTargets)
    return dataset, noisyDataList,torch.tensor(originTargets)
