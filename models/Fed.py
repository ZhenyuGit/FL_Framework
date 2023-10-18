import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg_weight(w,weight):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k]=w_avg[k]*weight[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]*weight[i]
    return w_avg

def weight_minus(w1,w2):
    w_m = {"layer_input.weight":[],"layer_input.bias":[],
           "layer_hidden.weight":[],"layer_hidden.bias":[]}

    w_m["layer_input.weight"]=w1['layer_input.weight']-w2['layer_input.weight']
    w_m["layer_input.bias"]=w1['layer_input.bias']-w2['layer_input.bias']
    w_m["layer_hidden.weight"]=w1['layer_hidden.weight']-w2['layer_hidden.weight']
    w_m["layer_hidden.bias"]=w1['layer_hidden.bias']-w2['layer_hidden.bias']
    return w_m

def weight_scalar_mul(w,scalar):
    w_mul = {"layer_input.weight":[],"layer_input.bias":[],
           "layer_hidden.weight":[],"layer_hidden.bias":[]}
    w_mul["layer_input.weight"] = w['layer_input.weight']/scalar
    w_mul["layer_input.bias"] = w['layer_input.bias'] / scalar
    w_mul["layer_hidden.weight"] = w['layer_hidden.weight'] / scalar
    w_mul["layer_hidden.bias"] = w['layer_hidden.bias'] / scalar
    return w_mul

def weight_tovec(w):
    w_tovec = []
    for k, v in w.items():
        value = v
        for item in value:
            w_tovec.append(value)
    return w_tovec


