import torch
import numpy as np

NMAX = 7
LMAX = 4

SUBSHELLS={
    's':0,
    'p':1,
    'd':2,
    'f':3
}


import json
def load_element_json(path):
    with open(path) as infile:
        data = infile.read()
    jsondata = json.loads(data)
    return jsondata['elements']

def get_electron_configuration(elementdict, i):
    config_str = elementdict[i]['electron_configuration']
    config = torch.zeros(NMAX, LMAX)
    for substr in config_str.split(" "):
        n = int(substr[0])-1
        l = SUBSHELLS[substr[1]]
        occ = int(substr[2:])
        config[n, l] = occ
    return config

def get_valence_electron_configuration(elementdict, i):
    config = get_electron_configuration(elementdict,i)
    config_str = elementdict[i]['electron_configuration_semantic']
    if len(config_str.split(" ")) == 1:
        valence_config = config
        core_config = torch.zeros(NMAX, LMAX)
    else:
        core_config = config
        valence_config = torch.zeros(NMAX, LMAX)
        for substr in config_str.split(" ")[1:]:
            n = int(substr[0])-1
            l = SUBSHELLS[substr[1]]
            occ = int(substr[2:])
            valence_config[n, l] = occ
            core_config[n,l] = 0
    return core_config, valence_config

def build_element_config_tensors(elementdict):
    n_elements = len(elementdict)
    element_tensor = torch.zeros(n_elements, NMAX, LMAX)
    for i in range(n_elements):
        element_tensor[i,:,:] = get_electron_configuration(elementdict, i)
    return element_tensor.view(n_elements, -1)

def build_element_valence_config_tensors(elementdict):
    n_elements = len(elementdict)
    element_core_tensor = torch.zeros(n_elements, NMAX, LMAX)
    element_valence_tensor = torch.zeros(n_elements, NMAX, LMAX)
    for i in range(n_elements):
        core, valence = get_valence_electron_configuration(elementdict, i)
        element_core_tensor[i,:,:] = core
        element_valence_tensor[i,:,:] = valence
    return element_core_tensor.view(n_elements, -1), element_valence_tensor.view(n_elements, -1)