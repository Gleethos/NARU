
import torch
import os
import errno
from os import listdir
from os.path import isfile, join


def save_params(params: list, folder: str):
    if not os.path.exists(os.path.dirname(folder)):
        try:
            os.makedirs(os.path.dirname(folder))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    #for i, tensor in enumerate(params):
    #    filename = folder+'parameter_'+str(i)+'.pt'
    #    if os.path.exists(filename):
    #        os.remove(filename)

    for i, tensor in enumerate(params):
        filename = folder+'parameter_'+str(i)+'.pt'
        torch.save(tensor, filename)


def load_params(folder: str):
    if os.path.exists(os.path.dirname(folder)):
        params = []
        only_files = [f for f in listdir(folder) if isfile(join(folder, f))]
        for i, file in enumerate(only_files):
            tensor = torch.load(folder+'parameter_'+str(i)+'.pt')
            params.append(tensor)
        return params