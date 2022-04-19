
import torch

# ----------------------------------------------------------------------------------------------------------------------#


# A useful method used later on...
def sig(x, derive=False):
    s = torch.sigmoid(x)
    if derive: return s * (1 - s)
    else: return s


def swish(x, derive=False):
    s = sig(x) * x # return sig(x, derive)
    if derive: return s + sig(x) * (1-s)
    else: return s


def activation(x, derive=False):  # State of the Art activation function, SWISH
    #return torch.relu(x)
    return mish(x, derive=derive)
    #return swish(x, derive=derive)


def mish(x, derive=False):
    sfp = torch.nn.Softplus()
    if derive:
        omega = torch.exp(3 * x) + 4 * torch.exp(2 * x) + (6 + 4 * x) * torch.exp(x) + 4 * (1 + x)
        delta = 1 + ( (torch.exp(x) + 1)**2 )
        return torch.exp(x) * omega / (delta**2)
    else:
        return x * torch.tanh(sfp(x))


def gaus(x, mean=0, std=0.5):
    return torch.exp((-(x - mean) ** 2)/(2* std ** 2))


def tanh(x):
    return torch.tanh(x)


def gatu(x):
    cube = x * x * x
    return tanh(cube)


def gasu(x):
    cube = x * x * x
    return cube / (1 + torch.abs(cube))

def none(x):
    return x




