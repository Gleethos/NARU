import torch

from lib.model.comps.fun import mish

class Loss:

    def __init__(self):
        self.mse = torch.nn.MSELoss()
        self.loss = 0

    def __call__(self, tensor: torch.Tensor, target: torch.Tensor):
        clone = tensor.clone()
        clone.requires_grad = True
        self.loss = self.mse(input=clone, target=target)
        self.loss.backward()
        self.loss = self.loss.item()
        return clone.grad


class Settings:

    def __init__(self, route=None, activation=None):
        from lib import Route
        if route is None:
            self.route = Route
        else:
            self.route = route

        if activation is None :
            self.activation = mish
        else:
            self.activation = activation
