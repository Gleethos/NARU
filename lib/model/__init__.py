import torch

class Loss:

    def __init__(self):
        self.mse = torch.nn.MSELoss()
        self.loss = 0

    def __call__(self, tensor : torch.Tensor, target : torch.Tensor):
        clone = tensor.clone()
        clone.requires_grad = True
        self.loss = self.mse(input=clone, target=target)
        self.loss.backward()
        self.loss = self.loss.item()
        return clone.grad