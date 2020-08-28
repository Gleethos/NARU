import torch
import torch.nn.functional as F
from utility.lara_modules import Route

torch.manual_seed(842)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, D_out = 64, 1000, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = Route(D_in=D_in, D_out=D_out)

i = 0
for parameter in model.parameters(): i += 1
assert i == 3

expected = [
    0.000969238462857902,
    1.2204567911844322e-10,
    7.205314470071222e-12,
    3.4170290295204575e-12,
    2.5288070248929984e-12,
    2.3300549857607766e-12,
    1.810662405600516e-12
]

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
i = 0
for t in range(750):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x, x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
        delta = abs(expected[i] - loss.item())
        assert delta < 0.00000001
        i += 1

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
