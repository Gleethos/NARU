import torch
import torch.nn as nn
from utility.lara_structures import create
torch.autograd.set_detect_anomaly(True)

class LARANet(nn.Module):

    def __init__(self, D_in, D_out):
        super(LARANet, self).__init__()
        self._sub_modules = torch.nn.ModuleList([])
        depth = 4
        self._struc = create(depth=depth, max_dim=100, max_paths=5, D_in=D_in, D_out=D_out)
        for layer in self._struc["layers"]:
            for node in layer:
                for route in node["routes"]:
                    self._sub_modules.append(route["route"])
                for source in node["sources"]:
                    self._sub_modules.append(source["source"])

    def _route(self, z, layers, node, best, li):
        for route in node["routes"]:
            target = layers[li + 1][route["target_index"]]
            if target["state"] is not None:

                if z is None:
                    z = route["route"](node["state"], target["state"])
                else:
                    z = z + route["route"](node["state"], target["state"])

                if route["route"].candidness() > best["score"]:
                    best["score"] = route["route"].candidness()
                    best["target"] = target
            else:
                best["target"] = target
                best["score"] = 1  # <- All nodes must have been active at least once!
        return z

    def _source(self, z, node, li):
        for source in node["sources"]:
            target = self._struc["layers"][li - 1][source["target_index"]]
            #if 2 in [p._version for p in source["source"].parameters()] :
            #    print('L'+str(li), [p._version for p in source["source"].parameters()])
            if target["state"] is not None:
                if z is None:
                    z = source["source"](target["state"])
                else:
                    z = z + source["source"](target["state"])

        return z

    def forward(self, x):
        layers = self._struc["layers"]
        layers[0][0]["state"] = x
        layers[0][0]["active"] = True
        for li in range(len(layers)):
            layer = layers[li]
            for ni in range(len(layer)):
                node = layer[ni]

                if node["active"]:
                    best, z = {"target": None, "score": -10}, None
                    if li == 0: z = x

                    z = self._route(z=z, layers=layers, node=node, best=best, li=li)

                    if li < len(layers) - 1: best["target"]["active"] = True

                    print("Activation at: L[" + str(li) + "]N[" + str(ni) + "]; Best: S[" + str(best["score"]) + "]")

                    z = self._source(z=z, node=node, li=li)

                    if z is not None: node["state"] = torch.nn.functional.softplus(z)
                    node["active"] = False  # Reset activity (new target has been chosen...)

        return layers[len(layers) - 1][0]["state"]

    def detach_states(self):
        layers = self._struc["layers"]
        for li in range(len(layers)):
            layer = layers[li]
            for ni in range(len(layer)):
                node = layer[ni]
                if node['state'] is not None :
                    node['state'] = node['state'].clone()#.detach() # Fixes in-place operation error called by backprop

torch.manual_seed(842)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, D_out = 64, 1000, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = LARANet(D_in, D_out)

# model.to("cuda")

i = 0
for parameter in model.parameters(): i += 1
print(i)
assert i == 92

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

for t in range(10):  # range(750):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x.clone())

    # Compute and print loss
    loss = criterion(y_pred.clone(), y.clone())
    # if t % 100 == 99:
    print(t, loss.item())
    # assert loss.item()==expected[i]
    i += 1

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    #for p in model.parameters() : p.detach()
    optimizer.step()
    model.detach_states()


