import math
from collections import defaultdict

import torch
import torch.nn as nn

torch.manual_seed(666)


class Moment:
    def __init__(self):
        self.is_record = False
    pass


def sig(x, derive=False):
    s = torch.sigmoid(x)
    if derive:
        return s * (1 - s)
    else:
        return s


# ----------------------------------------------------------------------------------------------------------------------#
# HISTORY BASE CLASS

class Recorder:
    def __init__(self, default_lambda):
        self.history: dict = defaultdict(default_lambda)  # history

    def at(self, time: int):
        is_record = self.history[time].is_record
        while not is_record:
            time -= 1
            is_record = self.history[time].is_record
            if time < 0 : # first time step is recorded by default!
                self.history[0] = self.history[0]
                self.history[0].is_recorded = True
                return self.history[0]
        return self.history[time]

    def rec(self, time: int):
        self.history[time] = self.history[time]  # new entry (new Moment object)
        self.history[time].is_record = True
        return self.history[time]


# ----------------------------------------------------------------------------------------------------------------------#
# ROUTE

class Route(Recorder):
    def __init__(self, D_in=10, D_out=10):
        super().__init__(default_lambda=lambda: Moment())
        D_h = D_out
        self.Wr = torch.randn(D_in, D_out)
        self.Wr.grad = torch.zeros(D_in, D_out)
        self.Wgh = torch.randn(D_h, 1)
        self.Wgh.grad = torch.zeros(D_h, 1)
        self.Wgr = torch.randn(D_in, 1)
        self.Wgr.grad = torch.zeros(D_in, 1)
        self.pd_g: torch.Tensor = None  # the partial derivative of g

    def get_params(self):
        return [self.Wr, self.Wgr, self.Wgh]

    def forward(self, h: torch.Tensor, r: torch.Tensor, time):
        g = r.matmul(self.Wgr)
        if h is not None: g = h.matmul(self.Wgh) + g
        self.rec(time).pd_g = sig(g, derive=True)
        g = sig(g)
        self.rec(time).g = g.mean().item()  # g is used for routing!
        self.rec(time).z = r.matmul(self.Wr)
        c = g * self.at(time).z
        return c

    def backward(self, e_c, time, r, h):
        self.Wr.grad += r.T.matmul(e_c * self.at(time).g)
        g_r = e_c.matmul(self.Wr.T)
        e_g = e_c.matmul(self.at(time).z.T) * self.at(time).pd_g
        self.Wgr.grad += r.T.matmul(e_g)
        self.Wgh.grad += h.T.matmul(e_g)
        g_r += e_g.matmul(self.Wgr.T)
        g_h = e_g.matmul(self.Wgh.T)
        return (g_h, g_r)



# TESTING:

route = Route(D_in=3, D_out=2)
r = torch.ones(1, 3)
h = torch.ones(1, 2)

c = route.forward(h, r, 0)
assert str(c) == 'tensor([[-3.0741, -0.0344]])'

g_h, g_r = route.backward(
    torch.tensor([[-1.0, 1.0]]), 0,
    h=torch.tensor([[2.0, -3.0]]),
    r=torch.tensor([[-1.0, 4.0, 2.0]])
)
assert str(g_h) == 'tensor([[1.3836, 0.2425]])'
assert str(g_r) == 'tensor([[1.8145, 0.9111, 0.1612]])'

assert str(route.Wgh.grad) == "tensor([[ 1.0678],\n        [-1.6017]])"
assert str(route.Wgr.grad) == "tensor([[-0.5339],\n        [ 2.1356],\n        [ 1.0678]])"
assert str(route.Wr.grad) == "tensor([[ 0.8244, -0.8244],\n        [-3.2974,  3.2974],\n        [-1.6487,  1.6487]])"

del route, r, h, c, g_r, g_h


# ----------------------------------------------------------------------------------------------------------------------#
# SOURCE

def default_source_moment():
    m = Moment()
    m.g = -1

class Source(Recorder):
    def __init__(self, D_in: int, D_out: int):
        super().__init__(default_source_moment())
        self.Ws = torch.randn(D_in, D_out)
        self.Ws.grad = torch.zeros(D_in, D_out)  # Gradients!

    def get_params(self): return [self.Ws]

    def forward(self, s: torch.Tensor):
        return s.matmul(self.Ws)

    def backward(self, e_s: torch.Tensor, s: torch.Tensor):
        print(s.shape)
        print(e_s.shape)
        print(self.Ws.shape)
        self.Ws.grad += s.T.matmul(e_s)
        return e_s.matmul(self.Ws.T)



# TESTING:

source = Source(D_in=2, D_out=3)
s = torch.ones(1, 2)
c = source.forward(s)

assert str(c) == 'tensor([[-0.3784,  0.7585, -0.2807]])'# shape=(1,3)

e = torch.ones(1, 3)
e_c = source.backward(e, s=torch.tensor([[2.0, -1.0]])) #shape=(1,2)

assert str(e_c) == 'tensor([[ 1.2484, -1.1490]])'#shape=(1,2)
assert str(source.Ws.grad) == "tensor([[ 2.,  2.,  2.],\n        [-1., -1., -1.]])"

del s, c, e, e_c


# ----------------------------------------------------------------------------------------------------------------------#
# GROUP

# A moment holds information about a concrete moment in time t!
# This is important for back-propagation
def default_moment(dimensionality: int):
    m = Moment()
    m.state = torch.zeros(1, dimensionality)
    m.is_sleeping = True
    m.derivative = None
    return m


class Group(Recorder):

    def __init__(self, index: int, dimensionality: int):
        super().__init__(default_lambda=lambda: default_moment(dimensionality))
        self.from_conns = dict()
        self.to_conns = dict()
        self.index = index
        self.dimensionality = dimensionality

        self.targets = []  # indices of targeted routes! (routes to groups)
        self.error = None  # the current accumulated error
        self.error_count = 0  # the number of errors summed up within the current accumulated error

    # IO :

    def get_params(self):
        params = []
        for node, route in self.to_conns.items():
            params.extend(route.get_params())
        return params

    def str(self, level: str):
        return level + 'Group: {\n' + \
               level + '   from: ' + str(len(self.from_conns)) + '\n' + \
               level + '   to: ' + str(len(self.to_conns)) + '\n' + \
               level + '   index: ' + str(self.index) + ', ' + 'dimensionality: ' + str(self.dimensionality) + ', ' + \
               'is_sleeping: ' + str(self.is_sleeping) + '\n' + \
               level + '   targets: ' + str(self.targets) + '\n' + \
               level + '   state: ' + str(self.state) + '\n' + \
               level + '};\n'

    # Construction :

    def connect_forward(self, next_groups: list, cone_size: int, step: int):
        assert step > 0
        cone_size = min(cone_size, len(next_groups))
        for i in range(cone_size):
            target_index = (self.index + i * step) % len(next_groups)
            target_group = next_groups[target_index]
            self.targets.append(target_index)
            assert target_group.index == target_index
            self.to_conns[target_group] = Route(self.dimensionality, target_group.dimensionality)
            target_group.register_source(self)

    def register_source(self, origin_group):
        self.from_conns[origin_group] = Source(origin_group.dimensionality, self.dimensionality)

    # Execution :

    def start_with(self, x: torch.Tensor):
        this_is_start = len(self.from_conns) == 0
        assert this_is_start
        self.rec(0).state = x

    def forward(self, time: int):
        assert time >= 0
        this_is_start = len(self.from_conns) == 0

        if not self.at(time).is_sleeping or this_is_start:
            z = None
            if this_is_start:
                z = self.at(time).state  # Start group!
                assert z is not None

            # Source activations :
            for group, source in self.from_conns.items():

                if z is None:
                    z = source.forward(group.at(time).state)
                else:
                    z = z + source.forward(group.at(time).state)

            # Route activations :
            best_target = None
            best_score = 0
            for group, route in self.to_conns.items():

                if z is None:
                    z = route.forward(self.at(time).state, group.at(time).state, time)
                else:
                    z = z + route.forward(self.at(time).state, group.at(time).state, time)

                # Checking if this route is better than another :
                print('Route Gate:',route.at(time).g,'>?>',best_score)
                if route.at(time).g > best_score:
                    best_score = route.at(time).g
                    best_target = group

            if len(self.to_conns.items()) > 0:
                assert best_target is not None  # There has to be a choice!

                # We activate the best group of neurons :
                best_target.rec(time+1).is_sleeping = False  # wake up!
                # No we save the next neuron group which ought to be activated :

            assert z is not None

            if not this_is_start:
                self.rec(time).state = self.activation(z)  # If this is not the start of the network... : Activate!
                self.rec(time).derivative = self.activation(z, derive=True)
            else:
                self.rec(time).state = z

            return best_target  # The best group is being returned!

    def backward(self, time: int):
        current_error: torch.Tensor = self.error

        if not self.at(time).is_sleeping:  # Back-prop only when this group was active at that time!

            # Multiplying with the partial derivative of the activation of this group.
            current_error = current_error * self.at(time).derivative

            # Source (error) bac-prop :
            for group, source in self.from_conns.items():  # Distributing error to source groups...

                g_s = source.backward(
                    e_s=current_error,
                    s=group.at(time).state  # Needed to calculate gradients of the weights!
                )

                if group.error is None:
                    group.error = g_s  # setting or accumulating the error!
                else:
                    group.error = group.error + g_s
                group.error_count += 1

            # Route (error) back-prop :
            for group, route in self.to_conns.items():  # Distributing error to route groups...

                g_h, g_r = route.backward(
                    e_c=current_error,
                    time=time,
                    r=route.at(time).state,  # Needed to calculate gradients of the weights!
                    h=self.at(time).state  # Needed to calculate gradients of the weights!
                )

                if group.error is None:
                    group.error = g_r  # setting or accumulating the error!
                else:
                    group.error = group.error + g_r
                group.error_count += 1

                # Accumulating g_h to self:
                if self.error is None:
                    self.error = g_h
                else:
                    self.error = self.error + g_h
                # self.error_count += 1

            # Source Group backprop :
            for group, source in self.from_conns.items():
                group.backward(time - 1)

            # Route Groupe backprop :
            for group, route in self.to_conns.items():
                group.backward(time - 1)

    def activation(self, x, derive=False):  # State of the Art activation function, SWISH
        if derive:
            return torch.sigmoid(x)
        else:
            return torch.sigmoid(x) * x
        # return x * torch.tanh(nn.functional.softplus(x)) # MISH might also be interesting


# TESTING:

group = Group(index=0, dimensionality=3)

assert group.index == 0
assert group.dimensionality == 3
assert group.from_conns != None
assert group.to_conns != None
assert group.targets == []  # indices of targeted routes!
assert group.error == None
assert group.error_count == 0
assert group.at != None

other1 = Group(index=0,dimensionality=3)
other2 = Group(index=1,dimensionality=3)

group.connect_forward(next_groups=[other1,other2], cone_size=293943, step=1)

assert len(group.from_conns) == 0
assert len(group.to_conns) == 2
assert len(other1.from_conns) == 1
assert len(other2.from_conns) == 1
assert len(other1.to_conns) == 0
assert len(other2.to_conns) == 0

group.start_with(torch.tensor([[1.0, 2.0, 3.0]]))
groups = [group, other1, other2]
for g in groups: g.forward(0)

assert not other1.at(1).is_sleeping
assert other2.at(1).is_sleeping

assert [r.at(0).g for r in group.to_conns.values()] == [0.9598161578178406, 0.5216401815414429]

del group


# ----------------------------------------------------------------------------------------------------------------------#
# CAPSULE

class Capsule:

    def __init__(self, dimensionality, size: int):
        self.groups = []
        for i in range(size):  # creating "size" number of groups for this capsule
            self.groups.append(Group(i, dimensionality))

    # IO :

    def get_params(self):
        params = []
        for group in self.groups:
            params.extend(group.get_params())
        return params

    def str(self, level):
        asStr = level + 'Capsule: {\n'
        asStr += (level + '   height: ' + str(len(self.groups)) + '\n')
        for group in self.groups:
            asStr += group.str(level + '   ')
        asStr += ('\n' + level + '};\n')
        return asStr

    # Construction :

    def connect_forward(self, next_capsule, max_cone, step):
        for group in self.groups:
            group.connect_forward(next_capsule.groups, max_cone, step)

    # Execution :

    def start_with(self, x):
        # This is only allowed to be called
        # on first capsule of the network !
        assert len(self.groups) == 1
        for group in self.groups: group.start_with(x)


# ----------------------------------------------------------------------------------------------------------------------#
# Network

class Network:

    def __init__(self,
                 depth=5,
                 max_height=15,
                 max_dim=500,
                 max_cone=39,
                 D_in=100,
                 D_out=10,
                 ):
        assert depth > 3
        dims = [math.floor(float(x)) for x in self.girth_from(depth, D_in, max_dim, D_out)]
        heights = [math.floor(float(x)) for x in self.girth_from(depth, 1, max_height, 1)]
        self._capsules = []

        for i in range(depth):
            self._capsules.append(Capsule(dims[i], heights[i]))

        for i in range(depth):
            if i != depth - 1:
                assert len(self._capsules[i].groups) * max_cone >= len(self._capsules[i + 1].groups)
                current_cone = min(len(self._capsules[i + 1].groups), max_cone)
                step = max(1, int(len(self._capsules[i + 1].groups) / current_cone))
                self._capsules[i].connect_forward(self._capsules[i + 1], current_cone, step)

    def str(self):
        asStr = ''
        asStr += 'Network: {\n'
        asStr += '   length: '
        asStr += str(len(self._capsules)) + '\n'
        level = '   '
        for capsule in self._capsules:
            asStr += capsule.str(level)
        asStr += '\n};\n'
        return asStr

    def girth_from(self, length, start, max, end):
        g, mid = [], (length - 1) / 2
        for i in range(length):
            if math.fabs(mid - i) < 1:
                g.append(max)
            else:
                if i < mid:
                    ratio = i / mid
                    g.append((max * ratio) + start * (1 - ratio))
                else:
                    ratio = (i - mid) / mid
                    g.append((end * ratio) + max * (1 - ratio))
        return g


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTING :

net = Network(
    depth=5,
    max_height=15,
    max_dim=500,
    max_cone=8,
    D_in=100,
    D_out=10,
)
# print(net.str())

expected_structure = [
    {
        'height': 1, 'dimensionality': 100, 'from': 0, 'to': 8,
        'targets': [[0, 1, 2, 3, 4, 5, 6, 7]],
        'index': [0]
    },
    {
        'height': 8, 'dimensionality': 300, 'from': 1, 'to': 8,
        'targets': [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
            [3, 4, 5, 6, 7, 8, 9, 10],
            [4, 5, 6, 7, 8, 9, 10, 11],
            [5, 6, 7, 8, 9, 10, 11, 12],
            [6, 7, 8, 9, 10, 11, 12, 13],
            [7, 8, 9, 10, 11, 12, 13, 14],
        ],
        'index': [0, 1, 2, 3, 4, 5, 6, 7]
    },
    {
        'height': 15, 'dimensionality': 500, 'from': [1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1],
        'to': 8,
        'targets': [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0],
            [2, 3, 4, 5, 6, 7, 0, 1],
            [3, 4, 5, 6, 7, 0, 1, 2],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 6, 7, 0, 1, 2, 3, 4],
            [6, 7, 0, 1, 2, 3, 4, 5],
            [7, 0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0],
            [2, 3, 4, 5, 6, 7, 0, 1],
            [3, 4, 5, 6, 7, 0, 1, 2],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 6, 7, 0, 1, 2, 3, 4],
            [6, 7, 0, 1, 2, 3, 4, 5]
        ],
        'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    },
    {
        'height': 8, 'dimensionality': 255, 'from': 15, 'to': 1,
        'targets': [[0], [0], [0], [0], [0], [0], [0], [0]],
        'index': [0]
    },
    {
        'height': 1, 'dimensionality': 10, 'from': 8, 'to': 0,
        'targets': [[]],
        'index': [0]
    }
]

assert len(net._capsules) == len(expected_structure)

for ci in range(len(net._capsules)):
    expected_capsule = expected_structure[ci]
    given_capsule = net._capsules[ci]
    assert expected_capsule['height'] == len(given_capsule.groups)
    for gi in range(len(given_capsule.groups)):
        given_group = given_capsule.groups[gi]
        assert given_group.targets == expected_capsule['targets'][gi]
        if isinstance(expected_capsule['from'], list):
            assert len(given_group.from_conns) == expected_capsule['from'][gi]
        else:
            assert len(given_group.from_conns) == expected_capsule['from']
        assert len(given_group.to_conns) == expected_capsule['to']

net = Network(
    depth=4,
    max_height=3,
    max_dim=500,
    max_cone=8,
    D_in=100,
    D_out=10,
)
# print(net.str())
