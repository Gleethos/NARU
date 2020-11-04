import math

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------------------------------------------------#
# ROUTE

class Route(nn.Module):
    def __init__(self, D_h=None, D_in=10, D_out=10):
        super(Route, self).__init__()
        if D_h is None: D_h = D_in
        self._x_lin = torch.randn(D_out, D_in)  # nn.Linear(D_in, D_out, bias=False)
        self._h_dot = torch.randn(D_h, 1)  # nn.Linear(D_h,  1,     bias=False)
        self._x_dot = torch.randn(D_in, 1)  # nn.Linear(D_in, 1,     bias=False)
        self._g = 0

    def forward(self, h, x):
        g = x.matmul(self._x_dot)  # self._x_dot(x)
        if h is not None: g = h.matmul(self._h_dot) + g  # self._h_dot(h) + g
        g = torch.sigmoid(g)
        self._g = g.mean().item()
        y = g * x.matmul(self._x_lin)  # g * self._x_lin(x)
        return y

    def candidness(self):
        return self._g


# ----------------------------------------------------------------------------------------------------------------------#
# SOURCE

class Source(nn.Module):
    def __init__(self, D_in, D_out):
        super(Source, self).__init__()
        self.weight = torch.randn(D_out, D_in)  # torch.nn.Parameter(torch.randn(D_out, D_in))
        # self._x_lin = nn.Linear(D_in, D_out, bias=False)

    def forward(self, x):
        # if 2 in [p._version for p in self._x_lin.parameters()]:
        #    print('Version change in "Source":', [p._version for p in self._x_lin.parameters()])
        # return self._x_lin(x)
        return x.matmul(self.weight)  # self.weight.t()

    def candidness(self): return -1


# ----------------------------------------------------------------------------------------------------------------------#
# GROUP

class Group:

    def __init__(self, index, dimensionality):
        self._from = dict()
        self._to = dict()
        self._index = index
        self._dimensionality = dimensionality
        self._is_sleeping = False
        self._state = None
        self._targets = [] # indices of targeted routes!

    # IO :

    def index(self):
        return self._index

    def dimensionality(self):
        return self._dimensionality

    def is_woke(self):
        return not self._is_sleeping

    def fall_asleep(self):
        self._is_sleeping = True

    def wake_up(self):
        self._is_sleeping = False

    def state(self):
        return self._state

    def str(self, level):
        return level + 'Group: {\n' + \
               level + '   from: ' + str(len(self._from)) + '\n' + \
               level + '   to: ' + str(len(self._to)) + '\n' + \
               level + '   index: ' + str(self._index) + ', ' + 'dimensionality: ' + str(self._dimensionality) + ', ' + \
               'is_sleeping: ' + str(self._is_sleeping) + '\n' + \
               level + '   targets: ' + str(self._targets) + '\n' + \
               level + '   state: ' + str(self._state) + '\n' + \
               level + '};\n'

    # Construction :

    def connect_forward(self, next_groups, cone_size, step):
        cone_size = min(cone_size, len(next_groups))
        for i in range(cone_size):
            target_index = (self.index() + i * step) % len(next_groups)
            target_group = next_groups[target_index]
            self._targets.append(target_index)
            assert target_group.index() == target_index
            self._to[target_group] = Route(self.dimensionality(), target_group.dimensionality())
            target_group.register_source(self)

    def register_source(self, origin_group):
        self._from[origin_group] = Source(origin_group.dimensionality(), self.dimensionality())

    # Execution :

    def start_with(self, x):
        this_is_start = len(self._from) == 0
        assert this_is_start
        self._state = x

    def forward(self):

        this_is_start = len(self._from) == 0

        if self.is_woke() or this_is_start:
            z = None
            if this_is_start:
                z = self._state  # Start group!
                assert z is not None

            # Source activations :
            for group, source in self._from.items():
                if z is None:
                    z = source(group.state())
                else:
                    z = z + source(group.state())

            # Route activations :
            best_target = None
            best_score = 0
            for group, route in self._to.items():
                if z is None:
                    z = route(group.state(), self.state())
                else:
                    z = z + route(group.state(), self.state())
                if route.candidness() > best_score:
                    best_score = route.candidness
                    best_target = group

            assert best_target is not None
            best_target.wake_up()

            assert z is not None

            if not this_is_start:
                self._state = self.mish(z)

    def mish(self, x):  # State of the Art activation function
        return x * torch.tanh(nn.functional.softplus(x))


# ----------------------------------------------------------------------------------------------------------------------#
# CAPSULE

class Capsule:

    def __init__(self, dimensionality, size):
        self._groups = []
        for i in range(size):
            self._groups.append(Group(i, dimensionality))

    # IO :

    def groups(self):
        return self._groups

    def str(self, level):
        asStr = level + 'Capsule: {\n'
        asStr += (level+'   length: ' + str(len(self._groups)) + '\n')
        for group in self._groups:
            asStr += group.str(level + '   ')
        asStr += ('\n' + level + '};\n')
        return asStr

    # Construction :

    def connect_forward(self, next_capsule, max_cone, step):
        for group in self._groups:
            group.connect_forward(next_capsule.groups(), max_cone, step)

    # Execution :

    def forward(self):
        for group in self._groups:
            group.forward()

    def start_with(self, x):
        # This is only allowed to be called
        # on first capsule of the network !
        assert len(self._groups) == 1
        for group in self._groups: group.start_with(x)


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
                assert len(self._capsules[i]._groups) * max_cone >= len(self._capsules[i + 1]._groups)
                current_cone = min(len(self._capsules[i + 1]._groups), max_cone)
                step = max(1, int(len(self._capsules[i + 1]._groups) / current_cone))
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


# TESTING :


net = Network(
    depth=5,
    max_height=15,
    max_dim=500,
    max_cone=8,
    D_in=100,
    D_out=10,
)

print(net.str())

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
assert len(net._capsules[0]._groups) == 1

assert len(net._capsules) == len(expected_structure)

for ci in range(len(net._capsules)) :
    expected_capsule = expected_structure[ci]
    given_capsule = net._capsules[ci]
    assert expected_capsule['height'] == len(given_capsule._groups)
    for gi in range(len(given_capsule._groups)):
        given_group = given_capsule._groups[gi]
        assert given_group._targets == expected_capsule['targets'][gi]
        if isinstance(expected_capsule['from'],list):
            assert len(given_group._from) == expected_capsule['from'][gi]
        else :
            assert len(given_group._from) == expected_capsule['from']
        assert len(given_group._to) == expected_capsule['to']

