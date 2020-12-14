import math
from collections import defaultdict

import torch
import torch.nn as nn
torch.manual_seed(666)

class Moment : pass

def sig(x, derive=False) :
    s = torch.sigmoid(x)
    if derive : return s*(1-s)
    else : return s

# ----------------------------------------------------------------------------------------------------------------------#
# ROUTE

class Route:
    def __init__(self, D_h=None, D_in=10, D_out=10):
        if D_h is None: D_h = D_in
        self.Wr = torch.randn(D_in, D_out)
        self.Wgh = torch.randn(D_h, 1)
        self.Wgr = torch.randn(D_in, 1)
        self.g = 0
        self.pd_g = None # the partial derivative of g
        self.history = defaultdict(lambda: Moment())

    def at(self, time): return self.history[time]

    def getParams(self): return [self.Wr, self.Wgr, self.Wgh]

    def forward(self, h, r, time):
        g = r.matmul(self.Wgr)  # self._x_dot(x)
        if h is not None: g = h.matmul(self.Wgh) + g  # self._h_dot(h) + g
        self.history[time].pd_g = sig(g, derive=True)
        g = sig(g)
        self.at(time).g = g.mean().item()
        self.at(time).z = r.matmul(self.Wr)
        c = g * self.at(time).z
        return c

    def backward(self, e_c, time):
        g_r = e_c.matmul(self.Wr.T)
        e_g = e_c.matmul(self.at(time).z) * self.at(time).pd_g
        g_r += e_g.matmul(self.Wgr.T)
        g_h = e_g.matmul(self.Wgh.T)
        return (g_h, g_r)

    def candidness(self):
        return self.g


route = Route(D_in=3, D_out=2)
r = torch.ones(3)
h = torch.ones(3)

c = route.forward(h,r,0)
assert str(c) == 'tensor([-3.1501, -0.0353])'

g_h, g_r = route.backward(torch.tensor([-1.0, 1.0]),0)
assert str(g_h) == 'tensor([ 1.2534,  0.2197, -0.3332])'
assert str(g_r) == 'tensor([1.7005, 1.5327, 0.1335])'

del route, r, h, c, g_r, g_h

# ----------------------------------------------------------------------------------------------------------------------#
# SOURCE

class Source:
    def __init__(self, D_in, D_out):
        self.Ws = torch.randn(D_out, D_in)

    def getParams(self): return [self.Ws]

    def forward(self, s):
        return s.matmul(self.Ws)

    def backward(self, e_s):
        return e_s.matmul(self.Ws.T)

    def candidness(self): return -1

source = Source(D_in=2, D_out=3)
s = torch.ones(3)
c = source.forward(s)

assert str(c) == 'tensor([-0.1818,  0.2568])'

e = torch.ones(2)
e_c = source.backward(e)

assert str(e_c) == 'tensor([ 1.1008, -0.6303, -0.3955])'

del s, c, e, e_c

# ----------------------------------------------------------------------------------------------------------------------#
# GROUP

class Group:

    def __init__(self, index, dimensionality):
        self.from_conns = dict()
        self.to_conns = dict()
        self.index = index
        self.dimensionality = dimensionality
        self.is_sleeping = False
        self.state = None
        self.targets = [] # indices of targeted routes!
        self.error = None
        self.error_count = 0

    # IO :

    def fall_asleep(self):
        self.is_sleeping = True

    def wake_up(self):
        self.is_sleeping = False

    def state(self):
        return self.state

    def getParams(self):
        params = []
        for node, route in self.to_conns.items():
            params.extend(route.getParams())
        return params

    def str(self, level):
        return level + 'Group: {\n' + \
               level + '   from: ' + str(len(self.from_conns)) + '\n' + \
               level + '   to: ' + str(len(self.to_conns)) + '\n' + \
               level + '   index: ' + str(self.index) + ', ' + 'dimensionality: ' + str(self.dimensionality) + ', ' + \
               'is_sleeping: ' + str(self.is_sleeping) + '\n' + \
               level + '   targets: ' + str(self.targets) + '\n' + \
               level + '   state: ' + str(self.state) + '\n' + \
               level + '};\n'

    # Construction :

    def connect_forward(self, next_groups, cone_size, step):
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

    def start_with(self, x):
        this_is_start = len(self.from_conns) == 0
        assert this_is_start
        self.state = x

    def forward(self, time):

        this_is_start = len(self.from_conns) == 0

        if not self.is_sleeping or this_is_start:
            z = None
            if this_is_start:
                z = self.state  # Start group!
                assert z is not None

            # Source activations :
            for group, source in self.from_conns.items():

                if z is None : z = source.forward(group.state())
                else : z = z + source.forward(group.state())

            # Route activations :
            best_target = None
            best_score = 0
            for group, route in self.to_conns.items():

                if z is None : z = route.forward(self.state(), group.state(), time)
                else: z = z + route.forward(self.state(), group.state(), time)

                # Checking if this route is better than another :
                if route.candidness() > best_score:
                    best_score = route.candidness
                    best_target = group

            assert best_target is not None # There has to be a choice!

            # We activate the best group of neurons :
            best_target.wake_up()
            # No we save the next neuron group which ought to be activated :

            assert z is not None

            if not this_is_start:
                self.state = self.activation(z) # If this is not the start of the network... : Activate!

            return best_target # The best group is being returned!

    def backward(self, e, time):
        pass#if e is None :


    def activation(self, x, derive=False):  # State of the Art activation function, SWISH
        if derive : return torch.sigmoid(x)
        else : return torch.sigmoid(x) * x
        #return x * torch.tanh(nn.functional.softplus(x)) # MISH might also be interesting


# ----------------------------------------------------------------------------------------------------------------------#
# CAPSULE

class Capsule:

    def __init__(self, dimensionality, size):
        self._groups = []
        for i in range(size):
            self._groups.append(Group(i, dimensionality))

    # IO :

    def getParams(self):
        params = []
        for group in self._groups:
            params.extend(group.getParams())
        return params

    def groups(self):
        return self._groups

    def str(self, level):
        asStr = level + 'Capsule: {\n'
        asStr += (level+'   height: ' + str(len(self._groups)) + '\n')
        for group in self._groups:
            asStr += group.str(level + '   ')
        asStr += ('\n' + level + '};\n')
        return asStr

    # Construction :

    def connect_forward(self, next_capsule, max_cone, step):
        for group in self._groups:
            group.connect_forward(next_capsule.groups(), max_cone, step)

    # Execution :

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

        #self._sub_modules = torch.nn.ModuleList([])
        #for
            

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTING :

net = Network(
    depth=5,
    max_height=15,
    max_dim=500,
    max_cone=8,
    D_in=100,
    D_out=10,
)
#print(net.str())

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

for ci in range(len(net._capsules)) :
    expected_capsule = expected_structure[ci]
    given_capsule = net._capsules[ci]
    assert expected_capsule['height'] == len(given_capsule._groups)
    for gi in range(len(given_capsule._groups)):
        given_group = given_capsule._groups[gi]
        assert given_group.targets == expected_capsule['targets'][gi]
        if isinstance(expected_capsule['from'],list):
            assert len(given_group.from_conns) == expected_capsule['from'][gi]
        else :
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
#print(net.str())