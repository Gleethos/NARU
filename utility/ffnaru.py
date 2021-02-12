
from utility.classes import Group
from utility.classes import Loss
from utility.classes import CONTEXT
import math
import torch

# -------------------------------------------------------------------------------------------------------------------- #
# CAPSULE


class Capsule:

    def __init__(self, dimensionality, size: int, position=None):
        self.groups = []
        for i in range(size):  # creating "size" number of groups for this capsule
            self.groups.append(Group(i, dimensionality, position))

    def forward(self, time):
        for group in self.groups:
            group.forward(time)

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

    def start_with(self, time, x):
        # This is only allowed to be called
        # on first capsule of the network !
        assert len(self.groups) == 1
        for group in self.groups: group.rec(time).state = x


# -------------------------------------------------------------------------------------------------------------------- #
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
        self.loss = Loss()
        self.depth = depth
        assert depth > 3
        dims = [math.floor(float(x)) for x in self.girth_from(depth, D_in, max_dim, D_out)]
        heights = [math.floor(float(x)) for x in self.girth_from(depth, 1, max_height, 1)]
        self._capsules = []

        for i in range(depth):
            self._capsules.append(Capsule(dims[i], heights[i], position=i))

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

    def train_on(self, vectors):
        losses = []
        in_group = self._capsules[0]
        out_group = self._capsules[len(self._capsules)-1].groups[0]
        for time in range(len(vectors)+self.depth):
            print('\nStepping forward, current time:', time, '; Tokens:', len(vectors), '; Network depth:',self.depth,';')
            if time < len(vectors):
                in_group.start_with(time, vectors[time])

            for capsule in self._capsules:
                capsule.forward(time)

            if time >= self.depth:
                print('Back-propagating now!')
                expected = vectors[time-self.depth]
                e = self.loss(out_group.latest(time).state, expected)
                out_group.add_error(e,time)
                out_group.backward(time)
                losses.append(self.loss.loss)

        assert len(losses) == len(vectors)

        for r in CONTEXT.recorders: r.reset()  # Resetting allows for a repeat of the test!
        return losses

    def pred(self, vectors):
        preds = []
        in_group = self._capsules[0]
        out_group = self._capsules[len(self._capsules) - 1].groups[0]
        for time in range(len(vectors) + self.depth):
            if time < len(vectors):
                in_group.start_with(time, vectors[time])

            for capsule in self._capsules:
                capsule.forward(time)

            if time >= self.depth:
                preds.append(out_group.latest(time).state)

        for r in CONTEXT.recorders: r.reset()  # Resetting allows for a repeat of the test!
        return preds

    # TESTING :

torch.manual_seed(66642999)

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