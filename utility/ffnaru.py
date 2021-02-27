
from utility.classes import Group
from utility.classes import Loss
from utility.classes import CONTEXT
import math
import torch

# -------------------------------------------------------------------------------------------------------------------- #
# CAPSULE


class Capsule:

    def __init__(self, dimensionality, size: int, position=None, with_bias=False):
        self.position = position
        self.groups = []
        for i in range(size):  # creating "size" number of groups for this capsule
            self.groups.append(Group(i, dimensionality, position, with_bias=with_bias))

    def forward(self, time):
        actives = []
        for i, group in enumerate(self.groups):
            was_active, choice = group.forward(time)
            if was_active:
                actives.append(group.index)

        if len(actives) == 0: actives.append(-1)
        assert len(actives) == 1
        return actives[0]

    def backward(self, time):
        actives = []
        for i, group in enumerate(self.groups):
            was_active, choice = group.backward(time)
            if was_active:
                actives.append(group.index)

        if len(actives) == 0: actives.append(-1)
        assert len(actives) == 1 # Only one group can be active in one capsule
        return actives[0]

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
                 with_bias=False
                 ):
        self.loss = Loss()
        self.depth = depth
        assert depth > 3
        dims = [math.floor(float(x)) for x in self.girth_from(depth, D_in, max_dim, D_out)]
        heights = [math.floor(float(x)) for x in self.girth_from(depth, 1, max_height, 1)]
        self._capsules = []

        for i in range(depth):
            self._capsules.append(Capsule(dims[i], heights[i], position=i, with_bias=with_bias))

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
            if math.fabs(mid - i) < 1: g.append(max)
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
        choice_matrix = []
        in_group = self._capsules[0]
        out_group = self._capsules[len(self._capsules)-1].groups[0]
        print('Forward pass:')
        for time in range(len(vectors)+(self.depth-1)):
            #print('\nStepping forward, current time:', time, '; Tokens:', len(vectors), '; Network depth:',self.depth,';')
            if time < len(vectors):
                in_group.start_with(time, vectors[time])

            # The following will be used to perform assertions for validity:
            for recorder in CONTEXT.recorders: recorder.time_restrictions = [time-1, time, time+1]

            choice_indices = []
            for capsule in self._capsules:
                index = capsule.forward(time)
                choice_indices.append(index)
            choice_matrix.append(choice_indices)
            print('Choice indices:', choice_indices)

            # There is a time delay as large as the network is long:
            if time >= self.depth - 1:
                assert not out_group.latest(time).is_sleeping
                progress = (time - (self.depth - 1)) / (len(vectors)-1)
                dampener = 10 + 90 * ( 1 - progress )**4
                #print('Back-propagating now! Progress:', progress, '%; Dampener:', dampener, ';')
                expected = vectors[time-(self.depth-1)]
                predicted = out_group.latest(time).state
                e = self.loss(predicted, expected)
                e = e / dampener
                #out_group.add_error(e, time)
                out_group.at(time).error = e
                out_group.at(time).error_count = 1
                losses.append(self.loss.loss)
                print('Loss at ', time, ':', self.loss.loss)
            for recorder in CONTEXT.recorders: recorder.time_restrictions = None
        #else:
        #    assert out_group.latest(time).is_sleeping

        assert len(losses) == len(vectors)

        for time in range(len(vectors)+(self.depth-1)-1, -1, -1):
            if time >= self.depth - 1:
                print('backprop at:',time)
                for recorder in CONTEXT.recorders: recorder.time_restrictions = [time - 1, time, time+1]
                choice_indices = []
                for i, capsule in enumerate(self._capsules):
                    index = capsule.backward(time)
                    choice_indices.append(index)
                print('Backward Choice indices:', choice_indices)
                for recorder in CONTEXT.recorders: recorder.time_restrictions = None

        for r in CONTEXT.recorders: r.reset()  # Resetting allows for a repeat of the training!
        return choice_matrix, losses

    def pred(self, vectors):
        preds = []
        in_group = self._capsules[0]
        out_group = self._capsules[len(self._capsules) - 1].groups[0]
        for time in range(len(vectors) + self.depth):
            if time < len(vectors):
                in_group.start_with(time, vectors[time])

            for capsule in self._capsules: capsule.forward(time)

            if time >= self.depth: preds.append(out_group.latest(time).state)

        for r in CONTEXT.recorders: r.reset()  # Resetting allows for a repeat of the prediction!
        return preds

    def get_params(self):
        params = []
        for c in self._capsules:
            for g in c.groups:
                params.extend(g.get_params())
        return params

    def set_params(self, params):
        params = params.copy()
        for c in self._capsules:
            for g in c.groups: g.set_params(params)
        assert len(params) == 0

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