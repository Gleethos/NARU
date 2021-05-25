
from utility.classes import Bundle
from utility.classes import Loss
from utility.classes import CONTEXT
import math
import torch


def dampener_of(current_index: int, num_of_el: int):
    current_index, num_of_el = min(current_index, 13), min(num_of_el, 13)
    progress = current_index / num_of_el
    dampener = 1 + 999 * (1 - progress) ** 4
    return 1#dampener


# -------------------------------------------------------------------------------------------------------------------- #
# CAPSULE


class Capsule:

    def __init__(self, dimensionality, size: int, position=None, with_bias=False):
        self.position = position
        self.bundles = []
        for i in range(size):  # creating "size" number of groups for this capsule
            self.bundles.append(Bundle(i, dimensionality, position, with_bias=with_bias))


    def forward(self, time):
        actives = []
        for i, bundle in enumerate(self.bundles):
            was_active, choice = bundle.forward(time)
            if was_active:
                actives.append(bundle.index)

        if len(actives) == 0: actives.append(-1)
        assert len(actives) == 1 # Ths might fail when the recorders have not been cleared!!!
        return actives[0]


    # IO :

    def get_params(self):
        params = []
        for bundle in self.bundles:
            params.extend(bundle.get_params())
        return params

    def str(self, level):
        asStr = level + 'Capsule: {\n'
        asStr += (level + '   height: ' + str(len(self.bundles)) + '\n')
        for group in self.bundles:
            asStr += group.str(level + '   ')
        asStr += (level + '};\n')
        return asStr

    # Construction :

    def connect_forward(self, next_capsule, max_cone, step):
        for bundle in self.bundles:
            bundle.connect_forward(next_capsule.bundles, max_cone, step)

    # Execution :

    def start_with(self, time: int, x: torch.Tensor):
        # This is only allowed to be called
        # on first capsule of the network !
        assert len(self.bundles) == 1
        for bundle in self.bundles: bundle.start_with(time=time, x=x)


# -------------------------------------------------------------------------------------------------------------------- #
# Network

from utility.classes import activation

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
        self.ini_args = '(\n' \
                        '    depth      = '+str(depth)+',      # The number of capsules\n' \
                        '    max_height = '+str(max_height)+', \n' \
                        '    max_dim    = '+str(max_dim)+',    \n' \
                        '    max_cone   = '+str(max_cone)+',   \n' \
                        '    D_in       = '+str(D_in)+',       \n' \
                        '    D_out      = '+str(D_out)+',      \n' \
                        '    with_bias  = '+str(with_bias)+'   \n' \
                        ')'

        self.W_in = torch.rand(D_in, D_in, dtype=torch.float32, requires_grad=True)
        self.b_in = torch.rand(1, D_in, dtype=torch.float32, requires_grad=True)
        self.loss = Loss()
        self.depth = depth
        assert depth > 3
        dims = [math.floor(float(x)) for x in self.girth_from(depth, D_in, max_dim, D_out)]
        self.heights = [math.floor(float(x)) for x in self.girth_from(depth, 1, max_height, 1)]
        self.capsules = []

        for i in range(depth):
            self.capsules.append(Capsule(dims[i], self.heights[i], position=i, with_bias=with_bias))

        for i in range(depth):
            if i != depth - 1:
                assert len(self.capsules[i].bundles) * max_cone >= len(self.capsules[i + 1].bundles)
                current_cone = min(len(self.capsules[i + 1].bundles), max_cone)
                step = max(1, int(len(self.capsules[i + 1].bundles) / current_cone))
                self.capsules[i].connect_forward(self.capsules[i + 1], current_cone, step)

    def str(self):
        asStr = ''
        asStr += 'Network'+self.ini_args+': {\n'
        level = '   '
        for capsule in self.capsules:
            asStr += capsule.str(level)
        asStr += '\n};\n'
        return asStr

    def girth_from(self, length: int, start: int, max: int, end: int):
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

    # V2:
    def train_with_autograd_on(self, vectors: list):
        losses = []
        choice_matrix = []
        out_bundle = self.capsules[len(self.capsules) - 1].bundles[0]

        for time in range(len(vectors)-1+(self.depth-1)):

            if time < len(vectors): self.start_with(time, vectors[time])

            choice_indices = []
            for capsule in self.capsules:
                index = capsule.forward(time)
                choice_indices.append(index)
            choice_matrix.append(choice_indices)

            # There is a time delay as large as the network is long:
            if time >= self.depth - 1:
                assert not out_bundle.at(time).is_sleeping
                vector_index = time - self.depth + 2
                assert vector_index > 0
                expected = vectors[vector_index]

                predicted = out_bundle.latest(time).state
                loss = torch.sum( (predicted - expected)**2 ) / torch.numel(predicted)
                loss = loss / dampener_of(current_index=(time - (self.depth - 1)), num_of_el=(len(vectors)-1))
                losses.append(loss)
            else:
                assert out_bundle.at(time).is_sleeping

        assert len(losses) == len(vectors) - 1

        total_loss = sum(losses)/len(losses)
        total_loss.backward()

        for r in CONTEXT.recorders: r.reset()  # Resetting allows for a repeat of the training!
        return choice_matrix, [l.item() for l in losses]

    def start_with(self, time: int, x: torch.Tensor):
        x = x.matmul(self.W_in) + self.b_in
        x = activation(x, derive=False)
        self.capsules[0].start_with(time, x)

    def pred(self, vectors: list):
        preds = []
        out_bundle = self.capsules[len(self.capsules) - 1].bundles[0]
        for time in range(len(vectors) - 1 + (self.depth-1)):
            if time < len(vectors):
                self.start_with(time, x=vectors[time])

            for capsule in self.capsules:
                capsule.forward(time)

            # There is a time delay as large as the network is long:
            if time >= self.depth-1:
                preds.append(out_bundle.at(time).state)
                assert out_bundle.at(time).is_sleeping is False
            else:
                assert out_bundle.at(time).is_sleeping is True

        for r in CONTEXT.recorders: r.reset()  # Resetting allows for a repeat of the prediction!
        return preds

    def get_params(self):
        params = [self.W_in, self.b_in]
        for c in self.capsules:
            for g in c.bundles:
                params.extend(g.get_params())
        return params

    def set_params(self, params: list):
        params = params.copy()
        self.W_in = params.pop(0)
        self.b_in = params.pop(0)
        for c in self.capsules:
            for g in c.bundles: g.set_params(params)
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
#print(net.str()) # TODO : asserts!


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

assert len(net.capsules) == len(expected_structure)

for ci in range(len(net.capsules)):
    expected_capsule = expected_structure[ci]
    given_capsule = net.capsules[ci]
    assert expected_capsule['height'] == len(given_capsule.bundles)
    for gi in range(len(given_capsule.bundles)):
        given_group = given_capsule.bundles[gi]
        assert given_group.targets == expected_capsule['targets'][gi]
        if isinstance(expected_capsule['from'], list):
            assert len(given_group.from_conns) == expected_capsule['from'][gi]
        else:
            assert len(given_group.from_conns) == expected_capsule['from']
        assert len(given_group.to_conns) == expected_capsule['to']

#net = Network(
#    depth=7,
#    max_height=18,
#    max_dim=128,
#    max_cone=6,
#    D_in=50,
#    D_out=50,
#)
#print(net.str())