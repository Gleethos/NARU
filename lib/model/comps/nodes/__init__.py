
from lib.model.comps import Moment, Recorder, CONTEXT
from lib.model.comps.fun import activation
from lib.model.comps.connections import Route
import torch


# ----------------------------------------------------------------------------------------------------------------------#
# Bundle

# A moment holds information about a concrete point in time step t!
# This is important for back-propagation
def default_moment(dimensionality: int):
    m = Moment()
    m.state = torch.zeros(1, dimensionality)
    m.is_sleeping = True
    m.derivative = None
    m.error = None
    m.error_count = 0
    m.conns = dict()
    m.time = None
    return m


class Bundle(Recorder):

    def __init__(self, index: int, dimensionality: int, position=None, with_bias=False):
        super().__init__(default_lambda=lambda: default_moment(dimensionality))
        self.position = position
        self.from_conns = dict()
        self.to_conns = dict()
        self.index = index
        self.dimensionality = dimensionality
        self.targets = []  # indices of targeted routes! (routes to groups)
        if with_bias:
            self.bias = torch.randn(1, dimensionality) / 10
            self.bias.grad = torch.zeros(1, dimensionality)
        else:
            self.bias = None

    def nid(self):
        return 'p' + str(self.position) + 'i' + str(self.index) + 'f' + str(len(self.from_conns)) + 't' + str(
            len(self.to_conns)) + 'd' + str(self.dimensionality)

    # IO :

    def add_error(self, e, time):
        moment = self.latest(time)
        if moment.error is None:
            self.rec(time).error = e
            self.rec(time).error_count = moment.error_count + 1  # -> incrementing the counter! (for normalization)
            # Rule: A group can only have a non zero state if it is sleeping
            state_is_all_zero = torch.equal(moment.state, moment.state * 0)
            assert state_is_all_zero and moment.is_sleeping or not moment.is_sleeping
        else:
            moment.error = moment.error + e
            moment.error_count = moment.error_count + 1  # -> incrementing the counter! (for normalization)

    def get_params(self):
        params = []
        if self.bias is not None and len(self.to_conns) == 0: params.append(self.bias)
        for node, route in self.to_conns.items(): params.extend(route.get_params())
        for node, source in self.from_conns.items(): params.extend(source.get_params())
        return params

    def set_params(self, params):
        for node, route in self.to_conns.items(): route.set_params(params)
        for node, source in self.from_conns.items(): source.set_params(params)

    def str(self, level: str):
        return level + 'Bundle ' + str(self.index) + ' : {' + \
               ' sources: ' + str(len(self.from_conns)) + ',' + \
               ' routes: ' + str(len(self.to_conns)) + ',' + \
               ' dimensionality: ' + str(self.dimensionality) + ', ' + \
               ' targets: ' + str(self.targets) + ' };\n'

    # Construction :

    def connect_forward(self, next_groups: list, cone_size: int, step: int):
        assert step > 0
        next_groups = [g for g in next_groups if g not in self.from_conns and g not in self.to_conns]
        cone_size = min(cone_size, len(next_groups))
        for i in range(cone_size):
            target_index = (self.index + i * step) % len(next_groups)
            target_group = next_groups[target_index]
            self.targets.append(target_index)
            assert target_group.index == target_index
            self.to_conns[target_group] = Route(D_in=target_group.dimensionality, D_out=self.dimensionality)
            target_group.register_source(self)

    def register_source(self, origin_group):
        self.from_conns[origin_group] = Route(D_in=origin_group.dimensionality, D_out=self.dimensionality)

    def number_of_connections(self):
        count = len(self.from_conns) + len(self.to_conns)
        assert count > 0
        return count

    # Execution :

    def start_with(self, time, x: torch.Tensor):
        this_is_start = len(self.from_conns) == 0
        assert this_is_start
        self.rec(time).state = x
        self.rec(time).is_sleeping = False

    def forward(self, time: int):
        assert time >= 0
        this_is_start = len(self.from_conns) == 0
        this_is_end = len(self.to_conns) == 0
        assert not (this_is_start and this_is_end)
        current_moment = self.at(time)
        number_of_connections = self.number_of_connections()

        if not current_moment.is_sleeping:

            # print('Awake:', self.nid(), '-', time)
            current_moment.message = 'Was active!'
            z = None
            if this_is_start:
                z = current_moment.state  # Start group! (maybe "latest" is better)
                assert z is not None

            # Source activations :
            for group, source in self.from_conns.items():
                h, r = self.latest(time - 1).state, group.latest(time - 1).state
                # print('Forward-Src:',self.nid()+' - t'+str(time),': h='+str(h),'r='+str(r))
                if z is None:
                    z = source.forward(h, r, rec=current_moment.conns)
                else:
                    z = z + source.forward(h, r, rec=current_moment.conns)

            # Route activations :
            best_target: Bundle = None
            best_score: float = -1
            for group, route in self.to_conns.items():
                h, r = self.latest(time - 1).state, group.latest(time - 1).state
                # print('Forward-Rte:',self.nid()+' - t'+str(time),': h='+str(h),'r='+str(r))
                if z is None:
                    z = route.forward(h, r, rec=current_moment.conns)
                else:
                    z = z + route.forward(h, r, rec=current_moment.conns)

                # Checking if this route is better than another :
                g = current_moment.conns[route].g
                if not 0 <= g <= 1: print('Illegal gate:', g)
                assert 0 <= g <= 1
                if g > best_score: best_score, best_target = g, group

            if len(self.to_conns.items()) > 0:
                if best_target is None: print(self.to_conns.items())
                assert best_target is not None  # There has to be a choice!

                # We activate the best group of neurons :
                best_target.rec(time + 1).is_sleeping = False  # wake up!
                best_target.rec(time + 1).message = 'Just woke up!'
                # print('Chose:',best_target.position, best_target.index, ', at ', self.position, self.index)
                # No we save the next neuron group which ought to be activated :

            assert z is not None
            if self.bias is not None and not this_is_end:
                z = z + self.bias  # Bias is optional!
            z = z / number_of_connections

            if this_is_start:
                current_moment.state = z  # If this is not the start of the network... : Activate!
                current_moment.derivative = (z * 0 + 1) / number_of_connections
            elif this_is_end:
                current_moment.state = z  # If this is not the end of the network... : No activation!!
                current_moment.derivative = 1 / number_of_connections  # The derivative is simple because no function...
            else:
                current_moment.state = activation(x=z)  # If this is not the start of the network... : Activate!
                # current_moment.derivative = activation(x=z, derive=True) / number_of_connections

            # print('Fwd-'+str(self.nid())+'-'+str(z)+'-Choice:', best_target)

            return True, best_target  # The best group is being returned!
        return False, None


print('Group loaded! Unit-Testing now...')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTING:

torch.manual_seed(66642999)


def test_simple_net(group, other1, other2, output):
    # connecting them...
    group.connect_forward(next_groups=[other1, other2], cone_size=293943, step=1)
    other1.connect_forward(next_groups=[output], cone_size=123, step=1)
    other2.connect_forward(next_groups=[output], cone_size=123, step=1)

    assert len(group.from_conns) == 0
    assert len(group.to_conns) == 2
    assert len(other1.from_conns) == 1
    assert len(other2.from_conns) == 1
    assert len(other1.to_conns) == 1
    assert len(other2.to_conns) == 1
    assert len(output.from_conns) == 2
    assert len(output.to_conns) == 0
    #    assert len(CONTEXT.recorders) == 4 + 2 + 2 + 2 + 2 # four groups and 6 connections

    # group.rec(-1).state = torch.tensor([[1.0, 2.0, 3.0]])
    group.start_with(time=0, x=torch.tensor([[1.0, 2.0, 3.0]]))
    groups = [group, other1, other2, output]
    for g in groups: g.forward(0)

    assert not other1.latest(1).is_sleeping  # CHOICE: other1
    assert other2.latest(1).is_sleeping
    assert output.latest(1).is_sleeping
    # assert [r.latest(0).g for r in group.to_conns.values()] == [0.10948651283979416, 0.0009388707112520933]

    # Future states don't know anything:
    assert other1.at(2).is_sleeping
    assert other2.at(2).is_sleeping
    assert output.at(2).is_sleeping

    # group.rec(0).state = torch.tensor([[-3.0, -1.0, -4.0]])
    group.start_with(time=1, x=torch.tensor([[-3.0, -1.0, -4.0]]))
    for g in groups: g.forward(1)

    assert not other1.at(1).is_sleeping  # first step is still recorded...
    assert other2.at(1).is_sleeping
    # assert [r.latest(0).g for r in group.to_conns.values()] == [0.10948651283979416, 0.0009388707112520933]

    # print([r.latest(1).g for r in group.to_conns.values()])
    assert not other1.at(2).is_sleeping  # New step as well!
    assert other2.at(2).is_sleeping  # CHOICE: other2
    assert not output.at(2).is_sleeping
    # assert [r.latest(1).g for r in group.to_conns.values()] == [0.881648063659668, 0.9996621608734131]
    # assert [r.latest(1).g for r in other1.to_conns.values()] == [0.5]

    # last activation (activates output)
    group.rec(1).state = torch.tensor([[2.0, 4.0, -1.0]])
    for g in groups: g.forward(2)
    for g in groups: assert g.latest(0).error_count == 0
    print(str(output.at(2).state))
    # assert str(output.at(2).state) == 'tensor([[1.2334]])'#'tensor([[0.9552]])'#'tensor([[5.1944]])'#'tensor([[-0.2231]])'
    output.add_error(torch.tensor([[1]]), 2)
    assert output.latest(1).error_count == 0
    assert output.latest(2).error_count == 1
    assert output.latest(3).error_count == 0


group = Bundle(index=0, dimensionality=3)

assert group.index == 0
assert group.dimensionality == 3
assert group.from_conns != None
assert group.to_conns != None
assert group.targets == []  # indices of targeted routes!
assert group.latest(0).error == None
assert group.latest(0).error_count == 0
assert group.latest != None

other1 = Bundle(index=0, dimensionality=3)
other2 = Bundle(index=1, dimensionality=3)
output = Bundle(index=0, dimensionality=1)

test_simple_net(group, other1, other2, output)
for r in CONTEXT.recorders: r.reset()  # Resetting allows for a repeat of the test!
test_simple_net(group, other1, other2, output)

del group, other1, other2, output

CONTEXT.recorders = []
print('Bundle Unit-Testing successful!')
print('==============================\n')

# -----------------------------------------

