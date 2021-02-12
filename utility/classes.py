
from collections import defaultdict
import threading
import torch
import copy

CONTEXT = threading.local()
CONTEXT.recorders = []

torch.manual_seed(666)

# Moments are responsible for holding variables needed for propagating back through time.
class Moment:
    def __init__(self):
        self.is_record = False # Moments can be records or "on the fly creations" provided by the defaultdict...
    pass

# A useful method used later on...
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
        self.default_lambda = default_lambda
        self.history: dict = None  # history
        self.reset()
        CONTEXT.recorders.append(self)

    def reset(self):
        self.history: dict = defaultdict(self.default_lambda)  # history

    def latest(self, time: int):
        moment = self.history[time]
        is_record = moment.is_record
        while not is_record:
            time -= 1
            moment = self.history[time]
            is_record = moment.is_record
            if time < 0 : # first time step is recorded by default!
                self.history[0] = self.history[0]
                self.history[0].is_record = True
                return self.history[0]

        return self.history[time]

    def at(self, time:int):
        return self.history[time]

    def rec(self, time: int):
        current = self.history[time]
        if not current.is_record:
            self.history[time] = copy.copy(self.latest(time))  # new entry (new Moment object)
        self.history[time].is_record = True
        return self.history[time]


# -------------------------------------------------------------------------------------------------------------------- #
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
        g = r.matmul(self.Wgr) + h.matmul(self.Wgh)
        self.rec(time).pd_g = sig(g, derive=True)
        g = sig(g)
        self.rec(time).g = g.mean().item()  # g is used for routing!
        self.rec(time).z = r.matmul(self.Wr)
        c = g * self.latest(time).z
        return c

    def backward(self, e_c, time, r, h):
        self.Wr.grad += r.T.matmul(e_c * self.latest(time).g)
        self.Wr.grad += r.T.matmul(e_c * self.latest(time).g)
        g_r = e_c.matmul(self.Wr.T)
        e_g = e_c.matmul(self.latest(time).z.T) * self.latest(time).pd_g
        self.Wgr.grad += r.T.matmul(e_g)
        self.Wgh.grad += h.T.matmul(e_g)
        g_r += e_g.matmul(self.Wgr.T)
        g_h = e_g.matmul(self.Wgh.T)
        return (g_h, g_r)


print('Route loaded! Unit-Testing now...')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTING:

torch.manual_seed(66642999)

route = Route(D_in=3, D_out=2)

assert len(CONTEXT.recorders) == 1
assert CONTEXT.recorders[0] == route

r = torch.ones(1, 3)
h = torch.ones(1, 2)

c = route.forward(h, r, 0)
assert str(c) == 'tensor([[ 0.1663, -0.0411]])'

g_h, g_r = route.backward(
    torch.tensor([[-1.0, 1.0]]), 0,
    h=torch.tensor([[2.0, -3.0]]),
    r=torch.tensor([[-1.0, 4.0, 2.0]])
)
assert str(g_h) == 'tensor([[-0.0995,  0.1821]])'
assert str(g_r) == 'tensor([[ 1.0981, -2.0502,  0.3621]])'

assert str(route.Wgh.grad) == "tensor([[-0.2689],\n        [ 0.4033]])"
assert str(route.Wgr.grad) == "tensor([[ 0.1344],\n        [-0.5378],\n        [-0.2689]])"
#assert str(route.Wr.grad) == "tensor([[ 0.3515, -0.3515],\n        [-1.4062,  1.4062],\n        [-0.7031,  0.7031]])"

del route, r, h, c, g_r, g_h
CONTEXT.recorders = []
print('Route Unit-Testing successful!')

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
        self.Ws.grad += s.T.matmul(e_s)
        return e_s.matmul(self.Ws.T)


print('Source loaded! Unit-Testing now...')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTING:

torch.manual_seed(66642999)

source = Source(D_in=2, D_out=3)

assert len(CONTEXT.recorders) == 1
assert CONTEXT.recorders[0] == source

s = torch.ones(1, 2)
c = source.forward(s)
assert str(c) == 'tensor([[-2.4602,  1.0154,  1.8009]])' # shape=(1,3)

e = torch.ones(1, 3)
e_c = source.backward(e, s=torch.tensor([[2.0, -1.0]])) # shape=(1,2)
assert str(e_c) == 'tensor([[-0.3409,  0.6971]])' # shape=(1,2)
assert str(source.Ws.grad) == "tensor([[ 2.,  2.,  2.],\n        [-1., -1., -1.]])"

del s, c, e, e_c
CONTEXT.recorders = []
print('Source Unit-Testing successful!')

# ----------------------------------------------------------------------------------------------------------------------#
# GROUP

# A moment holds information about a concrete point in time step t!
# This is important for back-propagation
def default_moment(dimensionality: int):
    m = Moment()
    m.state = torch.zeros(1, dimensionality)
    m.is_sleeping = True
    m.derivative = None
    m.error = None
    m.error_count = 0
    return m

class Group(Recorder):

    def __init__(self, index: int, dimensionality: int, position=None):
        super().__init__(default_lambda=lambda: default_moment(dimensionality))
        self.position = position
        self.from_conns = dict()
        self.to_conns = dict()
        self.index = index
        self.dimensionality = dimensionality
        self.targets = []  # indices of targeted routes! (routes to groups)

    def nid(self):
        return 'i'+str(self.index)+'f'+str(len(self.from_conns))+'t'+str(len(self.to_conns))+'d'+str(self.dimensionality)

    # IO :

    def add_error(self, e, time):
        moment = self.latest(time)
        if moment.error is None:
            self.rec(time).error = e
            self.rec(time).error_count = moment.error_count + 1  # -> incrementing the counter! (for normalization)
        else:
            moment.error = moment.error + e
            moment.error_count = moment.error_count + 1 #-> incrementing the counter! (for normalization)

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
        self.from_conns[origin_group] = Source(D_in=origin_group.dimensionality, D_out=self.dimensionality)

    # Execution :

    def start_with(self, time, x: torch.Tensor):
        this_is_start = len(self.from_conns) == 0
        assert this_is_start
        self.rec(time-1).state = x

    def forward(self, time: int):
        assert time >= 0
        this_is_start = len(self.from_conns) == 0

        if not self.at(time).is_sleeping or this_is_start:
            z = None
            if this_is_start:
                z = self.latest(time).state  # Start group!
                print('Starting with: ', str(z.shape))
                assert z is not None

            # Source activations :
            for group, source in self.from_conns.items():
                s = group.latest(time-1).state
                print(self.nid() + ' - t' + str(time), ': s=' + str(s.shape))
                if z is None:
                    z = source.forward(s)
                else:
                    z = z + source.forward(s)

            # Route activations :
            best_target = None
            best_score = -1
            for group, route in self.to_conns.items():
                h, r = self.latest(time-1).state, group.latest(time-1).state
                print(self.nid()+' - t'+str(time),': h='+str(h.shape),'r='+str(r.shape))
                if z is None:
                    z = route.forward(h, r, time)
                else:
                    z = z + route.forward(h, r, time)

                # Checking if this route is better than another :
                #print('Route Gate:', route.latest(time).g, '>?>', best_score)
                if route.latest(time).g > best_score:
                    best_score = route.latest(time).g
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

            return best_target  # The best group is being returned!

    def backward(self, time: int):

        current_error: torch.Tensor = self.latest(time).error

        if not self.at(time).is_sleeping:  # Back-prop only when this group was active at that time!

            # Multiplying with the partial derivative of the activation of this group.
            current_error = current_error * self.latest(time).derivative
            current_state = self.latest(time).state

            assert self.latest(time).error_count > 0 # This node should already have an error! (because of route connection...)
            # Normalization technique so that gradients do not explode:
            current_error = current_error / self.latest(time).error_count

            # Source (error) bac-prop :
            for group, source in self.from_conns.items(): # Distributing error to source groups...

                g_s = source.backward(
                    e_s=current_error,
                    s=group.latest(time - 1).state  # Needed to calculate gradients of the weights!
                )
                group.add_error(g_s, time - 1) # setting or accumulating the error!

            # Route (error) back-prop :
            for group, route in self.to_conns.items():  # Distributing error to route groups...

                g_h, g_r = route.backward(
                    e_c=current_error,
                    time=time,
                    r=group.latest(time - 1).state,  # Needed to calculate gradients of the weights!
                    h=current_state  # Needed to calculate gradients of the weights!
                )
                group.add_error(g_r, time - 1)

                # Accumulating g_h to self:
                self.add_error(g_h, time - 1)

            # Source Group backprop :
            for group, source in self.from_conns.items():
                group.backward(time - 1)

            # Route Group backprop :
            for group, route in self.to_conns.items():
                group.backward(time - 1)


    def activation(self, x, derive=False):  # State of the Art activation function, SWISH
        if derive:
            return sig(x)
        else:
            return sig(x) * x
        # return x * torch.tanh(nn.functional.softplus(x)) # MISH might also be interesting


print('Group loaded! Unit-Testing now...')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTING:

torch.manual_seed(66642999)


def test_simple_net(group, other1, other2, output):
    # connecting them...
    group.connect_forward(next_groups=[other1,other2], cone_size=293943, step=1)
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
    assert len(CONTEXT.recorders) == 4 + 2 + 2 + 2 + 2 # four groups and 6 connections

    group.rec(-1).state = torch.tensor([[1.0, 2.0, 3.0]])
    groups = [group, other1, other2, output]
    for g in groups: g.forward(0)

    assert not other1.latest(1).is_sleeping # CHOICE: other1
    assert other2.latest(1).is_sleeping
    assert output.latest(1).is_sleeping
    assert [r.latest(0).g for r in group.to_conns.values()] == [0.10948651283979416, 0.0009388707112520933]

    # Future states don't know anything:
    assert other1.at(2).is_sleeping
    assert other2.at(2).is_sleeping
    assert output.at(2).is_sleeping

    group.rec(0).state = torch.tensor([[-3.0, -1.0, -4.0]])
    for g in groups: g.forward(1)

    assert not other1.at(1).is_sleeping # first step is still recorded...
    assert other2.at(1).is_sleeping
    assert [r.latest(0).g for r in group.to_conns.values()] == [0.10948651283979416, 0.0009388707112520933]

    print([r.latest(1).g for r in group.to_conns.values()])
    assert other1.at(2).is_sleeping # New step as well!
    assert not other2.at(2).is_sleeping # CHOICE: other2
    assert not output.at(2).is_sleeping
    assert [r.latest(1).g for r in group.to_conns.values()] == [0.881648063659668, 0.9996621608734131]
    assert [r.latest(1).g for r in other1.to_conns.values()] == [0.5]

    #last activation (activates output)
    group.rec(1).state = torch.tensor([[2.0, 4.0, -1.0]])
    for g in groups: g.forward(2)
    for g in groups: assert g.latest(0).error_count == 0
    print(str(output.at(2).state))
    assert str(output.at(2).state) == 'tensor([[5.1944]])'#'tensor([[-0.2231]])'
    output.add_error(torch.tensor([[1]]),2)
    assert output.latest(1).error_count == 0
    assert output.latest(2).error_count == 1
    assert output.latest(3).error_count == 0

    output.backward(2) # backprop happens recursively!

    grad = str(next(iter(other1.from_conns.values())).Ws.grad)
    print(grad)
    #assert grad in [
    #    'tensor([[ 0.0116, -0.1126,  1.6734],\n        [ 0.0231, -0.2252,  3.3468],\n        [-0.0058,  0.0563, -0.8367]])',
    #    'tensor([[ 0.0231, -0.2252,  3.3468],\n        [ 0.0463, -0.4505,  6.6936],\n        [-0.0116,  0.1126, -1.6734]])'
    #]
    grad = str(next(iter(other2.from_conns.values())).Ws.grad)
    assert grad == 'tensor([[0., 0., 0.],\n        [0., 0., 0.],\n        [0., 0., 0.]])'

    assert str(next(iter(group.to_conns.values())).Wgh.grad) == 'tensor([[0.],\n        [0.],\n        [0.]])'
    assert str(next(iter(group.to_conns.values())).Wgr.grad) == 'tensor([[0.],\n        [0.],\n        [0.]])'
    assert str(next(iter(group.to_conns.values())).Wr.grad) == 'tensor([[0., 0., 0.],\n        [0., 0., 0.],\n        [0., 0., 0.]])'

    assert str(next(iter(other1.to_conns.values())).Wgh.grad) == 'tensor([[0.],\n        [0.],\n        [0.]])'
    assert str(next(iter(other1.to_conns.values())).Wgr.grad) == 'tensor([[0.]])'
    assert str(next(iter(other1.to_conns.values())).Wr.grad) == 'tensor([[0., 0., 0.]])'

    assert str(next(iter(other2.to_conns.values())).Wgh.grad) == 'tensor([[0.],\n        [0.],\n        [0.]])'
    assert str(next(iter(other2.to_conns.values())).Wgr.grad) == 'tensor([[0.]])'
    assert str(next(iter(other2.to_conns.values())).Wr.grad) == 'tensor([[0., 0., 0.]])'



group = Group(index=0, dimensionality=3)

assert group.index == 0
assert group.dimensionality == 3
assert group.from_conns != None
assert group.to_conns != None
assert group.targets == []  # indices of targeted routes!
assert group.latest(0).error == None
assert group.latest(0).error_count == 0
assert group.latest != None

other1 = Group(index=0,dimensionality=3)
other2 = Group(index=1,dimensionality=3)
output = Group(index=0,dimensionality=1)

test_simple_net(group, other1, other2, output)
for r in CONTEXT.recorders: r.reset() # Resetting allows for a repeat of the test!
test_simple_net(group, other1, other2, output)

del group, other1, other2, output

CONTEXT.recorders = []
print('Group Unit-Testing successful!')
print('==============================\n')

# -----------------------------------------


class Loss:

    def __init__(self):
        self.mse = torch.nn.MSELoss()
        self.loss = 0

    def __call__(self, tensor : torch.Tensor, target : torch.Tensor):
        clone = tensor.clone()
        clone.requires_grad = True
        self.loss = self.mse(clone, target)
        self.loss.backward()
        self.loss = self.loss.item()
        return clone.grad





