
from collections import defaultdict
import threading
import torch
import copy

CONTEXT = threading.local()
CONTEXT.recorders = []
CONTEXT.BPTT_limit = 10

torch.manual_seed(666)

# ----------------------------------------------------------------------------------------------------------------------#


# A useful method used later on...
def sig(x, derive=False):
    s = torch.sigmoid(x)
    if derive: return s * (1 - s)
    else: return s


def activation(x, derive=False):  # State of the Art activation function, SWISH
    s = sig(x) * x # return sig(x, derive)
    if derive: return s + sig(x) * (1-s)
    else: return s


def mish(x, derive=False):
    sfp = torch.nn.Softplus()
    if derive:
        omega = torch.exp(3 * x) + 4 * torch.exp(2 * x) + (6 + 4 * x) * torch.exp(x) + 4 * (1 + x)
        delta = 1 + ( (torch.exp(x) + 1)**2 )
        return torch.exp(x) * omega / (delta**2)
    else:
        return x * torch.tanh(sfp(x))


# ----------------------------------------------------------------------------------------------------------------------#
# HISTORY BASE CLASS


# Moments are responsible for holding variables needed for propagating back through time.
class Moment:
    def __init__(self):
        self.is_record = False # Moments can be records or "on the fly creations" provided by the defaultdict...
    pass


class Recorder:
    def __init__(self, default_lambda):
        self.default_lambda = default_lambda
        self.history: dict = None  # history
        self.reset()
        CONTEXT.recorders.append(self)
        self.time_restrictions : list = None

    def reset(self): self.history: dict = defaultdict(self.default_lambda)  # history

    def latest(self, time: int):
        if self.time_restrictions is not None: assert time in self.time_restrictions
        moment = self.history[time]
        is_record = moment.is_record
        while not is_record:
            time -= 1
            moment = self.history[time]
            is_record = moment.is_record
            if time < 0 : # first time step is recorded by default!
                self.history[-1] = self.history[-1]
                self.history[-1].is_record = True
                return self.history[-1]

        return self.history[time]

    def at(self, time:int):
        if self.time_restrictions is not None: assert time in self.time_restrictions
        return self.history[time]

    def rec(self, time: int):
        if self.time_restrictions is not None: assert time in self.time_restrictions
        if not self.history[time].is_record:
            self.history[time] = self.history[time]#self.latest(time)#copy.copy(self.latest(time))  # new entry (new Moment object)
            self.history[time].time = time
        self.history[time].is_record = True
        return self.history[time]


# -------------------------------------------------------------------------------------------------------------------- #
# ROUTE

class Route:

    def __init__(self, D_in=10, D_out=10):
        D_h = D_out
        self.Wr = torch.randn(D_in, D_out)
        self.Wr.grad = torch.zeros(D_in, D_out)

        self.Wgh = torch.randn(D_h, 1)
        self.Wgh.grad = torch.zeros(D_h, 1)

        self.Wgr = torch.randn(D_in, 1)
        self.Wgr.grad = torch.zeros(D_in, 1)
        self.pd_g: torch.Tensor = None  # the partial derivative of g

    def get_params(self): return [self.Wr, self.Wgr, self.Wgh]

    def set_params(self, params):
        grad1, grad2, grad3 = self.Wr.grad, self.Wgr.grad, self.Wgh.grad
        self.Wr  *= 0
        self.Wgr *= 0
        self.Wgh *= 0
        self.Wr  += params.pop(0)
        self.Wgr += params.pop(0)
        self.Wgh += params.pop(0)
        self.Wr.grad = grad1
        self.Wgr.grad = grad2
        self.Wgh.grad = grad3

    def forward(self, h: torch.Tensor, r: torch.Tensor, rec: dict):
        m = Moment()
        g = (r.matmul(self.Wgr) + h.matmul(self.Wgh)) / 2
        m.pd_g = sig(g, derive=True) / 2 # inner times outer derivative
        g = sig(g)
        assert 0 <= g <= 1  # Sigmoid should be within bounds!
        m.g = g.mean().item()  # g is used for routing! Largest gate wins!
        m.z = r.matmul(self.Wr) # z is saved for back-prop.
        c = g * m.z
        rec[self] = m
        return c

    def backward(self, e_c: torch.Tensor, rec: dict, r: torch.Tensor, h: torch.Tensor):
        m = rec[self]
        self.Wr.grad += r.T.matmul(e_c * m.g)
        self.Wr.grad += r.T.matmul(e_c * m.g)
        g_r = e_c.matmul(self.Wr.T)
        e_g = e_c.matmul(m.z.T) * m.pd_g
        self.Wgr.grad += r.T.matmul(e_g)
        self.Wgh.grad += h.T.matmul(e_g)
        g_r += e_g.matmul(self.Wgr.T)
        g_h = e_g.matmul(self.Wgh.T)
        return g_h, g_r


print('Route loaded! Unit-Testing now...')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTING:

torch.manual_seed(66642999)

route = Route(D_in=3, D_out=2)

assert len(CONTEXT.recorders) == 0
#assert CONTEXT.recorders[0] == route

r, h = torch.ones(1, 3), torch.ones(1, 2)

#c = route.forward(h, r, 0)
#assert str(c) == 'tensor([[ 0.1663, -0.0411]])'

#g_h, g_r = route.backward(
#    torch.tensor([[-1.0, 1.0]]), 0,
#    h=torch.tensor([[2.0, -3.0]]),
#    r=torch.tensor([[-1.0, 4.0, 2.0]])
#)
#assert str(g_h) == 'tensor([[-0.0995,  0.1821]])'
#assert str(g_r) == 'tensor([[ 1.0981, -2.0502,  0.3621]])'

#assert str(route.Wgh.grad) == "tensor([[-0.2689],\n        [ 0.4033]])"
#assert str(route.Wgr.grad) == "tensor([[ 0.1344],\n        [-0.5378],\n        [-0.2689]])"
#assert str(route.Wr.grad) == "tensor([[ 0.3515, -0.3515],\n        [-1.4062,  1.4062],\n        [-0.7031,  0.7031]])"

del route, r, h#, c, g_r, g_h
CONTEXT.recorders = []
print('Route Unit-Testing successful!')


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
    m.conns = dict()
    m.time = None
    return m


class Group(Recorder):

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
        return 'p'+str(self.position)+'i'+str(self.index)+'f'+str(len(self.from_conns))+'t'+str(len(self.to_conns))+'d'+str(self.dimensionality)

    # IO :

    def add_error(self, e, time):
        moment = self.latest(time)
        if moment.error is None:
            self.rec(time).error = e
            self.rec(time).error_count = moment.error_count + 1  # -> incrementing the counter! (for normalization)
            assert not torch.equal(moment.state, moment.state * 0) or moment.is_sleeping
        else:
            moment.error = moment.error + e
            moment.error_count = moment.error_count + 1 #-> incrementing the counter! (for normalization)

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
        return level + 'Group: {\n' + \
               level + '   from: ' + str(len(self.from_conns)) + ',' + \
               level + '   to: ' + str(len(self.to_conns)) + ',' + \
               level + '   index: ' + str(self.index) + ', ' + 'dimensionality: ' + str(self.dimensionality) + ', ' + \
               '   targets: ' + str(self.targets) + '\n' + \
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
        self.from_conns[origin_group] = Route(D_in=origin_group.dimensionality, D_out=self.dimensionality)

    def number_of_connections(self):
        count = len(self.from_conns) + len(self.to_conns)
        assert count > 0
        return count

    # Execution :

    def start_with(self, time, x: torch.Tensor):
        this_is_start = len(self.from_conns) == 0
        assert this_is_start
        self.rec(time-1).state = x
        self.rec(time-1).is_sleeping = False

    def forward(self, time: int):
        assert time >= 0
        this_is_start = len(self.from_conns) == 0
        this_is_end = len(self.to_conns) == 0
        current_moment = self.at(time)
        number_of_connections = self.number_of_connections()

        if not current_moment.is_sleeping or this_is_start:

            print('Awake:', self.nid(), '-', time)
            current_moment.message = 'Was active!'
            z = None
            if this_is_start:
                z = current_moment.state  # Start group! (maybe "latest" is better)
                assert z is not None

            # Source activations :
            for group, source in self.from_conns.items():
                h, r = self.latest(time - 1).state, group.latest(time - 1).state
                #print('Forward-Src:',self.nid()+' - t'+str(time),': h='+str(h),'r='+str(r))
                if z is None: z = source.forward(h, r, rec=current_moment.conns)
                else: z = z + source.forward(h, r, rec=current_moment.conns)

            # Route activations :
            best_target: Group = None
            best_score: float = -1
            for group, route in self.to_conns.items():
                h, r = self.latest(time-1).state, group.latest(time-1).state
                #print('Forward-Rte:',self.nid()+' - t'+str(time),': h='+str(h),'r='+str(r))
                if z is None: z = route.forward(h, r, rec=current_moment.conns)
                else: z = z + route.forward(h, r, rec=current_moment.conns)

                # Checking if this route is better than another :
                g = current_moment.conns[route].g
                if not 0 <= g <= 1: print('Illegal gate:', g)
                assert 0 <= g <= 1
                if g > best_score: best_score, best_target = g, group

            if len(self.to_conns.items()) > 0:
                if best_target is None: print(self.to_conns.items())
                assert best_target is not None  # There has to be a choice!

                # We activate the best group of neurons :
                best_target.rec(time+1).is_sleeping = False  # wake up!
                best_target.rec(time+1).message = 'Just woke up!'
                #print('Chose:',best_target.position, best_target.index, ', at ', self.position, self.index)
                # No we save the next neuron group which ought to be activated :

            assert z is not None
            if self.bias is not None and not this_is_end:
                z = z + self.bias  # Bias is optional!
            z = z / number_of_connections

            if this_is_end:
                current_moment.state = z  # If this is not the end of the network... : No activation!!
                current_moment.derivative = 1 / number_of_connections # The derivative is simple because no function...
            elif not this_is_start:
                current_moment.state = activation(x=z)  # If this is not the start of the network... : Activate!
                current_moment.derivative = activation(x=z, derive=True) / number_of_connections

            #print('Fwd-'+str(self.nid())+'-'+str(z)+'-Choice:', best_target)

            return True, best_target  # The best group is being returned!
        return False, None

    def backward(self, time: int):
        assert time >= 0
        this_is_start = len(self.from_conns) == 0
        this_is_end = len(self.to_conns) == 0

        current_moment = self.at(time)

        if not current_moment.is_sleeping and current_moment.error is not None:  # Back-prop only when this group was active at that time!

            current_error: torch.Tensor = current_moment.error
            print('Was awake, now backprop:', self.nid(), '-', time)
            # Multiplying with the partial derivative of the activation of this group.
            if not this_is_start:
                current_error = current_error * current_moment.derivative
                if len(self.to_conns) > 0:
                    assert len([r for r, g in self.to_conns.items() if not r.at(time+1).is_sleeping]) == 1

            assert current_moment.error_count > 0 # This node should already have an error! (because of route connection...)

            # Normalization technique so that gradients do not explode:
            #current_error = current_error / self.latest(time).error_count

            # Resetting the error:
            current_moment.error = None
            current_moment.error_count = 0

            if self.bias is not None and not this_is_end: # Bias is optional
                self.bias.grad += current_error

            # Source (error) bac-prop :
            for group, source in self.from_conns.items(): # Distributing error to source groups...

                g_h, g_r = source.backward(
                    e_c=current_error,
                    rec=current_moment.conns,
                    r=group.latest(time - 1).state,  # Needed to calculate gradients of the weights!
                    h=self.latest(time - 1).state  # Needed to calculate gradients of the weights!
                )
                group.add_error(g_r, time - 1)
                # Accumulating g_h to self:
                self.add_error(g_h, time - 1) # setting or accumulating the error!
                #print('Back-Src: ',self.nid() + ' - t' + str(time), ': g_h=' + str(g_h), 'g_r=' + str(g_r))

            # Route (error) back-prop :
            for group, route in self.to_conns.items():  # Distributing error to route groups...

                g_h, g_r = route.backward(
                    e_c=current_error,
                    rec=current_moment.conns,
                    r=group.latest(time - 1).state,  # Needed to calculate gradients of the weights!
                    h=self.latest(time - 1).state  # Needed to calculate gradients of the weights!
                )
                group.add_error(g_r, time - 1)

                # Accumulating g_h to self:
                self.add_error(g_h, time - 1)
                #print('Back-Rte: ', self.nid() + ' - t' + str(time), ': g_h=' + str(g_h), 'g_r=' + str(g_r))

            #print('Bwd-' + str(self.nid()) + '- e:' + str(current_error))

            if this_is_end: return True, None
            else:
                found = [g for g, r in self.to_conns.items() if not g.at(time+1).is_sleeping]
                assert len(found) == 1
                return True, found[0]  # The best group is being returned!
        return False, None


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
#    assert len(CONTEXT.recorders) == 4 + 2 + 2 + 2 + 2 # four groups and 6 connections

    group.rec(-1).state = torch.tensor([[1.0, 2.0, 3.0]])
    groups = [group, other1, other2, output]
    for g in groups: g.forward(0)

    assert not other1.latest(1).is_sleeping # CHOICE: other1
    assert other2.latest(1).is_sleeping
    assert output.latest(1).is_sleeping
    #assert [r.latest(0).g for r in group.to_conns.values()] == [0.10948651283979416, 0.0009388707112520933]

    # Future states don't know anything:
    assert other1.at(2).is_sleeping
    assert other2.at(2).is_sleeping
    assert output.at(2).is_sleeping

    group.rec(0).state = torch.tensor([[-3.0, -1.0, -4.0]])
    for g in groups: g.forward(1)

    assert not other1.at(1).is_sleeping # first step is still recorded...
    assert other2.at(1).is_sleeping
    #assert [r.latest(0).g for r in group.to_conns.values()] == [0.10948651283979416, 0.0009388707112520933]

    #print([r.latest(1).g for r in group.to_conns.values()])
    assert other1.at(2).is_sleeping # New step as well!
    assert not other2.at(2).is_sleeping # CHOICE: other2
    assert not output.at(2).is_sleeping
    #assert [r.latest(1).g for r in group.to_conns.values()] == [0.881648063659668, 0.9996621608734131]
    #assert [r.latest(1).g for r in other1.to_conns.values()] == [0.5]

    #last activation (activates output)
    group.rec(1).state = torch.tensor([[2.0, 4.0, -1.0]])
    for g in groups: g.forward(2)
    for g in groups: assert g.latest(0).error_count == 0
    print(str(output.at(2).state))
    #assert str(output.at(2).state) == 'tensor([[1.2334]])'#'tensor([[0.9552]])'#'tensor([[5.1944]])'#'tensor([[-0.2231]])'
    output.add_error(torch.tensor([[1]]),2)
    assert output.latest(1).error_count == 0
    assert output.latest(2).error_count == 1
    assert output.latest(3).error_count == 0

    output.backward(2) # backprop happens recursively!

    #grad = str(next(iter(other1.from_conns.values())).Ws.grad)
    #print(grad)
    #assert grad in [
    #    'tensor([[-0.0209,  0.0304, -0.4715],\n        [-0.0070,  0.0101, -0.1572],\n        [-0.0278,  0.0406, -0.6287]])',
    #    'tensor([[-0.0417,  0.0608, -0.9431],\n        [-0.0139,  0.0203, -0.3144],\n        [-0.0556,  0.0811, -1.2574]])'
    #]
    #grad = str(next(iter(other2.from_conns.values())).Ws.grad)
    #assert grad == 'tensor([[0., 0., 0.],\n        [0., 0., 0.],\n        [0., 0., 0.]])'

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





