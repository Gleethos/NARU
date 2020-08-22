
# LARA is all you need #
## *Learned Activity Routing Agent* ##

In this paper we want to propose a new type of neural network model
coupled with a novel layer routing algorithm 
which tries to solve wide spread issues regarding current cutting edge network architectures.
The principle behind the following algorithm has been heavily 
inspired by 

[
    Mason McGill, Pietro Perona : <br> 
    Dynamic Routing in Artificial Neural Networks
](https://arxiv.org/pdf/1703.06217.pdf)

and 

[
    Sara Sabour, Nicholas Frosst, Geoffrey E Hinton : <br> 
    Dynamic Routing Between Capsules
](https://arxiv.org/abs/1710.09829), 

as well as 

[
    Noam Shazeer, Azalia Mirhoseini†, Krzysztof Maziarz, Andy Davis, Quoc Le, GeoffreyHinton and Jeff Dean : <br>
    THE SPARSELY GATED MIXTURE OF EXPERTS LAYER
](https://arxiv.org/pdf/1701.06538.pdf).


However these Models suffer major issues concerning efficient training and further scaling.

The Model described within this paper tries to explore sparse neural activity 
of sparse connections as it can be observed in biological systems.

The layer routing algorithm allows for arbitrary interconnectedness 
between multiple layers in a multi-directional fashion. 
A System based on this algorithm would allow information to flow 
freely from layer to layer, backwards as well as forward depending on it's connections. 
Most layers would not need to be active for as long as they are not being activated by an active layer. 
Activity is free to travel through the entire network. This System also allows for 
dynamic layer creation and deletion if layers are not attractive for 
the "activity thread" propagating through the network.

Although the term layer is being used this type of algorithm does n
ot necessitate the network to be a feed forward architectures.

*The System will now be described by answering the following questions:*

---

### How are layers trained if they are inactive potentially most of the time? ###

Assuming layer A is sometimes active and is connected to layer B. <br>
Layer B however is not active as it has not been "chosen" by layer A (or any other layer), 
B will still receive error values due to the fact that A is connected to B.
Layer A can reuse the computed input value of it's Weighted matrix multiplication with 
B as long as B stays inactive and does not change it's activation state.
During Backpropagation B will further accumulate it's error.

### What if the accumulated error of layer B becomes too large? ###

B and every other layer keeps track of it's behavioral history.  <br>
The accumulated error is divided by the number of time steps B has been inactive up until reactivation. 
This value will be referred to as "resting phase".

### How does a layer decide which one to activate next? ###

Consider layer A, B, C and D.
Layer A has been previously activated by layer X and activates on it's own within the next step.
It is connected to layer B, C, D and X (maybe even itself) and has an input I for every connection.
Therefore a single scalar neuron within layer A will possess the following scalar inputs:
Iab, Iac, Iad, Iax;
Routing happens through a voting process accomplished by a group of gating functions. 
Every connection from one layer to another has a single scalar gate which 
modifies the calculated input towards that layer and ultimately determines 
the importance of the connected layer.
We can therefore derive the following gates for layer A:
Gab, Gac, Gax.
The following has to be added:
These gates are used by layer A only. If B is connected to A, 
then B would have the Gate Gba, C would have Gca, etc...

Gates are calculated by at least two components. 
The Gate Gab for example has to have a weighted connections to B and it's host layer A! 
Optionally it may also be calculated by being connected to every other layer within A's connections. 
However this approach was not chosen in order to keep the algorithm simple and efficient.
 
Before A activates fully, every Gate produces it's output, most preferably between 0 and 1. 
This is achieved by using a sigmoid, gaussian or softmax function.
These values are then used as modfiers for the calculated input values of A.
A has an input value for every connection.
Consider the following input value:
Iabi is the i'th vector product of layer B and A's i'th weight vector for its connection to A. 
The activation for A's i'th neuron is calculated by: 
a(Iabi * Gab + Iaci * Gac + Iadi * Gad + Iaxi * Gax)
a is the main activation function of a neuron. 
We recommend to use the Rectefier function as it posesses both gating and linearity functionality.
It might be possible to not use an activation function at all because 
non-linearity is already given by the gating function!

The next layer will be chosen simply by picking the layer whose Gating value 
is largest compared to every other value.

### Why would this work? ###

The proposed system describes layers as independent nodes which posess 
the ability to route through a layer network based on the assumption of a sort of "agency".
This might sound unintuitive at first, however there is real merit to this upon further thought.
The state of a single neuron is inherently one dimensional, 
this fact denies it the ability to encode complex (multi dimensional) states. 
Layers however can contain such information. 
Wth layers (Sets of neurons) there is a very real possibility in seeing them as 
states of "neural thought". 
A layers activation may therefore be perfectly capable of encoding information 
about where to "go next" within a network of connected layers.

Why would layers converge towards a state in which information is 
routed diversively and processed efficiently and effective?

The short answer to this question would simply be: 
"You'll know where to go by walking and looking"
To elaborate further:

As soon as a layers gating functions have generated their 
values it becomes clear which connection will be activated next.
Let's assume the network finds itself in a state in which 
one connection is always preferable to any other. 
This connection will be trained thoroughly and may become extremely useful for the network.
However as soon as an activation state is being processed which 
does not fit the layer well, meaning the state of the connected 
layer becomes unfit for the responsible gating function is therefore not ideal anymore.
To prevent routing bias, we simply remove the bias of the gating 
functions as well as the biases of any other function within a layer.
If an input were to have a bias, the layer would be able to 
converge towards an ideal state internally possibly without 
ever 'considering' the outside input.

To put it shortly:
An inherent convergable state would make the search of the routing 
algorithm inherently invalid : the best state!

In the proposed architecture the functionality of biases is replaced 
by the fact that inactive layers (and the inputs they generate within other layers) 
are already acting as semi stable biases.
As soon as a layer starts relying on it's internal state 
(it's set of inputs, which have been inactive for some time), it will favour them and 
therefore cause the gating function to grant them more importance.
This then leads to reactivation which changes the state and most likely the distribution 
of perceived 'usefulness' to other layers.

New and unknown information will therefore be routed randomly 
until "it comes across" layers who can handle it most well, 
allowing the network to process data evenly and also learn 
without getting stuck in a local minima 
(one layer or cascade of such which is always preferred and active).

Biases would pose a thread to the mechanism described above.

How can the network be stable without biases?

In a sense biases provide 'default' / 'fallback' states which 'save' neurons 
from exploding and or vanishing values. 
In a LARA network however such a scenario is definitely a thread.
We therefore use Gaussian activation functions for the gates. 
This makes sure that exploding or out of range inputs to a given layer will always be nullified!
A sigmoid function on the other hand would make inputs 
uncapped to either - or + infinity depending on the sign of a given weight scalar.

---

WRONG:
In order to get the best of both world we decided to allow for dead biases which are simply fixed values unable to change.

---------------

Because we use semi static input states as a replacement for biasis we 
also sabotage untrained to start off exploratory as the initaial input 
state will not at all be ideal. It will however not become ideal if it's not trainable like biases are. 
We therefore allow layers to train their weight matrices even though the connections they serve might be inactive.

This allows them to serve the purpose of biases and also 


### General predictions: ###

As established before, inactive inputs will act as pseudo biases. 
Due to the fact that a layer is most likely connected to multiple inputs which are most likely not active all at once (because allowing too much activity would diminish performance), these act as a sort of sub set of activation. Premature activation states which are yet to be combined with each other to form an output.
These pseudo biases form a sort of memory state. 
For every input within layer a there is a vector corresponding to the size of layer a!
Due to the fact that every gating function within layer a will use a as input recurrently...
Combinatory effect on input set....
Blaa


