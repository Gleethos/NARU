
import time
from utility.embedding import Encoder
import torch
from utility.ffnaru import Network
from utility.data_loader import load_jokes
from utility.trainer import exec_trial
from utility.persistence import save_params, load_params
from utility.classes import CONTEXT

# ---------------------------------------------------------------------

CONTEXT.BPTT_limit = 10

# feed-forward-NARU
model = Network(
    depth=5,
    max_height=10,
    max_dim=64,
    max_cone=5,
    D_in=50,
    D_out=50
)

model.set_params(load_params('models/test_model/'))

jokes = load_jokes()
training_data = jokes#[:2]

choice_matrices = exec_trial(
    model=model,
    encoder=Encoder(),
    optimizer=torch.optim.Adam(model.get_params(), lr=0.0001),
    training_data=training_data,
    test_data=jokes[2:4],
    epochs=1
)

print(choice_matrices)

# SAVING PARAMETERS:

target_folder = 'models/feed-forward-NARU_'+time.strftime("%Y%m%d-%H%M%S")+'/'
save_params( model.get_params(), target_folder )
#loaded = load_params(target_folder)

print('FFNN-NARU TEST DONE!')