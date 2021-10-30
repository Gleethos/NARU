
import torch
import random
from lib.net_analysis import load_and_plot, avg_saturation

import collections
import os
import json

import sys


def tame_gradients(
        weights: list,
        accumulation_count: int,
        clip: float = 1.0
):
    for W in weights:
        assert torch.isfinite(W).all()
        assert torch.isfinite(W.grad).all()
        W.grad /= accumulation_count
    torch.nn.utils.clip_grad_norm_(weights, clip)


def exec_trial_with_autograd(
        model,
        optimizer,
        encoder,
        training_data,
        test_data=None,
        epochs=10,
        batch_size=None,
        do_ini_full_batch=True,
        path='models/',
        make_plots=True,
        print_epochs=True
):
    assert epochs > 0

    # The neural network should learn data more randomly:
    random.Random(666 + epochs + 999).shuffle(training_data)  # ... so we shuffle it! :)
    batch_step = 0
    if batch_size is None: batch_size = len(training_data) # We do a full batch!
    else: batch_step = batch_size
    training_data = collections.deque(training_data)

    print('\nStart trial now!')
    print(
        'Number of training samples:', len(training_data), '\n',
        'Number of test samples:', len(test_data), '\n',
        'Batch size: ', batch_size
    )
    print('----------------------------------------------------------------------------')

    torch.manual_seed(42)
    # model.to(device) # To GPU - WIP

    epoch_losses = []
    validation_losses = []
    choice_changes = []
    initial_network_utilisation = None
    all_choices = []
    previous_matrices = None
    choice_matrices = dict()

    # Optionally we can start off by doing a full batch training step!
    # This is so that we initialize the choice matrices dictionary before proceeding.
    # Having a full choice matrices dictionary will enable us to do route change stats!
    if do_ini_full_batch:
        for epoch, sentence in enumerate(training_data):
            choice_matrix, losses = model.train_with_autograd_on(encoder.sequence_words_in(sentence))
            choice_matrices[' '.join(sentence)] = choice_matrix
            sys.stdout.write("\r" + 'Initial full batch training: '+str(epoch+1)+' of '+str(len(training_data))+' completed!')
            sys.stdout.flush()
            tame_gradients(weights=model.get_params(), accumulation_count=1)

        tame_gradients(weights=model.get_params(), accumulation_count=len(training_data))
        optimizer.step()
        optimizer.zero_grad()
        initial_network_utilisation = avg_saturation(choice_matrices=choice_matrices, sizes=model.heights)
        print('\nInitial full batch training step done!')
        print('----------------------------------------------------------------------------')

    print('Looping through epochs now...')
    assert batch_size > 0
    for epoch in range(epochs):

        last_losses = []
        sample_losses = []
        batch = list(training_data)[:batch_size] # Note: we "rotate" the training data after each batch further below!
        for sentence in batch:
            vectors = encoder.sequence_words_in(sentence)
            choice_matrix, losses = model.train_with_autograd_on(vectors)
            last_losses.append(losses[len(losses)-1])
            sample_losses.append(sum(losses) / len(losses))
            choice_matrices[' '.join(sentence)] = choice_matrix

        training_data.rotate(batch_step) # To enable mini batches!
        tame_gradients(weights=model.get_params(), accumulation_count=batch_size)
        optimizer.step()
        optimizer.zero_grad()
        if print_epochs:
            print("Epoch "+str(epoch)+'/'+str(epochs)+': latest token loss avg ='+str(sum(last_losses)/len(last_losses))+'; Avg loss ='+str(sum(sample_losses)/len(sample_losses)))

        epoch_losses.append(sum(sample_losses)/len(sample_losses))

        # If we have previous choices we count the changes! This gives useful insight into the model!
        if previous_matrices is not None:
            choice_changes.append(
                number_of_changes(
                    choice_matrices=choice_matrices,
                    previous_matrices=previous_matrices
                )
            )

        # Here we record choice matrices so that we can compare differences for the next epoch!
        previous_matrices = choice_matrices.copy()
        all_choices.append(previous_matrices)
        if initial_network_utilisation is None:
            initial_network_utilisation = avg_saturation(choice_matrices=choice_matrices, sizes=model.heights)

        if test_data is not None:
            validation_losses.append(
                validate(model=model, encoder=encoder, validation_data=test_data)
            )

    print('Trial done! \n===========')
    print('')

    # saving data now!

    plot_path = path + 'plots/'
    data_path = path + 'data/'

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    with open(data_path+'/data.json', 'w') as fout:
        data = {
            'loss':{'validation':validation_losses, 'training':epoch_losses},
            'route-changes':choice_changes,
            'full-record':all_choices,
            'initial-utilisation':list(initial_network_utilisation),
            'choice-matrices':choice_matrices,
            'heights':model.heights
        }
        json.dump(data, fout, indent=4, sort_keys=True)
        fout.close()

    # done saving data!
    if make_plots:
        load_and_plot(data_path=data_path, plot_path=plot_path)

    return choice_matrices


def validate(model, encoder, validation_data):
    if len(validation_data) == 0:
        print('ERROR! : Validation data list is empty!!')
    sum_loss = 0
    for sentence in validation_data:
        sentence = encoder.sequence_words_in(sentence)
        pred_vecs = model.pred(sentence)
        sum_loss += (
                sum(  # The predicted token is always the next one in the sentence not the current one!
                    [torch.mean((pred_vecs[i] - sentence[i+1]) ** 2) for i in range(len(pred_vecs)-1)]
                ) / len(pred_vecs)
        ).item()
    return sum_loss / len(validation_data)


def number_of_changes(choice_matrices: dict, previous_matrices: dict):
    changes = 0
    for s in choice_matrices.keys():
        if s in previous_matrices.keys():
            if choice_matrices[s] != previous_matrices[s]:
                changes = changes + 1
    return changes










