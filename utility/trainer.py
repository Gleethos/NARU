
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from utility.net_analysis import epoch_deviations

import collections
import os


def exec_trial_with_autograd(
        model,
        optimizer,
        encoder,
        training_data,
        test_data=None,
        epochs=10,
        batch_size=None,
        do_ini_full_batch=True,
        plot_path='models/plots/'
):
    assert epochs > 0
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

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
            print('Initial full batch training:', epoch+1, 'of', len(training_data), 'completed!')
        for W in model.get_params(): W.grad /= len(training_data)
        optimizer.step()
        optimizer.zero_grad()
        initial_network_utilisation = avg_saturation(choice_matrices=choice_matrices, sizes=model.heights)
        print('Initial full batch training step done!\n...')

    for epoch in range(epochs):

        instance_losses = []
        batch = list(training_data)[:batch_size]
        for sentence in batch:
            vectors = encoder.sequence_words_in(sentence)
            choice_matrix, losses = model.train_with_autograd_on(vectors)
            instance_losses.append(sum(losses) / len(losses))
            choice_matrices[' '.join(sentence)] = choice_matrix

        training_data.rotate(batch_step) # To enable mini batches!
        for W in model.get_params(): W.grad /= batch_size
        optimizer.step()
        optimizer.zero_grad()

        print('Epoch', epoch, ' done! latest loss =', instance_losses[len(instance_losses) - 1],'; Avg loss =', sum(instance_losses)/len(instance_losses), '')
        epoch_losses.append(sum(instance_losses)/len(instance_losses))

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

    # Training and Validation:
    #------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle('')
    if test_data is not None:
        ax1.plot(validation_losses, 'tab:green')

    ax2.plot(epoch_losses, 'tab:blue')
    ax1.set_title('Validation Losses')
    ax2.set_title('Training Losses')
    plt.savefig(plot_path+'loss.png')
    plt.savefig(plot_path+'loss.pdf')
    plt.show()

    # Route changes:
    #--------------
    plt.bar(
        range(len(choice_changes)),
        choice_changes,
        width=1.0,
        label='number of route changes per epoch',
        fc=(0, 0, 1, 0.25)
    )
    # Smooth lines:
    plt.plot(
        moving_average(np.array(choice_changes), 32),
        '--',
        label='32 epoch moving average',
        color='green'
    )
    plt.xlabel("epoch")
    plt.ylabel("number of route changes")
    # Title:
    plt.title('Route Changes')
    plt.legend()
    plt.savefig(plot_path+'route-changes.png')
    plt.savefig(plot_path+'route-changes.pdf')
    plt.show()

    # Routing Bias:
    #-------------
    deviations = epoch_deviations(all_matrices=all_choices, sizes=model.heights)
    plt.plot(
        deviations,
        '-',
        label='routing bias',
        color='blue'#fc=(0, 0, 1, 0.25)
    )
    # Smooth line:
    plt.plot(
        moving_average(np.array(deviations), 32),
        '--',
        label='32 epoch moving average',
        color='green'
    )
    plt.plot(
        moving_average(np.array(deviations), 64),
        '-.',
        label='64 epoch moving average',
        color='red'
    )
    plt.xlabel("epoch")
    plt.ylabel("standard deviation")
    # Title:
    plt.title('Routing Bias')
    plt.legend()
    plt.savefig(plot_path+'routing-bias.png')
    plt.savefig(plot_path+'routing-bias.pdf')
    plt.show()

    # Cumulative saturation:
    #----------------------

    plt.plot(
        avg_saturation(choice_matrices=choice_matrices, sizes=model.heights),
        '-',
        label='cumulative network utilisation',
        color='green'
    )
    plt.plot(
        initial_network_utilisation,
        '-.',
        label='initial cumulative network utilisation',
        color='blue'
    )
    plt.xlabel("token index")
    plt.ylabel("cumulative network utilisation")
    # Title:
    plt.title('Average Cumulative Network Utilization')
    plt.legend()
    plt.savefig(plot_path+'cumulative-network-utilisation.png')
    plt.savefig(plot_path+'cumulative-network-utilisation.pdf')
    plt.show()

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


def moving_average(x, w):
    filter = np.ones(w) / w
    return np.convolve(
                x,
                filter,
                'valid' # Maybe use full?
            )


def number_of_changes(choice_matrices: dict, previous_matrices: dict):
    changes = 0
    for s in choice_matrices.keys():
        if s in previous_matrices.keys():
            if choice_matrices[s] != previous_matrices[s]:
                changes = changes + 1
    return changes


def avg_saturation(choice_matrices: dict, sizes: list):
    structure = [ [c != 1] * c for c in sizes ]
    conditional = [item for sublist in structure for item in sublist].count(True)
    choice_matrices = choice_matrices.values()
    token_index_counts = dict()
    for sentence in choice_matrices:
        structure = [ [c != 1] * c for c in sizes ]
        for i, token in enumerate(sentence):
            if i not in token_index_counts:
                token_index_counts[i] = []

            for capsule_index, choice in enumerate(token):
                structure[capsule_index][choice] = False

            saturation = [item for sublist in structure for item in sublist].count(True)
            token_index_counts[i].append(1-(saturation / conditional))


    for i, v in token_index_counts.items():
        token_index_counts[i] = (sum(v) / len(v))

    return token_index_counts.values()












