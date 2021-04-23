
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from utility.net_analysis import epoch_deviations


def exec_trial_with_autograd(
        model,
        optimizer,
        encoder,
        training_data,
        test_data=None,
        epochs=10
):
    print('\nStart trial now!')
    print('Number of training samples:', len(training_data), '; Number of test samples:', len(test_data), ';')
    print('----------------------------------------------------------------------------')

    torch.manual_seed(42)
    # model.to(device) # To GPU - WIP

    epoch_losses = []
    validation_losses = []
    all_choices = []
    previous_matrices = None
    choice_changes = []

    for epoch in range(epochs):
        choice_matrices = dict()

        # The neural network should learn data more randomly:
        random.Random(666 + epoch + 999).shuffle(training_data)  # ... so we shuffle it! :)
        instance_losses = []
        sentence_losses = []

        for i, sentence in enumerate(training_data):
            vectors = encoder.sequence_words_in(sentence)
            choice_matrix, losses = model.train_with_autograd_on(vectors)
            sentence_losses.append(losses)
            instance_losses.append(sum(losses) / len(losses))
            choice_matrices[' '.join(sentence)] = choice_matrix

        optimizer.step()
        optimizer.zero_grad()

        print('Epoch', epoch, ' done! latest loss =', instance_losses[len(instance_losses) - 1],'; Avg loss =', sum(instance_losses)/len(instance_losses), '')
        epoch_losses.append(sum(instance_losses)/len(instance_losses))

        if previous_matrices is not None:
            choice_changes.append(
                number_of_changes(
                    choice_matrices=choice_matrices,
                    previous_matrices=previous_matrices
                )
            )

        all_choices.append(choice_matrices)
        previous_matrices = choice_matrices.copy()

        if test_data is not None:
            validation_losses.append(
                validate(model=model, encoder=encoder, validation_data=test_data)
            )

    print('Trial done! \n===========')
    print('')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle('')
    if test_data is not None:
        ax1.plot(validation_losses, 'tab:green')

    ax2.plot(epoch_losses, 'tab:blue')
    ax1.set_title('Validation Losses')
    ax2.set_title('Training Losses')
    plt.show()

    # Route changes:
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
    plt.show()

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
        if choice_matrices[s] != previous_matrices[s]:
            changes = changes + 1
    return changes
