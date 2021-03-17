
import torch
import random
import matplotlib.pyplot as plt


def exec_trial_with_autograd(
        model,
        optimizer,
        encoder,
        training_data,
        test_data=None,
        epochs=10,
        use_custom_backprop = False
):
    print('\nStart trial now!')
    print('Number of training samples:', len(training_data), '; Number of test samples:', len(test_data), ';')
    print('----------------------------------------------------------------------------')

    torch.manual_seed(42)
    # model.to(device) # To GPU - WIP

    epoch_losses = []
    validation_losses = []
    choice_matrices = dict()
    previous_matrices = None
    choice_changes = []

    for epoch in range(epochs):
        # The neural network should learn data more randomly:
        random.Random(666 + epoch + 999).shuffle(training_data)  # ... so we shuffle it! :)
        instance_losses = []
        sentence_losses = []

        for i, sentence in enumerate(training_data):
            vectors = encoder.sequence_words_in(sentence)
            if not use_custom_backprop: choice_matrix, losses = model.train_with_autograd_on(vectors)
            else: choice_matrix, losses = model.train_on(vectors)
            sentence_losses.append(losses)
            instance_losses.append(sum(losses) / len(losses))
            choice_matrices[' '.join(sentence)] = choice_matrix

        optimizer.step()
        optimizer.zero_grad()

        print('Epoch', epoch, ' done! latest loss =', instance_losses[len(instance_losses) - 1],'; Avg loss =', sum(instance_losses)/len(instance_losses), '')
        epoch_losses.append(sum(instance_losses)/len(instance_losses))

        if previous_matrices is not None:
            choice_changes.append(
                number_of_changes(choice_matrices=choice_matrices, previous_matrices=previous_matrices)
            )

        previous_matrices = choice_matrices.copy()

        if test_data is not None:
            validation_losses.append(
                validate(model=model, encoder=encoder, validation_data=test_data)
            )

    print('Trial done! \n===========')
    print('')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle('Horizontally stacked subplots')
    if test_data is not None:
        ax1.plot(instance_losses, 'tab:green')

    ax2.plot(epoch_losses, 'tab:blue')
    ax1.set_title('Validation Losses')
    ax2.set_title('Training Losses')
    plt.show()

    plt.plot(choice_changes)
    plt.title('Route Changes')
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


def number_of_changes(choice_matrices: dict, previous_matrices: dict):
    changes = 0
    for s in choice_matrices.keys():
        if choice_matrices[s] != previous_matrices[s]:
            changes = changes + 1
    return changes



