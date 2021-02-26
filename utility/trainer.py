
import torch
import random
import matplotlib.pyplot as plt


def exec_trial(
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

    instance_losses = []
    sentence_losses = []
    validation_losses = []
    choice_matrices = dict()

    for epoch in range(epochs):
        # The neural network should learn data more randomly:
        random.Random(666 + epoch + 999).shuffle(training_data)  # ... so we shuffle it! :)

        optimizer.zero_grad()

        for i, sentence in enumerate(training_data):
            print('Training on sentence', i, ': "', sentence, '"')
            vectors = encoder.sequence_words_in(sentence)
            choice_matrix, losses = model.train_on(vectors)
            sentence_losses.append(losses)
            instance_losses.append(sum(losses) / len(losses))
            choice_matrices[' '.join(sentence)] = choice_matrix

        for W in model.get_params(): W.grad /= (len(training_data))

        #for W in model.get_params(): W += W.grad
        optimizer.step()

        print('===================================================')
        print('Epoch', epoch, ' done! loss =', instance_losses[len(instance_losses) - 1],'; Avg =', sum(instance_losses)/len(instance_losses), '\n')

        if test_data is not None:
            sum_loss = 0
            for sentence in test_data:
                sentence = encoder.sequence_words_in(sentence)
                preds = model.pred(sentence)
                #print(' '.join(encoder.sequence_vecs_in(preds)))
                sum_loss += (
                    sum(
                        [torch.mean((preds[i] - sentence[i]) ** 2) for i in range(len(preds))]
                    ) / len(preds)
                ).item()

            validation_losses.append(sum_loss / len(test_data))

    print('Trial done! \n===========')
    print(validation_losses)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle('Horizontally stacked subplots')
    ax1.plot(instance_losses, 'tab:green')
    if test_data is not None:
        ax1.plot(validation_losses)

    ax2.plot([item for sublist in sentence_losses for item in sublist], 'tab:blue')
    ax1.set_title('Epoch Losses')
    ax2.set_title('Word Losses')

    return choice_matrices

