from utility.embedding import load_word_vec_dict
import torch
from utility.ffnaru import Network
from utility.data_loader import load_jokes
import matplotlib.pyplot as plt
import random  # for shuffling the training data

# ---------------------------------------------------------------------


# Word to index encoding...
class Encoder:

    def __init__(self, training_data=None):
        self.word_to_vec = load_word_vec_dict()

    def sequence_words_in(self, seq):
        return [self.word_to_vec[w] for w in seq]


# ---------------------------------------------------------------------

# feed-forward-NARU
model = Network(
    depth=5,
    max_height=15,
    max_dim=500,
    max_cone=39,
    D_in=50,
    D_out=50
)

encoder = Encoder()

jokes = load_jokes()


# ---------------------------------------------------------------------


def exec_trial(
        training_data,
        test_data=None,
        epochs=300
):
    torch.manual_seed(42)
    # model.to(device) # To GPU - WIP

    instance_losses = []
    sentence_losses = []
    validation_losses = []

    for epoch in range(epochs):
        # The neural network should learn data more randomly:
        random.Random(666 + epoch + 999).shuffle(training_data)  # ... so we shuffle it! :)

        for sentence in training_data:
            sentence = encoder.sequence_words_in(sentence)

            losses = model.train_on(sentence)

            sentence_losses.append(losses)
            instance_losses.append([sum(losses) / len(losses)])

        print('Epoch', epoch, 'loss =', instance_losses[len(instance_losses) - 1])

        if test_data != None:

            sum_loss = 0
            for sentence in test_data:
                sentence = encode.sequence_words_in(sentence)
                preds = model.pred(sentence)

                sum_loss += sum([((preds[i] - sentences[i]) ** 2).mean(axis=0) for i in range(len(preds))])

            validation_losses.append(sum_loss / len(test_data))

    print('Trial done! \n===========')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle('Horizontally stacked subplots')
    ax1.plot(instance_losses, 'tab:green')
    if test_data != None:
        ax1.plot(validation_losses)
    ax2.plot(sentence_losses, 'tab:blue')
    ax1.set_title('Epoch Losses')
    ax2.set_title('Word Losses')

    return model


training_data = [j.split() for j in jokes]

exec_trial(training_data)