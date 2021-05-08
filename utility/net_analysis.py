
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np


def avg_sequence_length(data):
    data = [len(s) for s in data]
    return sum(data) / len(data)


def epoch_deviations(all_matrices: list, sizes: list):
    return [all_sentences_of_epoch_average(m, sizes=sizes) for m in all_matrices]


def all_sentences_of_epoch_average(epoch_matrices: dict, sizes: list):
    sum = 0
    for sentence, choices in epoch_matrices.items():
        sum += total_deviation_for(choices, sizes)

    return sum / len(epoch_matrices)


def total_deviation_for(epoch_matrices: list, sizes: list): # TODO: Rename this from epoch_matrices! to matrix
    all_flattened = epoch_matrices#[] # TODO: Maybe revisit if numbers are still crazy large!
    #for sentence, choices in epoch_matrices.items():
    #    all_flattened.extend(choices)

    relative_counts, total_counts = choice_avg(matrix=all_flattened, sizes=sizes)

    # In a perfect world we would see the following relative occurrences:
    expected_averages = []
    for i, total in enumerate(total_counts):
        expected_averages.append(total / sizes[i])

    # Let's calculate the accumulated difference:
    total_deviation = 0
    N = 0
    for i, total in enumerate(total_counts):
        for pos, count in relative_counts[i].items():
            total_deviation += abs( expected_averages[i] - count )**2
            N += 1

    return (total_deviation / N) ** 0.5


def choice_avg(matrix: list, sizes: list):
    total_counts = [0] *len(sizes)
    for row in matrix:
        for i, e in enumerate(row):
            if e >= 0: total_counts[i] += 1

    # Now we create a dict for every layer to count relative occurrences!
    relative_counts = [{i: 0 for i in range(s)} for s in sizes]
    # Let's count the relative occurrences:
    for row in matrix:
        for i, e in enumerate(row):
            if e >= 0: relative_counts[i][e] += 1

    assert len(relative_counts) == len(total_counts)
    return relative_counts, total_counts


test_data = [
    {'hi': [[0, -1], [0, 1], [1, 1], [-1, 0]], 'hu': [[0, -1], [1, 1], [1, 0], [-1, 0]]},
    {'ho': [[0, -1], [0, 1], [1, 1], [-1, 1]], 'hu': [[0, -1], [1, 1], [1, 0], [-1, 0]]},
    {'he': [[0, -1], [0, 0], [1, 0], [-1, 0]], 'hu': [[0, -1], [1, 0], [1, 0], [-1, 0]]}
]

#print(epoch_deviations(all_matrices=test_data, sizes=[2, 2]))
assert epoch_deviations(all_matrices=test_data, sizes=[2, 2]) == [0.5, 0.8090169943749475, 1.118033988749895]#[0.0, 0.7071067811865476, 2.1213203435596424]


def moving_average(x, w):
    filter = np.ones(w) / w
    return np.convolve(
                x,
                filter,
                'valid' # Maybe use full?
            )


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


def load_and_plot(data_path, plot_path):
    sns.set_theme()
    with open(data_path+'/data.json') as json_file:
        data = json.load(json_file)
        json_file.close()
        validation_losses = data['loss']['validation']
        epoch_losses = data['loss']['training']
        choice_changes = data['route-changes']
        all_choices = data['full-record']
        choice_matrices = data['choice-matrices']
        initial_network_utilisation = data['initial-utilisation']
        heights = data['heights']

        # Training and Validation:
        # ------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
        fig.suptitle('')
        if test_data is not None:
            ax1.plot(validation_losses, 'tab:green')

        ax2.plot(epoch_losses, 'tab:blue')
        ax1.set_title('Validation Losses')
        ax2.set_title('Training Losses')
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("loss")
        plt.savefig(plot_path + 'validation-and-training-loss.png', dpi=200)
        plt.savefig(plot_path + 'validation-and-training-loss.pdf')
        plt.show()

        # Route changes:
        # --------------
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
        plt.savefig(plot_path + 'route-changes.png', dpi=200)
        plt.savefig(plot_path + 'route-changes.pdf')
        plt.show()

        # Routing Bias:
        # -------------
        deviations = epoch_deviations(all_matrices=all_choices, sizes=heights)
        plt.plot(
            deviations,
            '-',
            label='routing bias',
            color='blue'  # fc=(0, 0, 1, 0.25)
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
        plt.savefig(plot_path + 'routing-bias.png', dpi=200)
        plt.savefig(plot_path + 'routing-bias.pdf')
        plt.show()

        # Cumulative saturation:
        # ----------------------

        plt.plot(
            avg_saturation(choice_matrices=choice_matrices, sizes=heights),
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
        plt.title('Average Cumulative Network Utilisation')
        plt.legend()
        plt.savefig(plot_path + 'cumulative-network-utilisation.png', dpi=200)
        plt.savefig(plot_path + 'cumulative-network-utilisation.pdf')
        plt.show()
