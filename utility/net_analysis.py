
def epoch_deviations(all_matrices: list, sizes: list):
    return [total_deviation_for(m, sizes=sizes) for m in all_matrices]


def total_deviation_for(epoch_matrices: dict, sizes: list):
    all_flattened = []
    for sentence, choices in epoch_matrices.items():
        all_flattened.extend(choices)

    relative_counts, total_counts = choice_avg(matrix=all_flattened, sizes=sizes)

    # In a perfect world we would see the following relative occurrences:
    expected_averages = []
    for i, total in enumerate(total_counts):
        expected_averages.append(total /sizes[i])

    # Let's calculate the accumulated difference:
    total_deviation = 0
    N = 0
    for i, total in enumerate(total_counts):
        for pos, count in relative_counts[i].items():
            total_deviation += abs( expected_averages[i] - count )**2
            N += 1

    return (total_deviation / N) ** 0.5


def choice_avg(matrix: list, sizes: list):
    total_counts = [0 ] *len(sizes)
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

assert epoch_deviations(all_matrices=test_data, sizes=[2, 2]) == [0.0, 0.7071067811865476, 2.1213203435596424]
