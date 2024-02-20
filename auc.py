import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics


def compute_matches(file_train, file_test, n, r):
    # train using the given training file, and compute the scores for the given test file
    cmd = f'java -jar negsel2.jar -self {file_train} -n {n} -r {r} -c -l < {file_test}'
    p = os.popen(cmd)
    return list(map(float, p.read().strip().split('\n')))  # parse as list of floats


def compute_scores(file_train, file_test_self, file_test_anomaly, n, r):
    # compute the self and anomaly scores using the java program
    scores_self = compute_matches(file_train, file_test_self, n, r)
    scores_anomaly = compute_matches(file_train, file_test_anomaly, n, r)

    # label the scores as self/anomaly, i.e. false/true
    scores = np.array(
        [[score, False] for score in scores_self] +
        [[score,  True] for score in scores_anomaly]
    )

    # sort the scores
    return scores[scores[:, 0].argsort()], len(scores_self), len(scores_anomaly)


def compute_stats(scores, num_self, num_anomaly):
    # compute all sensitivity and specificity value in O(n)
    sensitivities = []
    specificities = []

    count_anomaly = 0
    for cutoff_index, [score, anomalous] in enumerate(scores):
        if anomalous:
            count_anomaly += 1

        sensitivities.append((num_anomaly - count_anomaly) / num_anomaly)
        specificities.append(1 - (cutoff_index - count_anomaly) / num_self)

    sensitivities = [1] + sensitivities + [0]
    specificities = [1] + specificities + [0]

    auc = sklearn.metrics.auc(specificities, sensitivities)
    return sensitivities, specificities, auc


def generate_plot(sensitivities, specificities, auc, file_train, file_test_self, file_test_anomaly, n, r):
    plt.subplots(figsize=(8, 8))
    plt.style.use('bmh')

    plt.plot(specificities, sensitivities, color='#0e1111')
    plt.plot([0, 1], [0, 1], '--', color='#0e1111')

    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')

    plt.title(
        f'Trained on {file_train}, Tested on {file_test_self} and {file_test_anomaly}\n\
        Using n = {n} and r = {r}, AUC = {auc}'
    )
    plt.show()


def compute_auc(file_train, file_test_self, file_test_anomaly, n, r):
    scores, num_self, num_anomaly = compute_scores(file_train, file_test_self, file_test_anomaly, n, r)
    sensitivities, specificities, auc = compute_stats(scores, num_self, num_anomaly)
    generate_plot(sensitivities, specificities, auc, file_train, file_test_self, file_test_anomaly, n, r)


def main():
    file_train = 'english.train'
    file_test_self = 'english.test'
    file_test_anomaly = 'tagalog.test'

    n = 10
    r = 4

    compute_auc(file_train, file_test_self, file_test_anomaly, n, r)


if __name__ == '__main__':
    main()
