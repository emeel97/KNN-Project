import numpy as np
import Utilities as utils
from scipy.spatial import distance


class KNN:
    def __init__(self, k: int):
        self.k = k
        self.train_samples = None
        self.train_labels = None

    def reset(self):
        self.train_samples = None
        self.train_labels = None

    def _update_features(self, samples, features):
        updated_samples = []
        for sample in samples:
            updated_samples.append(np.multiply(sample, features))
        return updated_samples

    def fit(self, samples, labels, features=None):
        assert len(samples) == len(labels)
        assert len(samples) >= self.k
        if features:
            assert len(features) == len(samples[0])
            samples = self._update_features(samples, features)
        self.train_samples = samples
        self.train_labels = labels

    def _k_nearest(self, sample):
        dists = []
        for i, train_sample in enumerate(self.train_samples):
            dists.append(distance.euclidean(sample, train_sample))
        # dists.sort(key=lambda element: element[0])
        min_k_indexes = np.argpartition(dists, self.k)
        return min_k_indexes[:self.k]

    def _get_votes(self, distances):
        votes = [0, 0]
        for idx in distances:
            votes[self.train_labels[idx]] += 1
        return votes

    def _majority(self, distances):
        votes = self._get_votes(distances)
        return np.argmax(votes)

    def predict(self, samples, features=None) -> list:
        y_predict = []
        if features:
            assert len(features) == len(samples[0])
            samples = self._update_features(samples, features)
        for sample in samples:
            k_nearest = self._k_nearest(sample)
            y_predict.append(self._majority(k_nearest))
        return y_predict


def main():
    accuracies = []
    vals_of_k = [i for i in range(1, 51)]
    for i in vals_of_k:
        knn_results: utils.KNNResults = utils.run_knn_classifier(KNN, i, normlize_data=False)
        accuracies.append(knn_results.accuracy)
    print(accuracies)
    print(vals_of_k)
    utils.draw_graph(vals_of_k, accuracies, "Value Of K", "Accuracy", "Graph of accuracy as a function of K")


if __name__ == "__main__":
    main()
