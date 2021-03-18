import numpy as np
from scipy.spatial import distance
import KNN1
import Utilities as utils


class WKNN1(KNN1.KNN):

    def __init__(self, k):
        super().__init__(k)

    def _k_nearest(self, sample):
        dists = []
        for i, train_sample in enumerate(self.train_samples):
            dists.append(distance.euclidean(sample, train_sample))
        min_k_indexes = np.argpartition(dists, self.k)
        dists.sort()
        return min_k_indexes[:self.k], dists[:self.k]

    def _weight_function(self, dist):
        return 1 / dist

    def _w_get_votes(self, distances_indexes, distances_values):
        votes = [0, 0]
        for idx, val in zip(distances_indexes, distances_values):
            votes[self.train_labels[idx]] += self._weight_function(val)
        return votes

    def _w_majority(self, distances_indexes, distances_values):
        votes = self._w_get_votes(distances_indexes, distances_values)
        return np.argmax(votes)

    def predict(self, samples, features=None) -> list:
        y_predict = []
        if features:
            assert len(features) == len(samples[0])
            samples = self._update_features(samples, features)
        for sample in samples:
            k_nearest_indexes, k_nearest_dists = self._k_nearest(sample)
            y_predict.append(self._w_majority(k_nearest_indexes, k_nearest_dists))
        return y_predict


def main():
    knn_results: utils.KNNResults = utils.run_knn_classifier(WKNN1, 17)
    print(knn_results.confusion_matrix)
    print(knn_results.accuracy)


if __name__ == "__main__":
    main()
