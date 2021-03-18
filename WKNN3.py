import numpy as np
from scipy.spatial import distance
import WKNN1
import Utilities as utils


class WKNN3(WKNN1.WKNN1):

    def __init__(self, k, m, v):
        super().__init__(k)
        self.v = v
        self.m = m

    def _m_nearest(self, sample):
        dists = []
        for i, train_sample in enumerate(self.train_samples):
            dists.append(distance.euclidean(sample, train_sample))
        min_k_indexes = np.argpartition(dists, self.m)
        return min_k_indexes[:self.m]

    def _weight_function3(self, idx, dist):
        return self.v * self._weight_function1(dist) + (1 - self.v) * self._weight_function2(idx)

    def _weight_function1(self, dist):
        return 1 / dist

    def _weight_function2(self, idx):
        sample = self.train_samples[idx]
        label = self.train_labels[idx]
        m_nearest_indexes = self._m_nearest(sample)
        counter = 0
        for index in m_nearest_indexes:
            if self.train_labels[index] == label:
                counter += 1
        return counter / self.m

    def _w_get_votes(self, distances_indexes, distances_values):
        votes = [0, 0]
        for idx, val in zip(distances_indexes, distances_values):
            votes[self.train_labels[idx]] += self._weight_function3(idx, val)
        return votes


def main():
    accuracies = []
    vals_of_v = [i * 0.01 for i in range(0, 101)]
    for v in vals_of_v:
        knn_results: utils.KNNResults = utils.run_knn_classifier(WKNN3, 17, 6, v)
        accuracies.append(knn_results.accuracy)
    print(vals_of_v)
    print(accuracies)
    utils.draw_graph(vals_of_v, accuracies, "Value Of V", "Accuracy", "Graph of accuracy as a function of V")


def best_v_m():
    accuracies = []
    vals_of_k = [i for i in range(1, 101)]
    for k in vals_of_k:
        knn_results: utils.KNNResults = utils.run_knn_classifier(WKNN3, k, 6, 0.4)
        accuracies.append(knn_results.accuracy)
    print(accuracies)
    print(vals_of_k)
    utils.draw_graph(vals_of_k, accuracies, "Value Of K", "Accuracy",
                     "Graph of accuracy as a function of K with the best m,v")


if __name__ == "__main__":
    main()
