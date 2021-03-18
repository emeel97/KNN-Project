import numpy as np
from scipy.spatial import distance
import WKNN1
import Utilities as utils


class WKNN2(WKNN1.WKNN1):

    def __init__(self, k, m):
        super().__init__(k)
        self.m = m

    def _m_nearest(self, sample):
        dists = []
        for i, train_sample in enumerate(self.train_samples):
            dists.append(distance.euclidean(sample, train_sample))
        min_k_indexes = np.argpartition(dists, self.m)
        return min_k_indexes[:self.m]

    def _weight_function(self, idx):
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
            votes[self.train_labels[idx]] += self._weight_function(idx)
        return votes


def main():
    accuracies = []
    vals_of_m = []
    for i in range(1, 101):
        knn_results: utils.KNNResults = utils.run_knn_classifier(WKNN2, 17, i)
        vals_of_m.append(i)
        accuracies.append(knn_results.accuracy)
    utils.draw_graph(vals_of_m, accuracies, "Value Of M", "Accuracy", "Graph of accuracy as a function of M")
    print(accuracies)
    print(vals_of_m)


if __name__ == "__main__":
    main()
