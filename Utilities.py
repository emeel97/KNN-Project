from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import DataLoader as dl
from typing import NamedTuple
from matplotlib import pyplot as plt


class KNNResults(NamedTuple):
    accuracy: float
    confusion_matrix: list
    error_w: int

    def __str__(self):
        return f"Accuracy={self.accuracy}\nError_w={self.error_w}\n{self.confusion_matrix}"


def accuracy(confusion_matrix):
    TN, FP, FN, TP = confusion_matrix.ravel()
    total = TP + FP + FN + TN
    hit = TP + TN
    return (hit / total) * 100, hit, (total - hit)


def Error_w(confusion_matrix):
    # tn, fp, fn, tp
    TN, FP, FN, TP = confusion_matrix.ravel()
    return 4 * FN + FP


def print_results(confusion_matrix):
    acc = accuracy(confusion_matrix)
    print(f"{confusion_matrix}")


def minmaxNormalization(train_samples, test_samples):
    scaller = MinMaxScaler(copy=False)
    scaller.fit(train_samples)

    scaller.transform(train_samples)
    scaller.transform(test_samples)


def run_knn_classifier(knn_ctor, k: int, m=0, v=-1, normlize_data=False, features_subset=None) -> KNNResults:
    train_dataset_path = "train.csv"
    test_dataset_path = "test.csv"
    train_samples, train_labels, _ = dl.extract_dataset(train_dataset_path)
    test_samples, test_labels, _ = dl.extract_dataset(test_dataset_path)
    if normlize_data:
        minmaxNormalization(train_samples, test_samples)  # Data Normalization
    knn_classifier = None
    if m == 0:
        if v == -1:
            knn_classifier = knn_ctor(k)
        else:
            knn_classifier = knn_ctor(k,0,v)
    else:
        if v == -1:
            knn_classifier = knn_ctor(k, m)
        else:
            knn_classifier = knn_ctor(k, m, v)
    knn_classifier.fit(train_samples, train_labels, features=features_subset)
    knn_predict = knn_classifier.predict(test_samples, features=features_subset)
    _accuracy = metrics.accuracy_score(test_labels, knn_predict)
    _c_m = metrics.confusion_matrix(test_labels, knn_predict)
    _error_w = Error_w(_c_m)
    return KNNResults(accuracy=_accuracy, confusion_matrix=_c_m, error_w=_error_w)


def draw_graph(x_axis_data: list, y_axis_data: list, x_label, y_label, graph_title):
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label, color='b')
    ax.set_ylabel(y_label, color='b')
    ax.tick_params(colors='r')
    ax.plot(x_axis_data, y_axis_data, 'b-')
    fig.tight_layout()
    plt.title(graph_title)
    plt.show()
