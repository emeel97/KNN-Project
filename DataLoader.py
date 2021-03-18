import csv
import numpy as np
from itertools import count


def extract_dataset(file_name):
    with open(file_name, "r", newline='') as file:
        csv_reader = csv.reader(file)
        csv_reader_iter = iter(csv_reader)
        label_tag = next(csv_reader_iter)
        labels = []
        samples = []
        for row in csv_reader_iter:
            labels.append(int(row[-1]))
            samples.append(list(map(lambda str: float(str), row[:-1])))
    return np.array(samples), np.array(labels), np.array(label_tag)


def extract_dataset_and_modify(file_name):
    with open(file_name, "r", newline='') as file:
        csv_reader = csv.reader(file)
        label_tag = next(csv_reader)  # skips the header
        file_rows = [row for row in csv_reader]
        positive_label = '1'
        negative_label = '0'
        p_samples_size = len(list(
            filter(lambda item: item[-1] == positive_label, file_rows)))  # number of positive labeled samplesx
        chosen_n_samples = list(filter(lambda item, c=count(): item[-1] == negative_label and next(c) < p_samples_size,
                                       file_rows))  # the first p_samples_size negative samples
        labels = []
        samples = []
        for row in file_rows:
            if (row[-1] == negative_label and row in chosen_n_samples) or row[-1] == positive_label:
                labels.append(int(row[-1]))
                samples.append(list(map(lambda item: float(item), row[:-1])))
    return samples, labels, label_tag
