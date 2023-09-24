import time
import pandas as pd
from perceptrons.multi_perceptron import *

# random.seed(1)


def main():
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv('../../training_data/mnist_train.csv')

    # Extract labels and pixel values
    labels = data['label'].values
    pixels = data.drop('label', axis=1).values

    # Convert labels to one-hot encoding
    num_classes = 10
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        one_hot_labels[i, labels[i]] = 1

    # Create NumPy arrays with pixel values and one-hot labels
    pixel_arrays = np.array(pixels)
    expected_arrays = np.array(one_hot_labels)

    # Now, pixel_arrays contains the 784 pixel values for each image, and
    # expected_arrays contains the one-hot encoded expected values.

    pixel_arrays = pixel_arrays[:10000]
    expected_arrays = expected_arrays[:10000]

    neuronNetwork = MultiPerceptron(784,
                                    2,
                                    16,
                                    10,
                                    theta_logistic,
                                    theta_logistic_derivative,
                                    0.15,
                                    1)

    start_time = time.time()
    print("starting training")
    error, w_min = neuronNetwork.train(0.35, 100, 0.5, pixel_arrays, expected_arrays, 1)
    end_time = time.time()
    print(f"error:{error}, time:{end_time - start_time}s", )


main()

