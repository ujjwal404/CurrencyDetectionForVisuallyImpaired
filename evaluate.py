"""Evaluate the model"""

import argparse
import logging
import os
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger


parser = argparse.ArgumentParser()
data_directory = os.getcwd().rsplit("/", 1)[0] + "/data/224x224_currency"
parser.add_argument("--model_dir", default="experiments/test", help="Experiment directory containing params.json")
parser.add_argument("--data_dir", default=data_directory, help="Directory containing the dataset")
parser.add_argument(
    "--restore_from", default="best_weights", help="Subdirectory of model dir or file containing the weights"
)


if __name__ == "__main__":
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = "model/params.json"
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, "evaluate.log"))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, "test_dir")

    # Get the filenames from the test set
    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith(".jpg")]

    test_labels = [int(f.split("/")[-1][0]) for f in test_filenames]

    labelencoder = LabelEncoder()
    test_labels = labelencoder.fit_transform(test_labels)

    # Load the model
    model_dir = os.getcwd().rsplit("/", 1)[0] + "/saved_model"
    model = tf.keras.models.load_model(model_dir)

    for i in range(len(test_filenames)):
        img = tf.keras.preprocessing.image.load_img(test_filenames[i], target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
                labelencoder.inverse_transform([tf.argmax(score)]), 100 * tf.reduce_max(score)
            )
        )

"""
20_sjdfhajks.jpg 
200_sadjhfgk.jpg


"""
