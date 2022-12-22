"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.input_fn import input_fn
from model.model_fn import make_model
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="experiments", help="Experiment directory containing params.json")

data_directory = os.getcwd().rsplit("/", 1)[0] + "/data/224x224_currency"


parser.add_argument("--data_dir", default=data_directory, help="Directory containing the dataset")
parser.add_argument(
    "--restore_from", default=None, help="Optional, directory or file containing weights to reload before training"
)


if __name__ == "__main__":
    # Set the random seed for the whole graph for reproductible experiments
    tf.random.set_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = "model/params.json"
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, "train.log"))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train_dir")
    dev_data_dir = os.path.join(data_dir, "dev_dir")

    # Get the filenames from the train and dev sets
    train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith(".jpg")]
    eval_filenames = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir) if f.endswith(".jpg")]

    # Labels will be between 0 and 5 included (6 classes in total)
    train_labels = [int(f.split("/")[-1].split("_")[0]) for f in train_filenames]
    eval_labels = [int(f.split("/")[-1].split("_")[0]) for f in eval_filenames]

    labelencoder = LabelEncoder()
    train_labels = labelencoder.fit_transform(train_labels)
    eval_labels = labelencoder.fit_transform(eval_labels)

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)

    train_labels = tf.cast(train_labels, tf.int32)
    eval_labels = tf.cast(eval_labels, tf.int32)

    # Create the two iterators over the two datasets [tensors are returned by the input_fn with images, labels and iterator]

    train_inputs = input_fn(True, train_filenames, train_labels, params)
    eval_inputs = input_fn(False, eval_filenames, eval_labels, params)
    logging.info("Creating the model...")
    make_model(train_inputs, eval_inputs, params)
    # make_model_test(train_inputs, eval_inputs, params)
"""
    # Define the model
    
    train_model_spec = model_fn('train', train_inputs, params)
    #eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
    """
