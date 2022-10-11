import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from model.utils import Params
from model.input_fn import input_fn
from model.cnn import CNN

test_dir = os.getcwd().rsplit("/", 1)[0] + "/data/224x224_currency/test_dir"

if __name__ == "__main__":
    # get params
    json_path = "model/params.json"
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    test_filenames = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".jpg")]
    test_labels = [int(f.split("/")[-1].split("_")[0]) for f in test_filenames]

    params.test_size = len(test_filenames)

    labelencoder = LabelEncoder()
    test_labels = labelencoder.fit_transform(test_labels)

    test_labels = tf.cast(test_labels, tf.int32)

    test_df = input_fn(False, test_filenames, test_labels, params)

    # EVALUATE MODEL
    ckpt = os.path.join(os.getcwd(), "experiments/checkpoints")

    # Instantiate model. This doesn't initialize the variables yet.
    model = CNN(num_classes=params.num_labels, checkpoint_directory=ckpt, params=params)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
    )
    model_dir = os.path.join(os.getcwd(), "experiments/saved_weights/weight")
    model.load_weights(model_dir)

    test_accuracy = model.evaluate(test_df)
    print("Test set accuracy: {:5.2f}%".format(100 * test_accuracy))
