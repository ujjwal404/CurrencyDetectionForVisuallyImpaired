"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


def _parse_function(filename, label, size):
    image_string = tf.io.read_file(filename)

    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize(image, [size, size])

    return resized_image, label


def train_preprocess(image, label, use_random_flip):
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def data_augmentation(image, label):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
        ]
    )
    image = data_augmentation(image)
    return image, label


def input_fn(is_training, filenames, labels, params):

    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    train_fn = lambda f, l: train_preprocess(f, l, params.use_random_flip)
    augment_fn = lambda f, l: data_augmentation(f, l)

    if is_training:
        dataset = (
            tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .map(augment_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (
            tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    return dataset
