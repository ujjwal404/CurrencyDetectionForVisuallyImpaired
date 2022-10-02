"""Define the model."""

import tensorflow as tf
from model.cnn import CNN
from model.tfliteconvert import convert_to_tflite
import os


def make_model(train_data, eval_data, params):
    out_dir = os.getcwd().rsplit("/", 1)[0]
    ckpt = os.path.join(out_dir, "experiments/checkpoints")

    # Define optimizer.
    optimizer = tf.optimizers.Adam()

    # Instantiate model. This doesn't initialize the variables yet.
    model = CNN(num_classes=params.num_labels + 1, checkpoint_directory=ckpt, params=params)
    # compile model. This initializes the variables.
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit_dataset(train_data, eval_data)
    # save the model
    tf.saved_model.save(model, os.path.join(out_dir, "saved_model"))
    # convert to tflite model
    convert_to_tflite(out_dir)


def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs["images"]

    # assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]

    out = images
    print(out.get_shape().as_list())

    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8, num_channels * 16]

    # L2 regulariazation
    regularizer = tf.keras.regularizers.L2(0.1)
    for i, c in enumerate(channels):
        with tf.compat.v1.variable_scope("block_{}".format(i + 1)):
            out = tf.compat.v1.layers.Conv2D(out, c, 3, padding="same", kernel_regularizer=regularizer)
            if params.use_batch_norm:
                out = tf.compat.v1.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.compat.v1.layers.max_pooling2d(out, 2, 2)
            print(out.get_shape().as_list())

    # assert out.get_shape().as_list() == [None, 8, 8, num_channels * 8]

    out = tf.reshape(out, [-1, 4 * 4 * num_channels * 16])

    with tf.compat.v1.variable_scope("fc_1"):
        out = tf.compat.v1.layers.dense(out, num_channels * 16)
        if params.use_batch_norm:
            out = tf.compat.v1.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
    with tf.compat.v1.variable_scope("fc_2"):
        logits = tf.compat.v1.layers.dense(out, params.num_labels)

    return logits


def model_fn(mode, inputs, params, reuse=False):
    is_training = mode == "train"
    labels = inputs["labels"]
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.compat.v1.variable_scope("model", reuse=reuse):
        logits = build_model(is_training, inputs, params)
        predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.compat.v1.variable_scope("metrics"):
        metrics = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            "loss": tf.metrics.mean(loss),
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.image("train_image", inputs["images"])

    # TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs["images"], mask_label)
        tf.summary.image("incorrectly_labeled_{}".format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec["variable_init_op"] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec["loss"] = loss
    model_spec["accuracy"] = accuracy
    model_spec["metrics_init_op"] = metrics_init_op
    model_spec["metrics"] = metrics
    model_spec["update_metrics"] = update_metrics_op
    model_spec["summary_op"] = tf.summary.merge_all()

    if is_training:
        model_spec["train_op"] = train_op

    return model_spec
