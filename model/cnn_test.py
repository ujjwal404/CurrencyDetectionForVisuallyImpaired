import os

# Turn off TensorFlow warning messages in program output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout


class Image_CNN(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, 3, input_shape=(..., 3), strides=1, activation="relu")
        self.conv2 = Conv2D(32, 3, strides=1, activation="relu")
        self.conv3 = Conv2D(32, 5, strides=2, activation="relu")

        self.pool1 = MaxPool2D(pool_size=(2, 2))
        self.batchnorm = BatchNormalization()
        self.dropout40 = Dropout(rate=0.4)

        self.flatten = Flatten()
        self.d128 = Dense(128, activation="relu")
        self.d10softmax = Dense(10, activation="softmax")

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.batchnorm(x)
        if training:
            x = self.dropout40(x, training=training)

        x = self.flatten(x)
        x = self.d128(x)
        x = self.d10softmax(x)

        return x


@tf.function
def train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels, model, loss_object, test_loss, test_accuracy):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def make_model_test(train_data, eval_data, params):
    import time

    start_time = time.time()

    train_ds = train_data
    test_ds = eval_data

    # Instantiate our neural network model from the predefined class. Also define the loss function and optimizer.
    model = Image_CNN()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    # Define the metrics for loss and accuracy.
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    # Run and iterate model over epochs
    EPOCHS = 5
    for epoch in range(EPOCHS):

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        # Train then test the model
        for images, labels in train_ds:
            train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy)
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels, model, loss_object, test_loss, test_accuracy)

        # Print results
        template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                test_loss.result(),
                test_accuracy.result() * 100,
            )
        )
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
