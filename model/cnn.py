import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time


class CNN(tf.keras.Model):
    def __init__(self, num_classes, device="cpu:0", checkpoint_directory=None, params=None):

        super(CNN, self).__init__()
        # alexnet implementation
        self.cnn1 = tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)

        self.cnn2 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)

        self.cnn3 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, activation="relu")
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.cnn4 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, activation="relu")
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.cnn5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, activation="relu")
        self.bn5 = tf.keras.layers.BatchNormalization()

        self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(4096, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(4096, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.classifier = tf.keras.layers.Dense(num_classes)

        self.device = device
        self.checkpoint = tf.train.Checkpoint(model=self)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        self.params = params

    def predict(self, inputs, training):
        # alexnet implementation
        x = self.cnn1(inputs)
        x = self.bn1(x, training=training)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.bn2(x, training=training)
        x = self.maxpool2(x)
        # x = self.cnn3(x)
        # x = self.bn3(x, training=training)
        # x = self.cnn4(x)
        # x = self.bn4(x, training=training)
        # x = self.cnn5(x)
        # x = self.bn5(x, training=training)
        # x = self.maxpool3(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        # x = self.dropout2(x, training=training)
        output = self.classifier(x)

        return output

    def loss_fn(self, y, logits):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y, logits)

    def restore_model(self):
        """Function to restore trained model."""

    def save_model(self, global_step=0):
        """function to save model"""
        self.checkpoint.save(self.checkpoint_prefix)

    def compute_accuracy(self, eval_data):
        total, total_correct = 0.0, 0

        for x, y in eval_data:
            logits = self.predict(x, training=False)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_correct += int(correct)
            total += x.shape[0]
        return total_correct / total

    def visualize(self, train_loss_results, train_accuracy_results):
        fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle("Training Metrics")

        axes[0].set_ylabel("Loss", fontsize=14)
        axes[0].plot(train_loss_results)

        axes[1].set_ylabel("Accuracy", fontsize=14)
        axes[1].set_xlabel("Epoch", fontsize=14)
        axes[1].plot(train_accuracy_results)
        plt.show()

    def fit_dataset(self, train_data, eval_data):
        train_loss_results = []
        train_accuracy_results = []

        # Prepare the metrics.
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_loss_avg = tf.keras.metrics.Mean()

        for epoch in range(self.params.num_epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            # total 8659 images in train folder, 32 batches, 270 steps per epoch
            for step, (x, y) in enumerate(train_data):
                with tf.GradientTape() as tape:
                    logits = self.predict(x, training=True)

                    # Compute the loss value for this minibatch.
                    loss_value = self.loss_fn(y, logits)

                grads = tape.gradient(loss_value, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

                # Update training metric.
                train_acc_metric.update_state(y, logits)
                epoch_loss_avg.update_state(loss_value)

                if step % 100 == 0:
                    print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                    print("Seen so far: %s samples" % ((step + 1) * self.params.batch_size))

                    # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            # put vaules in lists
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(train_acc_metric.result())

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            epoch_loss_avg.reset_states()

            # Run a validation loop at the end of each epoch.
            for step, (x_batch_val, y_batch_val) in enumerate(eval_data):
                val_logits = self.predict(x_batch_val, training=False)
                # Update val metrics
                val_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()

            print("Validation acc: %.4f" % (float(val_acc),))

            print("Time taken: %.2fs" % (time.time() - start_time))
            self.save_model()

        self.visualize(train_loss_results, train_accuracy_results)
