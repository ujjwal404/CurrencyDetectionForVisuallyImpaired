import tensorflow as tf
import os


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
        x = self.cnn3(x)
        x = self.bn3(x, training=training)
        x = self.cnn4(x)
        x = self.bn4(x, training=training)
        x = self.cnn5(x)
        x = self.bn5(x, training=training)
        x = self.maxpool3(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        output = self.classifier(x)

        return output

    def loss_fn(self, images, target, training):
        preds = self.predict(images, training)
        print(images.shape, preds.shape)
        calc = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = calc(target, preds)
        return loss

    def grads_fn(self, x, y, training):  # runs on each batch
        with tf.GradientTape() as tape:
            logits = self.predict(x, training=True)
            loss = tf.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.trainable_variables)
        return grads, loss

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

    def fit_dataset(self, train_data, eval_data):

        for epoch in range(self.params.num_epochs):
            # total 8659 images in train folder, 32 batches, 270 steps per epoch
            for step, (x, y) in enumerate(train_data):
                grads, loss = self.grads_fn(x, y, training=True)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                if step % 100 == 0:
                    print(epoch, step, "loss:", float(loss))

            acc = self.compute_accuracy(eval_data)
            print(epoch, "accuracy :", acc)
            self.save_model()
