import tensorflow as tf
import os


class CNN(tf.keras.Model):
    def __init__(self, num_classes, device="cpu:0", checkpoint_directory=None, params=None):

        super(CNN, self).__init__()

        self.cnn1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")
        self.cnn2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")

        self.mxpool1 = tf.keras.layers.MaxPool2D((2, 2), (2, 2), padding="same")

        self.cnn3 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")
        self.cnn4 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")

        self.mxpool2 = tf.keras.layers.MaxPool2D((2, 2), (2, 2), padding="same")

        self.cnn5 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")
        self.cnn6 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")
        self.cnn7 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")

        self.mxpool3 = tf.keras.layers.MaxPool2D((2, 2), (2, 2), padding="same")

        self.cnn8 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")
        self.cnn9 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")
        self.cnn10 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")

        self.mxpool4 = tf.keras.layers.MaxPool2D((2, 2), (2, 2), padding="same")

        self.cnn11 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")
        self.cnn12 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")
        self.cnn13 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")

        self.mxpool5 = tf.keras.layers.MaxPool2D((2, 2), (2, 2), padding="same")

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(4096, activation="relu")
        self.dense2 = tf.keras.layers.Dense(4096, activation="relu")
        self.classifier = tf.keras.layers.Dense(num_classes)

        self.device = device
        self.checkpoint = tf.train.Checkpoint(model=self)
        self.checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        self.params = params

    def predict(self, inputs, training):
        # vgg16 implementation
        x = self.cnn1(inputs)
        x = self.cnn2(x)
        x = self.mxpool1(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.mxpool2(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = self.cnn7(x)
        x = self.mxpool3(x)
        x = self.cnn8(x)
        x = self.cnn9(x)
        x = self.cnn10(x)
        x = self.mxpool4(x)
        x = self.cnn11(x)
        x = self.cnn12(x)
        x = self.cnn13(x)
        x = self.mxpool5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
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
        self.checkpoint.save(file_prefix=self.checkpoint_prefix, global_step=global_step)

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
