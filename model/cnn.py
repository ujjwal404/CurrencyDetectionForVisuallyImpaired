import tensorflow as tf


class CNN(tf.keras.Model):
    def __init__(self, num_classes, device="cpu:0", checkpoint_directory=None, params=None):

        super(CNN, self).__init__()

        self.cnn1 = tf.keras.layers.Conv2D(16, (5, 5), padding="same", strides=(2, 2), kernel_initializer="he_normal")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.cnn2 = tf.keras.layers.Conv2D(32, (5, 5), padding="same", strides=(2, 2), kernel_initializer="he_normal")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes)

        self.device = device
        self.checkpoint_directory = checkpoint_directory

    def predict(self, inputs, training):
        x = self.cnn1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)  # layer 1
        x = tf.nn.relu(self.bn2(self.cnn2(x)))  # layer 2
        x = self.pool(x)
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

    def fit_dataset(self, train_data, eval_data, epochs, batch_size):

        for epoch in range(epochs):
            # total 8659 images in train folder, 32 batches, 270 steps per epoch
            for step, (x, y) in enumerate(train_data):
                grads, loss = self.grads_fn(x, y, training=True)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                if step % 100 == 0:
                    print(epoch, step, "loss:", float(loss))

            acc = self.compute_accuracy(eval_data)
            print(epoch, "accuracy :", acc)
            # self.save_model(epoch)
