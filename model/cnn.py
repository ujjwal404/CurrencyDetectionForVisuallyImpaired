import tensorflow as tf


class CNN(tf.keras.Model):
    def __init__(self, num_classes, device="cpu:0", checkpoint_directory=None):

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

    def grads_fn(self, images, target, training):
        """Dynamically computes the gradients of the loss value
        with respect to the parameters of the model, in each
        forward pass.
        """
        with tf.GradientTape() as tape:
            loss = self.loss_fn(images, target, training)
        img = tf.Variable(self.variables)
        return tape.gradient(loss, img)

    def restore_model(self):
        """Function to restore trained model."""
        with tf.compat.v1.Session() as sess:
            with tf.device(self.device):
                # Run the model once to initialize variables
                dummy_input = tf.zeros((1, 48, 48, 1))
                dummy_pred = self.predict(dummy_input, training=False)
                # Restore the variables of the model
                saver = tf.compat.v1.train.Saver(self.variables)
                saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_directory))

    def save_model(self, global_step=0):
        with tf.compat.v1.Session() as sess:
            tf.compat.v1.train.Saver(self.variables).save(sess, self.checkpoint_directory, global_step=global_step)

    def compute_accuracy(self, input_data):
        """Compute the accuracy on the input data."""
        with tf.device(self.device):
            acc = tf.metrics.Accuracy()
            for images, targets in iter(input_data):
                # Predict the probability of each class
                logits = self.predict(images, training=False)
                # Select the class with the highest probability
                preds = tf.argmax(logits, axis=1)
                # Compute the accuracy
                acc(
                    tf.reshape(
                        targets,
                        [
                            -1,
                        ],
                    ),
                    preds,
                )
        return acc

    def fit_dataset(self, train_data, eval_data, epochs, batch_size):

        for epoch in range(epochs):
            # total 8659 images in train folder, 32 batches, 270 steps per epoch
            for step, (x, y) in enumerate(train_data):
                with tf.GradientTape() as tape:
                    logits = self.predict(x, training=True)
                    loss = tf.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
                    loss = tf.reduce_mean(loss)
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                if step % 100 == 0:
                    print(epoch, step, "loss:", float(loss))
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
            acc = total_correct / total
            print(epoch, "accuracy :", acc)
            # self.save_model(epoch)
