import tensorflow as tf

optimizer = tf.keras.optimizers.RMSprop()
loss_object = tf.keras.losses.CategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

EPOCHS = 4
LOG_DIR = 'kaggle_logs'
summary_writer = tf.summary.create_file_writer(LOG_DIR)


def train_animals(model, train_data, test_data):
    for epoch in range(EPOCHS):
        for img, lbl in train_data:
            train_step(model, img, lbl)
        with summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        # use test_data here to test for model acc
        temp = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(temp.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100))
        train_loss.reset_states()
        train_accuracy.reset_states()


@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as grad_tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    grads = grad_tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
