import numpy as np
import tensorflow as tf
from preprocessing import *
from pylab import show, plot, grid, title, xlabel, ylabel
import scipy.stats


class Model(tf.keras.Model):
    def __init__(self, token_vocab_size):
        super(Model, self).__init__()
        
        self.batch_size = 600
        self.window_size = 20
        self.windows_per_batch = int(self.batch_size / self.window_size)

        self.embedding_size = 64
        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.token_vocab_size = token_vocab_size
        self.hidden_layer_size = 2000
    
        self.E = tf.Variable(tf.random.truncated_normal([self.token_vocab_size, self.embedding_size], stddev=.1))
    
        self.layer1 = tf.keras.layers.GRU(self.embedding_size, return_sequences=True, return_state=True)
        self.layer2 = tf.keras.layers.GRU(self.embedding_size, return_sequences=True, return_state=True)

        self.dense_layer1 = tf.keras.layers.Dense(self.hidden_layer_size, activation='relu')
        self.dense_layer2 = tf.keras.layers.Dense(self.token_vocab_size, activation='softmax')

    def call(self, inputs, initial_state=None):
        """
        :param: inputs: input token ids
        :param: labels: label token ids
        return probs: probabilities as [batch_size x window_size x token_vocab_size] tensor
        """

        # embedding lookup
        embedding = tf.nn.embedding_lookup(self.E, inputs)
        embedding = tf.reshape(embedding, [self.windows_per_batch, self.window_size, self.embedding_size])

        # RNN layer
        whole_seq_out, final_state = self.layer1(embedding, initial_state=initial_state)
        final_seq, _ = self.layer2(whole_seq_out, initial_state=final_state)

        # fully connected linear layers with relu after first, softmax after second
        return self.dense_layer2(self.dense_layer1(final_seq))

    def loss_function(self, probs, labels):
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        return tf.math.reduce_mean(losses)


def visualize_loss(losses):
    """

    :param losses: list of losses from each batch
    :return:
    """
    x_values = np.arange(0, len(losses), 1)
    y_values = losses
    plot(x_values, y_values)
    xlabel('batch number')
    ylabel('categorical cross entropy loss')
    title('Loss per Batch')
    grid(True)
    show()


def train(model, train_inputs, train_labels):
    """
    main training loop for the models, returns a list representing
    the loss calculated for every batch
    """
    num_batches = int(len(train_inputs) / model.batch_size)
    losses = []

    for i in range(num_batches):
        print("TRAIN BATCH ", i+1, "/", num_batches)
        inputs, labels = get_batch(train_inputs, train_labels, i * model.batch_size, model.batch_size)

        # reshape inputs into windows
        inputs = tf.reshape(inputs, [model.windows_per_batch, model.window_size])

        with tf.GradientTape() as tape:
            probs = model.call(inputs)
            probs = tf.reshape(probs, [model.batch_size, model.token_vocab_size])
            loss = model.loss_function(probs, labels)
            losses.append(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return losses


def test(model, test_inputs, test_labels):
    num_batches = int(len(test_inputs) / model.batch_size)
    correct_predictions = 0

    for i in range(num_batches):
        inputs, labels = get_batch(test_inputs, test_labels, i * model.batch_size, model.batch_size)

        # reshape inputs into windows
        inputs = tf.reshape(inputs, [model.windows_per_batch, model.window_size])
        probs = model.call(inputs)

        # count how often the label with highest prob is correct
        probs = tf.reshape(probs, [model.batch_size, model.token_vocab_size])
        predicted = tf.argmax(probs, 1)
        correct_predictions += tf.reduce_sum(tf.cast(tf.equal(labels, predicted), tf.float32))

    return correct_predictions


def main():
    train_inputs, train_labels, test_inputs, test_labels, token_to_id = get_data()

    token_vocab_size = len(token_to_id)
    m = Model(token_vocab_size)

    losses = train(m, train_inputs, train_labels)
    visualize_loss(losses)

    num_correct = test(m, test_inputs, test_labels)
    print("Test Set Percent Accuracy: ", num_correct / len(test_inputs))

    # (low p-value indicates a small probability of choosing labels randomly and getting this many correct)
    p_value = 1 - scipy.stats.binom.cdf(num_correct-1, len(test_inputs), 1 / token_vocab_size)
    print("p-value: ", p_value)


if __name__ == '__main__':
    main()
