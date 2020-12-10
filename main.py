import sys
import tensorflow as tf
from preprocessing import *
from pylab import show, plot, grid, title, xlabel, ylabel
from random import seed
from random import randint
import scipy.stats
from preprocess_test import piano_roll_to_midi


class Model(tf.keras.Model):
    def __init__(self, token_vocab_size):
        super(Model, self).__init__()
        
        self.batch_size = 300
        self.window_size = 20
        self.windows_per_batch = int(self.batch_size / self.window_size)

        self.embedding_size = 64
        self.learning_rate = 1e-3
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
        # print("TRAIN BATCH ", i+1, "/", num_batches)
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
    if len(sys.argv) != 2 or sys.argv[1] not in {"TRAIN", "COMPOSE"}:
        print("Usage: python main.py <TRAIN/COMPOSE>")
        exit()
    
    # get all data no matter what, in order to build a full vocabulary
    train_inputs, train_labels, test_inputs, test_labels, token_to_id, id_to_token, piece_starters = get_data()
    token_vocab_size = len(token_to_id)
    
    if sys.argv[1] == "TRAIN":
        m = Model(token_vocab_size)
        epochs = 8
        losses = []
        for i in range(epochs):
            print("Train epoch ", i + 1, " out of ", epochs)
            epoch_losses = train(m, train_inputs, train_labels)
            losses += epoch_losses
        
        # save model weights
        m.save_weights('weights')

        visualize_loss(losses)

        num_correct = test(m, test_inputs, test_labels)
        print("Test Set Percent Accuracy: ", num_correct / len(test_inputs))

        # (low p-value indicates a small probability of choosing labels randomly and getting this many correct)
        p_value = 1 - scipy.stats.binom.cdf(num_correct-1, len(test_inputs), 1 / token_vocab_size)
        print("p-value: ", p_value)
    else:
        m = Model(len(token_to_id))
        m.load_weights('weights').expect_partial()
        num_repetitions = 200

        # Choose a random piece starter to compose the rest of the song from
        seed()
        inputs = tf.convert_to_tensor(piece_starters[randint(0, len(piece_starters))])
        #print("INPUTS: ", inputs)
        initial_state = None
        out_sequence = inputs
        
        for i in range(num_repetitions):
            embedding = tf.nn.embedding_lookup(m.E, out_sequence)
            whole_seq_out, final_state = m.layer1(tf.expand_dims(embedding, axis=1), initial_state=initial_state)
            final_seq, final_state = m.layer2(whole_seq_out, initial_state=final_state)
            initial_state = final_state
            probs = m.dense_layer2(m.dense_layer1(initial_state))
            
            # sample the probability distribution to generate new notes
            out_sequence = []
            for i in range(64):
                out = np.random.choice(len(token_to_id), 1, True, np.array(probs[i]))[0]
                out_sequence.append(out)
                    
            out_sequence =tf.convert_to_tensor(out_sequence)
        
        #print("OUT SEQ: ", out_sequence, " LEN: ", out_sequence.shape)
        pr = list(map(lambda x: id_to_token[x.numpy()], out_sequence))
        midi_file = piano_roll_to_midi(pr, 60)
        midi_file.ticks_per_beat = 120
        midi_file.save("output_test.mid")

        print("Finished!")
        return


if __name__ == '__main__':
    main()

"""    
We generate music by feeding a short seed sequence into our trained model. We generate new tokens
from the output distribution from our softmax and feed the new tokens back into our model. We used
a combination of two different sampling schemes: one which chooses the token with maximum
predicted probability and one which chooses a token from the entire softmax distribution

Idea: have a few different sample starter tracks, choose one randomly whenever this is run,
generate new tokens and get midi output
"""


