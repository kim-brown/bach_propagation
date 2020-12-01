import numpy as np
import tensorflow as tf
from preprocessing import *

class Model(tf.keras.Model):
    def __init__(self, token_vocab_size):
        super(Model, self).__init__()
        
        self.batch_size = 600
        self.embedding_size = 64
        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.token_vocab_size = token_vocab_size
    
        self.E = tf.Variable(tf.random.truncated_normal([self.token_vocab_size, self.embedding_size], stddev=.1))
    
        self.layer1 = tf.keras.layers.LSTM(self.embedding_size, return_sequences=True, return_state=True)
        self.layer2 = tf.keras.layers.LSTM(self.embedding_size, return_sequences=True, return_state=True)
    
        self.dense_layer = tf.keras.layers.Dense(self.token_vocab_size, activation='softmax')

    def call(self, inputs, labels):
        """
        :param: inputs: input token ids
        :param: labels: label token ids
        return probs: probabilities as [batch_size x window_size x token_vocab_size] tensor
        """
        # TODO: incorporate window size (in pre-processing)
        
        input_embeddings = tf.nn.embedding_lookup(self.E, inputs)
        label_embeddings = tf.nn.embedding_lookup(self.E, labels)
        #_, final_mem_state, final_carry_state = self.layer1(input_embeddings)
        #decoder_output, _, _ = self.layer2(label_embeddings, initial_state=(final_mem_state, final_carry_state))
        #return self.dense_layer(decoder_output) # layer contains softmax
        return None

    def loss_function(self, probs, labels):
        return tf.convert_to_tensor(0.0)

def visualize_results():
    pass

def train(model, train_inputs, train_labels):
    num_batches = int(len(train_inputs) / model.batch_size)
    for i in range(num_batches):
        print("TRAIN BATCH ", i+1, "/", num_batches)
        inputs, labels = get_batch(train_inputs, train_labels, i * model.batch_size, model.batch_size)
        with tf.GradientTape() as tape:
            probs = model.call(inputs, labels)
            loss = model.loss_function(probs, labels)
        #gradients = tape.gradient(loss, model.trainable_variables)
        #model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return

def test(model, test_inputs, test_labels):
    pass

def main():
    train_inputs, train_labels, test_inputs, test_labels, token_to_id = get_data()
    token_vocab_size = len(token_to_id)
    m = Model(token_vocab_size)
    m.__init__(token_vocab_size)
    train(m, train_inputs, train_labels)
    test(m, test_inputs, test_labels)

if __name__ == '__main__':
    main()
