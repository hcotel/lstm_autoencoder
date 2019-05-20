import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import LSTMCell
#tf.nn.embedding_lookup
#tf.gradients
#tf.trainable_variables
#optimizer apply gradient

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class LSTMCell():
    def __init__(self, hidden_size, concat_size):
        self.hidden_size = hidden_size
        self.concat_size = concat_size
        self.initialize_lstm_variables()

    def initialize_lstm_variables(self):
        self.Wf = tf.Variable(tf.random_normal([self.hidden_size, self.concat_size]), name="Wf")
        self.bf = tf.Variable(tf.zeros([self.hidden_size, 1]), name="bf")
        self.Wi = tf.Variable(tf.random_normal([self.hidden_size, self.concat_size]), name="Wi")
        self.bi = tf.Variable(tf.zeros([self.hidden_size, 1]), name="bi")
        self.Wc = tf.Variable(tf.random_normal([self.hidden_size, self.concat_size]), name="Wc")
        self.bc = tf.Variable(tf.zeros([self.hidden_size, 1]), name="bc")
        self.Wo = tf.Variable(tf.random_normal([self.hidden_size, self.concat_size]), name="Wo")
        self.bo = tf.Variable(tf.zeros([self.hidden_size, 1]), name="bo")
        self.prev_h = tf.Variable(tf.zeros([self.hidden_size, 1]), dtype=tf.float32, name="prev_h", trainable=False)
        self.prev_c = tf.Variable(tf.zeros([self.hidden_size, 1]), dtype=tf.float32, name="prev_c", trainable=False)

class LSTMAutoEncoder():
    def __init__(self, batch_size, padding_size, embedding_size, vocab_size):
        self.batch_size = batch_size
        self.padding_size = padding_size
        self.embedding_size = embedding_size
        self.hidden_size = 2 * self.embedding_size
        self.concat_size = 3 * self.embedding_size
        self.vocab_size = vocab_size
        self.feed_dict = dict()
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.create_placeholders()
        self.create_graph()
        self.sess.run(tf.global_variables_initializer())

    def create_placeholders(self):
        self.input = tf.placeholder(shape=[self.batch_size, self.padding_size], dtype=tf.int32, name='input')    #(32,10)
        #self.output = tf.placeholder(shape=[self.batch_size, self.padding_size], dtype=tf.int32, name='output')

    def create_graph(self):
        initializer = tf.glorot_normal_initializer()
        self.encoder_cell = LSTMCell(self.hidden_size, initializer=initializer)                       #hidden_size=256
        self.decoder_cell = LSTMCell(self.hidden_size, initializer=initializer)
        self.embeddings = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size]))     #(300,128)
        embedded_input = tf.nn.embedding_lookup(self.embeddings, tf.unstack, name='Embedding Layer')     #(1,10,128)
        embedded_input = tf.reshape(embedded_input, [-1, 1, self.embedding_size])
        encoded, _ = tf.contrib.rnn.static_rnn(self.encoder_cell, tf.unstack(embedded_input), dtype=tf.float32)


    def run_graph(self, batch):
        self.feed_dict[self.input] = batch
        self.sess.run(self.output, feed_dict=self.feed_dict)

        return self.output









