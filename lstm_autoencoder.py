import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell

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
        self.input = tf.placeholder(shape=[self.batch_size, self.padding_size], dtype=tf.int32, name='input')
        #self.output = tf.placeholder(shape=[self.batch_size, self.padding_size], dtype=tf.int32, name='output')

    def create_graph(self):
        e = encoder_lstm = LSTMCell(self.hidden_size, self.concat_size)
        d = decoder_lstm = LSTMCell(self.hidden_size, self.concat_size)
        self.embeddings = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size]))
        for slice in self.input:
            embedded_input = tf.nn.embedding_lookup(self.embeddings, slice, name='Embedding Layer')
            h_prev_encoder = tf.Variable(tf.zeros([self.padding_size, self.hidden_size]))
            z = np.hstack([embedded_input, h_prev_encoder])  # Z (10,384)

            ft = tf.sigmoid(tf.matmul(e.Wf, z) + e.bf)  # (256,1)
            it = tf.sigmoid(tf.matmul(e.Wi, z) + e.bi)  # (256,1)
            ct = tf.tanh(tf.matmul(e.Wc, z) + e.bc)  # (256,1)
            ot = tf.sigmoid(tf.matmul(e.Wo, z) + e.bi)  # (256,1)

            cell_state = tf.add(tf.multiply(prev_cell_st, ft), tf.multiply(it, ct))
            hidden_state = tf.multiply(ot, tf.tanh(cell_state))
            prev_cell_st = cell_state
            prev_hidden_st = hidden_state

    def run_graph(self, batch):
        self.feed_dict[self.input] = batch
        self.sess.run(self.output, feed_dict=self.feed_dict)

        return self.output









