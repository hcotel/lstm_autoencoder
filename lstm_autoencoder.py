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

class LSTMAutoEncoder():
    def __init__(self, batch_size, padding_size, embedding_size, vocab_size):
        self.batch_size = batch_size
        self.padding_size = padding_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.feed_dict = dict()
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.create_placeholders()
        self.create_graph()
        self.sess.run(tf.global_variables_initializer())

    def create_placeholders(self):
        self.input = tf.placeholder(shape=[self.batch_size, self.padding_size], dtype=tf.int32, name='inputs')
        self.embeddings = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size]))

    def create_graph(self):
        embed_output = tf.nn.embedding_lookup(self.embeddings, self.input, name='Embedding Layer')
        lstm = BasicLSTMCell(1)
        state = lstm.zero_state(self.batch_size, dtype=tf.float32)

    def run_graph(self, batch):
        self.feed_dict[self.input] = batch
        self.sess.run(self.output, feed_dict=self.feed_dict)

        return self.output

    # def LSTM_Cell(self, input, h_prev, c_prev, parameters):
    #     # input (128,1)  h_prev (256,1)
    #     input_size, batch_size = input.shape
    #     h_size, _ = h_prev.shape
    #
    #     #Retrieve parameters
    #     Wf = parameters["Wf"]
    #     bf = parameters["bf"]
    #     Wi = parameters["Wi"]
    #     bi = parameters["bi"]
    #     Wc = parameters["Wc"]
    #     bc = parameters["bc"]
    #     Wo = parameters["Wo"]
    #     bo = parameters["bo"]
    #     Wy = parameters["Wy"]
    #     by = parameters["by"]
    #
    #     concat = np.zeros((input_size + h_size, batch_size))
    #     concat[: input_size, :] = input
    #     concat[input_size:, :] = h_prev
    #
    #     ft = sigmoid(np.dot(Wf, concat) + bf)
    #     it = sigmoid(np.dot(Wi, concat) + bi)
    #     cct = np.tanh(np.dot(Wc, concat) + bc)
    #     c_next = ft * c_prev + it * cct
    #     ot = sigmoid(np.dot(Wo, concat) + bo)
    #     h_next = ot * np.tanh(c_next)
    #
    #     yt_pred = softmax(np.dot(Wy, h_next) + by)







