import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell

#tf.nn.embedding_lookup
#tf.gradients
#tf.trainable_variables
#optimizer apply gradient
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

    def run_graph(self, batch):
        self.feed_dict[self.input] = batch
        self.sess.run(self.output, feed_dict=self.feed_dict)
        return self.output





