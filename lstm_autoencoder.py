__author__ = "Huseyin Cotel"
__copyright__ = "Copyright 2018"
__credits__ = ["huseyincot"]
__license__ = "GPL"
__version__ = "1.0.1"
__email__ = "info@vircongroup.com"
__status__ = "Development"

import tensorflow as tf



#tf.nn.embedding_lookup
#tf.gradients
#tf.trainable_variables
#optimizer apply gradient
class Graph():
    def __init__(self, batch_size, padding_size, embedding_size):
        self.batch_size = batch_size
        self.padding_size = padding_size
        self.embedding_size = embedding_size
    def create_placeholders(self):
        inputs = tf.placeholder(shape=[self.batch_size, self.padding_size, self.embedding_size], dtype=tf.float32,
                                name='inputs')
    def create_graph(self):
        pass
    def run_graph(self):
        pass


