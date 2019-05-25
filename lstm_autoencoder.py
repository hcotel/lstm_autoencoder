import tensorflow as tf
import numpy as np
import hyperparameter as hp
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import LSTMCell
#tf.nn.embedding_lookup
#tf.gradients
#tf.trainable_variables
#optimizer apply gradient

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary(OOV) words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences
special_tokens = [UNKNOWN_TOKEN, PAD_TOKEN, SENTENCE_END, START_DECODING, STOP_DECODING]

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
        self.hidden_size = 2 * self.embedding_size
        self.concat_size = 3 * self.embedding_size
        self.vocab_size = vocab_size
        self.feed_dict = dict()
        self.filewriter_path = hp.FILEWRITER_PATH
        self.save_model_path = hp.SAVE_PATH
        tf.reset_default_graph()
        self.run_graph()

    def create_placeholders(self):
        self.input = tf.placeholder(shape=[self.batch_size, self.padding_size], dtype=tf.int32, name='input')    #(32,10)
        #self.output = tf.placeholder(shape=[self.batch_size, self.padding_size], dtype=tf.int32, name='output')

    def create_graph(self):
        initializer = tf.glorot_normal_initializer()
        self.encoder_cell = LSTMCell(self.hidden_size, initializer=initializer)                       #hidden_size=256
        self.decoder_cell = LSTMCell(self.hidden_size, initializer=initializer)
        #Embedding
        self.embeddings = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size]))     #(300,128)
        embedded_input = tf.nn.embedding_lookup(self.embeddings, self.input, name='embedding')     #(32,10,128)
        #embedded_input = tf.nn.dropout(embedded_input)
        #Encoder
        embedded_input = tf.reshape(embedded_input, [-1, 1, self.embedding_size])                      #(32x10,128)
        encoded, _ = tf.nn.dynamic_rnn(self.encoder_cell, embedded_input, dtype=tf.float32, scope='encoder')          #(256,1)
        #Decoder
        start_decoder_index = special_tokens.index(START_DECODING)
        embedded_start_decoder = tf.nn.embedding_lookup(self.embeddings, start_decoder_index, name='start_decoding_embedding')
        decoder_input = tf.concat([encoded, embedded_start_decoder], axis=1)
        decoded, _ = tf.nn.dynamic_rnn(self.decoder_cell, decoder_input, dtype=tf.float32, scope='decoder')

    def run_graph(self):
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.create_placeholders()
        self.create_graph()
        self.session.run(init)
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.filewriter_path, self.session.graph)
        self.merged = tf.summary.merge_all()


    def train_batch(self, batch):
        self.feed_dict[self.input] = batch
        self.session.run(self.output, feed_dict=self.feed_dict)
        self.saver.save(self.session, self.save_model_path)
        merged = tf.summary.merge_all()
        summ = self.session.run(merged)
        self.writer.add_summary(summ)

    def finalize_graph(self):
        self.writer.close()










