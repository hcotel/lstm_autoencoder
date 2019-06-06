import tensorflow as tf
import numpy as np
import hyperparameter as hp
from tensorflow.contrib.rnn import LSTMCell

# tf.nn.embedding_lookup
# tf.gradients
# tf.trainable_variables
# optimizer apply gradient

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary(OOV) words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences
special_tokens = [UNKNOWN_TOKEN, PAD_TOKEN, SENTENCE_END, START_DECODING, STOP_DECODING]

class LSTMAutoEncoder():
    def __init__(self, batch_size, padding_size, embedding_size, vocab_size):
        self.batch_size = batch_size
        self.padding_size = padding_size
        self.embedding_size = embedding_size
        self.hidden_size = 2 * self.embedding_size
        self.vocab_size = vocab_size
        self.feed_dict = dict()
        self.filewriter_path = hp.FILEWRITER_PATH
        self.save_model_path = hp.SAVE_PATH
        self.learning_rate = hp.LEARNING_RATE
        tf.reset_default_graph()
        self.run_graph()

    def create_placeholders(self):
        self.input = tf.placeholder(shape=[self.batch_size, self.padding_size], dtype=tf.int32, name='input')  # (32,112)
        self.output = tf.placeholder(shape=[self.batch_size, self.padding_size], dtype=tf.int32, name='output')  # (32,112)

    def create_graph(self):
        initializer = tf.glorot_normal_initializer()
        self.encoder_cell = LSTMCell(self.embedding_size, initializer=initializer, state_is_tuple=True)  # hidden_size=300
        self.decoder_cell = LSTMCell(self.embedding_size, initializer=initializer, state_is_tuple=True)
        self.dense_layer = tf.layers.Dense(units=self.vocab_size, bias_initializer=initializer, kernel_initializer=initializer)
        # Embedding
        self.embeddings = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size]))  # (vocab_size ,300)
        embedded_input = tf.nn.embedding_lookup(self.embeddings, self.input, name='embedding')  # (32,112,300)
        # embedded_input = tf.nn.dropout(embedded_input)
        embedded_sequence = tf.unstack(embedded_input, axis=1)  # (32,300),(32,300) 112 times ( #time_steps = sentence_length)
        encoder_outputs, encoder_state = tf.nn.static_rnn(self.encoder_cell, embedded_sequence, dtype=tf.float32, scope='encoder')      # (32,600)
        # Decoder
        start_decoder_index = special_tokens.index(START_DECODING)
        start_decoder = tf.ones(shape=self.batch_size, dtype=tf.int32) * start_decoder_index
        embedded_start_decoder = tf.nn.embedding_lookup(self.embeddings, start_decoder, name='start_decoding_embedding')    #(32,600)
        decoder_input = embedded_start_decoder  # (32,300)
        state = encoder_state  # (32,600)
        self.pred_outputs = []
        for i in range(self.padding_size):
            decoder_output, next_state = self.decoder_cell(decoder_input, state)
            pred_output = self.dense_layer(decoder_output)
            self.pred_outputs.append(pred_output)
            soft_output = tf.nn.softmax(pred_output)
            pred_word = tf.argmax(soft_output, 1)
            decoder_input = tf.nn.embedding_lookup(self.embeddings, pred_word, name='decoder_input')
            state = next_state

        self.logits = tf.transpose(tf.stack(self.pred_outputs), [1, 0, 2])
        self.output_one_hot = tf.one_hot(self.input, self.vocab_size)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output_one_hot, logits=self.logits))    #(32,112, vocab_size)
        tf.summary.scalar('loss', self.loss)
        train_vars = tf.trainable_variables()
        gradients = tf.gradients(self.loss, train_vars)
        grads, global_norm = tf.clip_by_global_norm(gradients, clip_norm=1)
        tf.summary.scalar('global_norm', global_norm)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = optimizer.apply_gradients(zip(grads, train_vars))
        self.merged = tf.summary.merge_all()

    def run_graph(self):
        self.create_placeholders()
        self.create_graph()
        self.saver = tf.train.Saver()

    def train_batch(self, sess, batch):
        self.feed_dict[self.input] = batch
        self.feed_dict[self.output] = batch
        self.summary, _, self.current_loss = sess.run([self.merged, self.train, self.loss], feed_dict=self.feed_dict)

    def finalize_graph(self, sess):
        self.saver.save(sess, self.save_model_path)
        self.writer.close()
