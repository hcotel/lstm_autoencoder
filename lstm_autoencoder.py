import tensorflow as tf
import numpy as np
import hyperparameter as hp
from tensorflow.contrib.rnn import BasicLSTMCell
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
        self.dense_layer = tf.layers.Dense(units=self.vocab_size)
        # Embedding
        self.embeddings = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size]))  # (vocab_size ,300)
        embedded_input = tf.nn.embedding_lookup(self.embeddings, self.input, name='embedding')  # (32,112,300)
        # embedded_input = tf.nn.dropout(embedded_input)
        print(embedded_input)
        embedded_sequence = tf.unstack(embedded_input, axis=1)  # (32,300),(32,300) 112 times ( #time_steps = sentence_length)
        print(embedded_sequence)
        encoder_outputs, encoder_state = tf.nn.static_rnn(self.encoder_cell, embedded_sequence, dtype=tf.float32, scope='encoder')      # (32,600)
        print(f"Encoded: {encoder_outputs}")
        print(f"Encoder_state: {encoder_state}")
        # Decoder
        start_decoder_index = special_tokens.index(START_DECODING)
        start_decoder = tf.ones(shape=self.batch_size, dtype=tf.int32) * start_decoder_index
        print(start_decoder)
        embedded_start_decoder = tf.nn.embedding_lookup(self.embeddings, start_decoder, name='start_decoding_embedding')    #(32,600)
        print(f"Start Decoding: {embedded_start_decoder}")
        #decoder_input = tf.concat([embedded_input, embedded_start_decoder], axis=1)
        #decoder_sequence = tf.unstack(decoder_input, axis=1)
        #print(decoder_input)
        # decoded, _ = tf.nn.static_rnn(self.decoder_cell, embedded_start_decoder, initial_state=encoder_state,
        #                               dtype=tf.float32, scope='decoder')
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
        print(f"logits: {self.logits}")
        print(f"y_one_hot: {self.output_one_hot}")

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.output_one_hot, logits=self.logits))    #(32,112, vocab_size)
        print(f"Loss: {self.loss}")
        self.train = tf.train.AdamOptimizer().minimize(self.loss)
        # grads = tf.gradients(self.loss, tf.trainable_variables())
        # grads, _ = tf.clip_by_global_norm(grads, 50)  # gradient clipping
        # grads_and_vars = list(zip(grads, tf.trainable_variables()))
        # self.train_op = optimizer.apply_gradients(grads_and_vars)
        # correct = tf.equal(self.decoder_output, pred)
        # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def run_graph(self):
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.create_placeholders()
        self.create_graph()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.filewriter_path, self.session.graph)
        self.merged = tf.summary.merge_all()
        self.session.run(init)

    def train_batch(self, batch):
        self.feed_dict[self.input] = batch
        self.feed_dict[self.output] = batch
        print(self.session)
        _, self.current_loss = self.session.run([self.train, self.loss], feed_dict=self.feed_dict)
        self.saver.save(self.session, self.save_model_path)
        # merged = tf.summary.merge_all()
        # summ = self.session.run(merged)
        # self.writer.add_summary(summ)

    def finalize_graph(self):
        self.writer.close()
