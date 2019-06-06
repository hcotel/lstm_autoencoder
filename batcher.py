import os
import random

import io
import numpy as np
import tensorflow as tf
from datetime import datetime

import hyperparameter as hp
from embedding import Embedding
from lstm_autoencoder import LSTMAutoEncoder
from preprocess import Preprocess
import time


class Batcher():
    def __init__(self):
        self.preprocess = Preprocess()
        #self.embedding = Embedding(self.preprocess.vocab)
        self.batch_size = hp.BATCH_SIZE
        self.input_path = hp.INPUT_PATH
        self.embedding_size = hp.SELF_EMBEDDING_SIZE
        self.ae = LSTMAutoEncoder(self.batch_size, self.preprocess.padding_size, self.embedding_size, self.preprocess.vocab_size)
        self.file_list = os.listdir(self.input_path)


    def batch(self):
        self.epoch = 1
        epoch_start_time = time.time()
        while True:
            if self.epoch <= hp.NUM_EPOCHS:
                if self.file_list:
                    filename = random.choice(self.file_list)
                    with io.open(f'data/test/{filename}', 'r', encoding='utf8', errors='ignore') as textfile:
                        readcontent = textfile.read().replace(f'\n',f'.\n',1)
                        self.file_list.remove(filename)
                        sentences = self.preprocess.split_into_sentences(readcontent)
                        for sentence in sentences:
                            yield sentence
                else:
                    epoch_end_time = time.time()
                    epoch_time = round(epoch_end_time - epoch_start_time)
                    print(f"File list is all read for Epoch:{self.epoch} in {epoch_time} secs.")
                    self.epoch += 1
                    epoch_start_time = time.time()
                    self.file_list = os.listdir(self.input_path)
            else:
                self.ae.finalize_graph(sess)
                exit(1)


if __name__ == '__main__':
    print(f"Batch size: {hp.BATCH_SIZE}")
    print(f"Padding Algorithm: {hp.PADDING_ALGORITHM}")
    print(f"Vocabulary Freq Threshold: {hp.VOCAB_FREQ_THRESHOLD}")
    print(f"Use Self Vocabulary: {hp.SELF_VOCAB}")
    print(f"Remove Stopwords: {hp.REMOVE_STOPWORDS}")
    print(f"Use Self Embedding: {hp.SELF_EMBEDDING}")
    batcher = Batcher()
    gen = batcher.batch()
    with tf.Session() as sess:
        batcher.ae.writer = tf.summary.FileWriter(hp.FILEWRITER_PATH, sess.graph, flush_secs=10)
        init = tf.global_variables_initializer()
        sess.run(init)
        while True:
            s_list = list(next(gen) for _ in range(batcher.batch_size))
            encoder_2d = np.empty(shape=(0, batcher.preprocess.padding_size), dtype=np.int32)
            for s in s_list:
                encoder_list = batcher.preprocess.convert_word_list_to_indexes(batcher.preprocess.add_tokens_to_sentence(batcher.preprocess.remove_punctuations(batcher.preprocess.remove_digits(s))))
                encoder_1d = np.array(encoder_list, dtype=np.int32).reshape(1, batcher.preprocess.padding_size)
                encoder_2d = np.concatenate((encoder_2d, encoder_1d))
            batcher.ae.train_batch(sess, encoder_2d)
            print(f"Epoch: {batcher.epoch} Loss: {batcher.ae.current_loss}")
            batcher.ae.writer.add_summary(batcher.ae.summary)



