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


class Batcher():
    def __init__(self):
        self.preprocess = Preprocess()
        #self.embedding = Embedding(self.preprocess.vocab)
        self.batch_size = hp.BATCH_SIZE
        self.input_path = hp.INPUT_PATH
        self.embedding_size = hp.SELF_EMBEDDING_SIZE
        self.graph = LSTMAutoEncoder(self.batch_size, self.preprocess.padding_size, self.embedding_size, self.preprocess.vocab_size)
        self.file_list = os.listdir('data/test')


    def batch(self):
        while True:
            if self.file_list:
                filename = random.choice(self.file_list)
                with io.open(f'data/test/{filename}', 'r', encoding='utf8') as textfile:
                    readcontent = textfile.read().replace(f'\n',f'.\n',1)
                    self.file_list.remove(filename)
                    sentences = self.preprocess.split_into_sentences(readcontent)
                    for sentence in sentences:
                        yield sentence
            else:
                exit(1)

    def train(self, batch):
        self.graph.run_graph(batch)

if __name__ == '__main__':
    print(f"Batch size: {hp.BATCH_SIZE}")
    print(f"Padding Algorithm: {hp.PADDING_ALGORITHM}")
    print(f"Vocabulary Freq Threshold: {hp.VOCAB_FREQ_THRESHOLD}")
    print(f"Use Self Vocabulary: {hp.SELF_VOCAB}")
    print(f"Remove Stopwords: {hp.REMOVE_STOPWORDS}")
    print(f"Use Self Embedding: {hp.SELF_EMBEDDING}")
    batcher = Batcher()
    gen = batcher.batch()
    encoder_2d = np.empty(shape=(0, batcher.preprocess.padding_size), dtype=np.int32)
    #decoder_matrix = np.empty(shape=(0, batcher.preprocess.padding_size + 2, batcher.embedding.embedding_size), dtype=np.int32)
    while True:
        s_list = list(next(gen) for _ in range(batcher.batch_size))
        for s in s_list:
            encoder_list = batcher.preprocess.convert_word_list_to_indexes(batcher.preprocess.add_tokens_to_sentence(batcher.preprocess.remove_punctuations(batcher.preprocess.remove_digits(s))))
            encoder_1d = np.array(encoder_list, dtype=np.int32).reshape(1, batcher.preprocess.padding_size)
            encoder_2d = np.concatenate((encoder_2d, encoder_1d))
            #decoder_array = np.array(batcher.preprocess.convert_word_list_to_indexes(batcher.preprocess.decoder_output_check_sentence(batcher.preprocess.remove_punctuations(batcher.preprocess.remove_digits(s)))), np.int32)
            #encoder_3d = np.concatenate((encoder_3d, encoder_2d))
            #decoder_matrix = np.vstack((decoder_matrix, decoder_array))
        batcher.train(encoder_2d)

