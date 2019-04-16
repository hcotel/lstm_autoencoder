__author__ = "Huseyin Cotel"
__copyright__ = "Copyright 2018"
__credits__ = ["huseyincot"]
__license__ = "GPL"
__version__ = "1.0.1"
__email__ = "info@vircongroup.com"
__status__ = "Development"

import os
import random

import io
import numpy as np
from datetime import datetime

import hyperparameter as hp
from embedding import Embedding
from lstm_autoencoder import Graph
from preprocess import Preprocess


class Batcher():
    def __init__(self):
        self.preprocess = Preprocess()
        self.embedding = Embedding(self.preprocess.vocab)
        self.batch_size = hp.BATCH_SIZE
        self.input_path = hp.INPUT_PATH
        self.graph = Graph(self.batch_size, self.preprocess.padding_size, self.embedding.embedding_size)
        self.file_list = os.listdir('data/input')
        st = datetime.now()
        i=1
        # for root, dirs, files in os.walk('data/42bin_haber/news'):
        #     for file in files:
        #         if file.endswith(".txt"):
        #             shutil.copy(os.path.join(root, file),f'data/input/{i}.txt')
        #             i+=1
        elapsed_time = datetime.now() - st
        print(f"Files copied in {elapsed_time} secs.")

    def batch(self):
        while True:
            filename = random.choice(self.file_list)
            with io.open(f'data/input/{filename}', 'r', encoding='utf8') as textfile:
                readcontent = textfile.read().replace(f'\n',f'.\n',1)
                self.file_list.remove(filename)
                sentences = self.preprocess.split_into_sentences(readcontent)
                for sentence in sentences:
                    yield sentence

if __name__ == '__main__':
    batcher = Batcher()
    gen = batcher.batch()
    encoder_3d = np.empty(shape=(0, batcher.preprocess.padding_size, batcher.embedding.embedding_size), dtype=np.int32)
    decoder_matrix = np.empty(shape=(0, batcher.preprocess.padding_size + 2, batcher.embedding.embedding_size), dtype=np.int32)
    while True:
        s_list = list(next(gen) for _ in range(batcher.batch_size))
        for s in s_list:
            encoder_2d = batcher.embedding.get_embedding_matrix_sentence(batcher.preprocess.convert_word_list_to_indexes(batcher.preprocess.add_tokens_to_sentence(batcher.preprocess.remove_punctuations(batcher.preprocess.remove_digits(s)))))
            encoder_2d = encoder_2d.reshape(1,batcher.preprocess.padding_size, batcher.embedding.embedding_size)

            #decoder_array = np.array(batcher.preprocess.convert_word_list_to_indexes(batcher.preprocess.decoder_output_check_sentence(batcher.preprocess.remove_punctuations(batcher.preprocess.remove_digits(s)))), np.int32)
            encoder_3d = np.concatenate((encoder_3d, encoder_2d))
            #decoder_matrix = np.vstack((decoder_matrix, decoder_array))
        print(encoder_3d)

