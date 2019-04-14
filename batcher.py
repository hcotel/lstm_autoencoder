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
from preprocess import Preprocess


class Batcher():
    def __init__(self):
        self.pp = Preprocess()
        self.batch_size = hp.BATCH_SIZE
        self.input_path = hp.INPUT_PATH
        st = datetime.now()
        i=1
        # for root, dirs, files in os.walk('data/42bin_haber/news'):
        #     for file in files:
        #         if file.endswith(".txt"):
        #             shutil.copy(os.path.join(root, file),f'data/input/{i}.txt')
        #             i+=1
        elapsed_time = datetime.now() - st
        print(f"Files copied in {elapsed_time} secs.")


    def file_generator(self):
        yield random.choice(os.listdir("data/input"))

    def batch(self):
        for filename in next(self.file_generator):
            with io.open(f'data/input/{filename}', 'r', encoding='utf8') as textfile:  # !TODO Proper sentence read
                sentences = self.pp.split_into_sentences(textfile.read())
                for sentence in sentences:
                    yield sentence

if __name__ == '__main__':
    b = Batcher()
    gen = b.batch()
    encoder_matrix = np.empty(shape=(0,b.pp.padding_size),dtype=np.int32)
    decoder_matrix = np.empty(shape=(0, b.pp.padding_size+2), dtype=np.int32)
    while True:
        s_list = list(next(gen) for _ in range(b.batch_size))
        for s in s_list:
            encoder_array = np.array(b.pp.convert_word_list_to_indexes(b.pp.add_tokens_to_sentence(b.pp.remove_punctuations(b.pp.remove_digits(s)))), np.int32)
            decoder_array = np.array(b.pp.convert_word_list_to_indexes(b.pp.decoder_output_check_sentence(b.pp.remove_punctuations(b.pp.remove_digits(s)))), np.int32)
            encoder_matrix = np.vstack((encoder_matrix, encoder_array))
            decoder_matrix = np.vstack((decoder_matrix, decoder_array))
        print("Encoder Matrix:")
        print(encoder_matrix)
        print("Decoder Matrix:")
        print(decoder_matrix)
