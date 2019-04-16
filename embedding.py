__author__ = "Huseyin Cotel"
__copyright__ = "Copyright 2018"
__credits__ = ["huseyincot"]
__license__ = "GPL"
__version__ = "1.0.1"
__email__ = "info@vircongroup.com"
__status__ = "Development"

import numpy as np
import time
from gensim.models import KeyedVectors


class Embedding():
    def __init__(self, vocab):
        st = time.time()
        self.word_vectors = KeyedVectors.load_word2vec_format('utils/word2vec_model', binary=True)
        self.embedding_size = self.word_vectors.wv.vector_size
        print(f"Word2vec load in {time.time() - st} secs.")
        self.embedding_list = np.empty(shape=(0,self.embedding_size))
        self.form_embedding_list(vocab)

    def get_embedding_matrix_sentence(self, sentence):
        for token in sentence:
            self.get_embedding_vector_token(token)

    def get_embedding_vector_token(self, token):
        return self.word_vectors.wv[token]

    def form_embedding_list(self, vocab):
        for token in vocab:
            if token in self.word_vectors.vocab:
                self.embedding_list = np.vstack((self.embedding_list, self.get_embedding_vector_token(token)))
            else:
                self.embedding_list = np.vstack((self.embedding_list, np.ones(shape=(1,400))))
if __name__ == "__main__":
    e = Embedding()
    result = e.get_embedding_vector_token('kahraman')
    print(result)







