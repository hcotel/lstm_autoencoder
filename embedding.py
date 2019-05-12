import numpy as np
import time
from gensim.models import KeyedVectors

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary(OOV) words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'

class Embedding():
    def __init__(self, vocab):
        st = time.time()
        self.word_vectors = KeyedVectors.load_word2vec_format('utils/word2vec_model', binary=True, unicode_errors='ignore')
        self.embedding_size = self.word_vectors.wv.vector_size
        print(f"Word2vec load in {round((time.time() - st),2)} secs.")
        self.embedding_list = np.empty(shape=(0,self.embedding_size))
        self.form_embedding_list(vocab)

    def get_embedding_matrix_sentence(self, sentence):
        embedding_matrix = np.empty(shape=(0,self.embedding_size))
        for token in sentence:
            a = self.embedding_list[token]
            embedding_matrix = np.vstack((embedding_matrix, self.embedding_list[token]))
        return embedding_matrix

    def get_embedding_vector_token(self, token):
        return self.word_vectors.wv[token]

    def form_embedding_list(self, vocab):
        for token in vocab:
            special_token_list = [UNKNOWN_TOKEN, PAD_TOKEN, SENTENCE_END, START_DECODING, STOP_DECODING]
            if token in special_token_list:
                a = special_token_list.index(token)
                self.embedding_list = np.vstack((self.embedding_list, a * np.ones(shape=(1,self.embedding_size))))
            elif token in self.word_vectors.vocab:
                self.embedding_list = np.vstack((self.embedding_list, self.get_embedding_vector_token(token)))
            else:
                self.embedding_list = np.vstack((self.embedding_list, np.ones(shape=(1,self.embedding_size))))

if __name__ == "__main__":
    e = Embedding()
    result = e.get_embedding_vector_token('kahraman')
    print(result)







