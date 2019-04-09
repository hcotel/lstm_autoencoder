import os
import operator
import numpy as np
import re

from sklearn.utils import shuffle

tr_dict = {}
# take the N most used word dict
dict_size = 3000

PKEYS = {"NULL": 0,
         "PAD_TOKEN": 1,
         "START_SEQUENCE": 2,
         "STOP_SEQUENCE": 3,
         "UNKNOWN_TOKEN": 4}

# Create word2idx and idx2word indexes

DATA_DIR = "./data"

for filename in os.listdir(DATA_DIR):
    if filename.endswith("_freq.txt"):
        with open(os.path.join(DATA_DIR, filename), "rb") as datafile:
            for line in datafile:
                key, val = line.decode(encoding='utf-8').split()
                tr_dict[key] = int(val)

# sort dictionary by values
tr_dict_sorted = [*sorted(PKEYS.items(), key=operator.itemgetter(1), reverse=False),
                  *sorted(tr_dict.items(), key=operator.itemgetter(1), reverse=True)]

# take the N most used word dict
dict_size = dict_size if len(tr_dict) > dict_size else len(tr_dict)
tr_dict_subs = tr_dict_sorted[0:dict_size]

# tr_dict_merged = {**PKEYS, **tr_dict_subs}

key, val = zip(*tr_dict_subs)
# Creating a mapping from unique words to indices
word2idx = {u: i for i, u in enumerate(key)}
idx2word = np.array(key)


class Preprocessing:
    def __init__(self):
        pass

    @staticmethod
    def sentence_desegmenter(_text, remove_punc=False):
        regex_getter = u"\<s\>(.+?)\</s\>+?"
        if remove_punc:
            _texts = re.findall(regex_getter, _text)
            return list(map(lambda tt: re.sub(r"[,.;@#?!&$`’]+", '', tt), _texts))

        return re.findall(regex_getter, _text)

    @staticmethod
    def line_desegmenter(_text, remove_punc=False):
        if remove_punc:
            return list(map(lambda tt: re.sub(r"[,.;@#?!&$`’]+", '', tt), _text))

        return list(_text, )

    @staticmethod
    def sequencer(segmented_sentences, inp_length=10):
        """
        :param segmented_sentences: array of sentence strings
        :param inp_length: # sentence length for encoder input
        """
        inp_sequence = []

        for sentence in segmented_sentences:
            temp_seq = list(map(lambda x: word2idx[x] if x in word2idx else word2idx["UNKNOWN_TOKEN"],
                                sentence.lower().split()))
            #  add pads to the end if sentence is short
            pads = [word2idx["PAD_TOKEN"] for _ in range(inp_length - len(temp_seq))]
            temp_seq.extend(pads)
            temp_seq = temp_seq[:inp_length]  # crop the sentence if it exceeds inp_length
            temp_seq.insert(0, word2idx["START_SEQUENCE"])
            temp_seq.append(word2idx["STOP_SEQUENCE"])
            inp_sequence.append(temp_seq)

        return inp_sequence


class Batcher:

    def __init__(self, inp_length=15, batch_size=32):
        self.prep = Preprocessing()
        self.batch_size = batch_size
        self.nbatch = 0  # number of batches (train_sentence_count / batch_size)
        self.inp_length = inp_length  # max number of word count in a sentence
        self.sent_vectors = []

    def load_train_dataset(self, dir_train_data, b_remove_punc=True):
        with open(os.path.join(DATA_DIR, dir_train_data), "rb") as train_dataset:
            sentences = []
            for line in train_dataset:
                # sentence_tokenized = line.decode(encoding='utf-8').split()
                sentences.append(self.prep.sentence_desegmenter(_text=line.decode(encoding='utf-8'),
                                                                remove_punc=b_remove_punc))
            sentences_unnested = [sentence[0] for sentence in sentences]
            self.sent_vectors = self.prep.sequencer(sentences_unnested, self.inp_length)
            self.new_batch()

    def new_batch(self):
        self.nbatch = len(self.sent_vectors) / self.batch_size
        self.sent_vectors = shuffle(self.sent_vectors)
        self.__iter__()  # reset iterator

    def next_batch(self):
        return self.__next__()

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):

        if self.n <= self.nbatch:
            self.n += 1
            return self.sent_vectors[self.n * self.batch_size:(self.n + 1) * self.batch_size]
        else:
            # Generate new batch
            self.new_batch()


if __name__ == "__main__":
    myprep = Preprocessing()
    input_text = u"<s>Dün gece, Türkiye’nin bugüne kadarki en büyük ve en canlı konseri yapıldı</s>" \
                 "<s>Bu sefer, kariyer konusuna mizahla mı dalıyorsun</s>"

    segmented_sentences = myprep.sentence_desegmenter(_text=input_text, remove_punc=True)
    print(segmented_sentences)

    print(myprep.sequencer(segmented_sentences, 15))

    batcher = Batcher(batch_size=5)
    batcher.load_train_dataset(dir_train_data="text_sample.txt")
    print(batcher.next_batch())
    print(batcher.next_batch())
    # TODO check indexes start stop  etc

    # dataset formats, json data text vs json or db