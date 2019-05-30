import os
from collections import Counter
from string import punctuation

import io
import re
import time

import hyperparameter as hp

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary(OOV) words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences

alphabets = "([A-Za-z])"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

class Preprocess():
    def __init__(self):
        self.freq_thresh = hp.VOCAB_FREQ_THRESHOLD
        self.padding_size = self.calculate_padding_size(hp.INPUT_PATH, hp.WEIGHTED_SOME_COEFFICIENT,
                                                       hp.WEIGHTED_SOME_NMOST, hp.PADDING_ALGORITHM)
        if hp.SELF_VOCAB:
            self.create_dict_from_data(hp.INPUT_PATH)
        else:
            self.create_dict_from_existing_dict()
    def word_to_index(self, word):
        return self.vocab.index(word)

    def index_to_word(self, ind):
        return self.vocab[ind]

    @staticmethod
    def remove_punctuations(text):
        return ''.join(c for c in text if c not in punctuation).lower()

    @staticmethod
    def remove_digits(text):
        return text.translate(str.maketrans('', '', '01234567890'))
        # return ''.join(filter(lambda x: x.isalpha(), text))

    @staticmethod
    def remove_stopwords(text, tokenize=True):
        with io.open('utils/stopwords.txt', 'r', encoding='utf8') as file:
            stopwords = file.read().splitlines()
        stopped = []
        for token in text.split():
            if token not in stopwords:
                stopped.append(token)
        if tokenize:
            return stopped
        else:
            return "".join(stopped)

    @staticmethod
    def tokenize(text):
        return text.split()

    def add_preprocess_tokens(self):
        pass

    def create_dict_from_data(self, path):
        self.vocab = []
        st = time.time()
        for special_token in [UNKNOWN_TOKEN, PAD_TOKEN, SENTENCE_END, START_DECODING, STOP_DECODING]:
            self.vocab.append(special_token)
        corpus_vocab = Counter()
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".txt"):
                    with io.open(os.path.join(root, file), 'r', encoding='utf8') as textfile:
                        if hp.REMOVE_STOPWORDS:
                            corpus_vocab += Counter(
                                self.remove_stopwords(self.remove_punctuations(self.remove_digits(textfile.read()))))
                        else:
                            corpus_vocab += Counter(self.remove_punctuations(self.remove_digits(textfile.read())).split(' '))
        vocab = [word for word, occurrences in corpus_vocab.items() if occurrences >= self.freq_thresh]
        self.vocab.extend(vocab)
        et = time.time() - st
        print(f"Vocabulary is created in {round(et,2)} secs.")
        print(f"Vocabulary size: {len(self.vocab)}")

    def create_dict_from_existing_dict(self):
        self.vocab = []
        st = time.time()
        with io.open('utils/stopwords.txt', 'r', encoding='utf8') as file:
            stopwords = file.read().splitlines()
        for special_token in [UNKNOWN_TOKEN, PAD_TOKEN, SENTENCE_END, START_DECODING, STOP_DECODING]:
            self.vocab.append(special_token)
        with io.open('utils/vocab.txt', 'r', encoding='utf8') as vocabfile:
            content = vocabfile.readlines()
        vocab = [line.strip().split(' ')[0] for line in content if int(line.strip().split(' ')[1]) >= self.freq_thresh]
        if hp.REMOVE_STOPWORDS:
            vocab = [item for item in vocab if item not in stopwords]
        self.vocab.extend(vocab)
        self.vocab_size = len(self.vocab)
        et = time.time() - st
        print(f"Vocabulary is created in {round(et,2)} secs.")
        print(f"Vocabulary size: {self.vocab_size}")

    def convert_word_list_to_indexes(self, word_list):
        indexes = []
        for word in word_list:
            indexes.append(self.word_to_index(word))
        return indexes

    def findOccurrences(self, s, ch):
        return [i for i, letter in enumerate(s) if letter == ch]

    def unify_numbers(self, text):
        a = self.findOccurrences(text, ".")
        for i in a:
            if 0 < i < len(text) - 1:
                if text[i + 1].isnumeric() or text[i - 1].isnumeric():
                    t = list(text)
                    t[i] = ""
                    text = "".join(t)
        return text

    def calculate_padding_size(self, path, c=0.8, n=10, algorithm='max_length'):
        sentence_lengths = Counter()
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".txt"):
                    with io.open(os.path.join(root, file), 'r', encoding='utf8') as textfile:
                        sentences = [x.strip() for x in textfile]
                        if hp.REMOVE_STOPWORDS:
                            sentence_lengths += Counter(len(self.remove_stopwords(x)) for x in sentences if x)
                        else:
                            sentence_lengths += Counter(len(x.split(' ')) for x in sentences if x)
        if algorithm == 'max_length':
            print(f"Padding size: {max(sentence_lengths.keys())}")
            return max(sentence_lengths.keys())
        elif algorithm == 'weighted_sum':
            total_words = 0
            total_occurence = 0
            for s_len, occurence in sentence_lengths.most_common(n):
                total_words += s_len * occurence
                total_occurence += occurence
            print(f"Padding size: {round(c * total_words / total_occurence)}")
            return round(c * total_words / total_occurence)

    def add_tokens_to_sentence(self, sentence):
        sentence_words = sentence.split(' ')
        sentence_tokens = []
        for word in sentence_words:
            if word in self.vocab:
                sentence_tokens.append(word)
            else:
                sentence_tokens.append('[UNK]')
        sentence_length = len(sentence_tokens)
        if sentence_length < self.padding_size:
            sentence_tokens.append('</s>')
            sentence_length += 1
            while sentence_length < self.padding_size:
                sentence_tokens.append('[PAD]')
                sentence_length += 1
            return sentence_tokens
        else:
            return sentence_tokens[:self.padding_size]

    def decoder_output_check_sentence(self, sentence):
        sentence_words = sentence.split(' ')
        decoder_output_check = [START_DECODING]
        for word in sentence_words:
            if word in self.vocab:
                decoder_output_check.append(word)
            else:
                decoder_output_check.append('[UNK]')
        sentence_length = len(decoder_output_check)
        if sentence_length < self.padding_size + 1:
            decoder_output_check.append(STOP_DECODING)
            sentence_length += 1
            while sentence_length < self.padding_size + 2:
                decoder_output_check.append('[PAD]')
                sentence_length += 1
            return decoder_output_check
        else:
            decoder_output_check = decoder_output_check[:self.padding_size+1]
            decoder_output_check.append(STOP_DECODING)
            return decoder_output_check

    def split_into_sentences(self,text):
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = self.unify_numbers(text)
        text = re.sub(websites, "<prd>\\1", text)
        text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
        if "”" in text: text = text.replace(".”", "”.")
        if "\"" in text: text = text.replace(".\"", "\".")
        if "!" in text: text = text.replace("!\"", "\"!")
        if "?" in text: text = text.replace("?\"", "\"?")
        text = text.replace("...", ".")
        text = text.replace(":", ".")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences

if __name__ == '__main__':
    p = Preprocess()
    print(p.convert_word_list_to_indexes(p.add_tokens_to_sentence('Ortak vizyonumuz nerde kaldi')))
    print((p.decoder_output_check_sentence('Ortak vizyonumuz nerde kaldi')))