from torch.utils.data import Dataset
from my_vocab_v1 import MyVocab
from collections import Counter
from tqdm import tqdm
import torch
import spacy
import pickle
import os
import re



class NerDataset(Dataset):

    def __init__(self,
                 dataset_path=None,
                 window_size=30,
                 window_shift=10,
                 train=True):
        self.train = train
        self.vocab = None
        self.vocab_label = None
        self.encoded_data = None
        self.window_size = window_size
        self.window_shift = window_shift
        if train:
            assert dataset_path is not None, 'you are creating a train dataset,' \
                                             '\n you MUST pass at least a dataset_file path'
            self.raw_data = self.read_data(dataset_path)
            self.window_data = self.create_window()
            self.create_vocab()
            self.encode_dataset()
        else:  # test
            print('the class is created is test moooood: '
                  '\ncall load_vocab(path) for continue to use it')

    def __len__(self):
        return len(self.window_data) if self.train else None

    def __getitem__(self, item):
        return self.encoded_data[item] if self.train else None

    def read_data(self, file_path):
        print('READING DATA')
        data = []
        temp_sentence = []
        temp_labels = []
        with open(file_path, 'r') as file:
            print('opening file...')
            for line in tqdm(file):
                if line.startswith('#'):
                    continue
                elif line.startswith('\n'):
                    data.append({'sentence': temp_sentence,
                                 'labels': temp_labels})
                    temp_sentence = []
                    temp_labels = []
                else:
                    word, label = line.split('\t')
                    word = self.preprocess(word)
                    label = label[:-1]  # remove '\n'
                    temp_sentence.append(word)
                    temp_labels.append(label)
        print('DATA READING COMPLETED')
        return data

    def create_window(self):
        data = []
        step = self.window_shift

        for sample in self.raw_data:
            for i in range(0, len(sample['sentence']), step):
                sentence_window = sample['sentence'][i: i + self.window_size]
                label_window = sample['labels'][i: i + self.window_size]
                len_sample_window = len(sentence_window)

                if len_sample_window < self.window_size:
                    sentence_window = sentence_window + \
                                      [None] * (self.window_size - len_sample_window)
                    label_window = label_window + \
                                   [None] * (self.window_size - len_sample_window)

                data.append({'sentence': sentence_window,
                             'labels': label_window})
        return data

    def create_vocab(self):
        print('START BUILD VOCABS')
        self.vocab = self.build_vocab(special_tokens=['<pad>', '<unk>'])
        self.vocab_label = self.build_vocab_label(special_tokens=['<pad>'])
        print('VOCABS OK')

    def build_vocab(self, special_tokens, min_freq=1):
        counter = Counter()
        for i in tqdm(range(len(self.raw_data))):
            for word in self.raw_data[i]['sentence']:
                if word is not None:
                    counter[word] += 1
        return MyVocab(counter,
                       min_freq=min_freq,
                       special_tokens=special_tokens)

    def build_vocab_label(self, special_tokens):
        data = ['B-PER', 'B-LOC', 'B-GRP', 'B-CORP', 'B-PROD',
                'B-CW', 'I-PER', 'I-LOC', 'I-GRP', 'I-CORP',
                'I-PROD', 'I-CW', 'O']
        return MyVocab(data, special_tokens=special_tokens, from_dict=False)

    def preprocess(self, word):
        '''function for normalize input form: convert Number in fixed form '''
        # convert 4 digit in sd format : XXXX ---> Date\n",
        pattern = r'^[0-9]{4}$'
        if re.match(pattern, word):
            return 'XXXX'
        else:
            return word

    def encode_dataset(self):
        self.encoded_data = list()
        for i in range(len(self.window_data)):
            elem = self.window_data[i]
            encoded_x = torch.LongTensor(self.encode_sentence(elem['sentence']))

            tag_list = [self.vocab_label[tag] if tag is not None
                        else self.vocab_label['<pad>'] for tag in elem['labels']]
            encoded_y = torch.LongTensor(tag_list)
            self.encoded_data.append({"inputs": encoded_x,
                                      "outputs": encoded_y})

    def encode_sentence(self, sentence):
        indices = list()
        for word in sentence:
            if word is None:
                indices.append(self.vocab["<pad>"])
            elif word in self.vocab.stoi:  # vocabulary string to integer
                indices.append(self.vocab[word])
            else:
                indices.append(self.vocab.unk_index)
        return indices

    def encode_test(self, sentence):
        indices = list()
        for word in sentence:
            if word in self.vocab.stoi:  # vocabulary string to integer
                indices.append(self.vocab[word])
            else:
                indices.append(self.vocab.unk_index)
        return indices

    def decode_text(self, sentence_id_form):
        sentence = []
        for word_id in sentence_id_form:
            sentence.append(self.vocab.itos[word_id])
        return sentence

    def decode_tag(self, prediction_id_tag, len_origins):
        tags = []
        counter = 0
        for tag_id in prediction_id_tag:
            if counter < len_origins:
                tags.append(self.vocab_label.itos[tag_id])
                counter += 1
        return tags

    def save_vocabs(self, path='.'):
        vocab_file = open(os.path.join(path, 'vocab.txt'), 'wb')
        vocab_tag_file = open(os.path.join(path, 'vocab_tag.txt'), 'wb')

        pickle.dump(self.vocab, vocab_file)
        pickle.dump(self.vocab_label, vocab_tag_file)

        vocab_file.close(), vocab_tag_file.close()

        print('all the vocab are saved in {}'.format('current path' if path == '.' else path))

    def load_vocab(self, path_dir='.'):
        vocab_path = os.path.join(path_dir, 'vocab.txt')
        vocab_tag_path = os.path.join(path_dir, 'vocab_tag.txt')

        with open(vocab_path, 'rb') as vocab_file, \
                open(vocab_tag_path, 'rb') as vocab_tag_file:
            self.vocab = pickle.load(vocab_file)
            self.vocab_label = pickle.load(vocab_tag_file)
        print('loaded all vocab successfully')
