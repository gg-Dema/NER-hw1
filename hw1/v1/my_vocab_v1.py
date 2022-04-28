from collections import Counter

class MyVocab:
    def __init__(self,
                 data,
                 special_tokens: list,
                 min_freq=None,
                 from_dict=True):
        assert len(special_tokens) <= 2, f'special_tokens: {special_tokens}'
        assert from_dict is False or min_freq <= 1, f'min_freq: {min_freq}'

        self.data = data
        if from_dict:
            self.data = self.remove_min_freq(data, min_freq)
        self.special_tokens = special_tokens
        self.unk_index = None
        self.pad_index = None
        self.stoi, self.itos = self.create_data_struct()

    def remove_min_freq(self, frequency_dict: Counter, min_freq:int):
        return {word: frequency_dict[word] for word in frequency_dict
                if frequency_dict[word] > min_freq}

    def create_data_struct(self):
        stoi, itos = self.add_special()
        for idx, word in enumerate(self.data, len(stoi)):
            itos.append(word)
            stoi[word] = idx
        return stoi, itos

    def add_special(self):
        stoi = {}
        itos = []
        stoi['<pad>'] = 0
        itos.append('<pad>')
        self.pad_index = 0

        if len(self.special_tokens) == 2:
            stoi['<unk>'] = 1
            itos.append('<unk>')
            self.unk_index = 1

        return stoi, itos

    def __getitem__(self, key:str):
        if key in self.stoi:
            return self.stoi[key]
        return self.stoi['<unk>']

    def __len__(self):
        return len(self.itos)
