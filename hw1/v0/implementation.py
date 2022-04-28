import torch
import numpy as np
from typing import List, Tuple


from model import Model

from my_code.v0.dataset import NerDataset
from my_code.v0.my_model import NerModel
from my_code.v0.Params import Params


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device)

class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self, device: str):
        super(StudentModel, self).__init__()
        self.device = device
        self.dataset = NerDataset(train=False)
        self.dataset.load_vocab(path_dir='model/vocab_file')

        self.model = NerModel(Params(
            len(self.dataset.vocab),
            len(self.dataset.vocab_label)
        ))
        self.model.load_state_dict(torch.load('model/model_weights/base_line.pth',
                                              map_location={'cuda:0': 'cpu'}
                                              ))
        self.model.to(self.device)
        # end init

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        out = []
        for token in tokens:
            x = self.dataset.encode_test(token)
            x = torch.LongTensor(x).to(self.device)
            pred = self.model.predict(x)
            out.append(self.dataset.decode_tag(pred, len(token)))
        return out


if __name__ == '__main__':
    print('STA ANDANDO TUTTO MALE ')
    model = build_model('cpu')
