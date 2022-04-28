import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from my_model import NerModel
from dataset import NerDataset
from Params import Params

class Trainer:

    def __init__(self,
                 model: nn.Module,
                 loss_funct,
                 optimizer,
                 device: str,
                 log_steps: int = 10000,
                 patience: int = 3):
        self.model = model
        self.loss_funct = loss_funct
        self.optimizer = optimizer
        self.log_steps = log_steps
        self.device = device
        self.MAX_PATIENCE = patience

    def train(self,
              train_dataset: DataLoader,
              val_dataset: DataLoader,
              epochs: int = 20):
        assert epochs > 1 and isinstance(epochs, int)
        print('START TRAINING')
        previous_loss = 100 #symbolic value for early stopping
        train_loss = 0.0
        trigger_times = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            self.model.train()

            for step, sample in enumerate(train_dataset):
                inputs = sample['inputs'].to(self.device)
                labels = sample['outputs'].to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model(inputs)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)

                sample_loss = self.loss_funct(predictions, labels)
                sample_loss.backward()
                self.optimizer.step()

                epoch_loss += sample_loss.tolist()
                if step % self.log_steps == self.log_steps - 1:
                    print('\t[E: {:2d} @ step {}]'
                          'current avg loss = {:0.4f}'.format(epoch, step, epoch_loss / (step + 1)))

            avg_loss = epoch_loss / len(train_dataset)
            train_loss += avg_loss
            validation_loss = self.evaluate(val_dataset)
            print('\t[E: {:2d}] train loss'
                  '= {:0.4f}'.format(epoch, avg_loss))
            print('\t[E: {:2d}] valid loss ='
                  '{:0.4f}'.format(epoch, validation_loss))
            print('__________________________________')

            # EARLY STOPPING
            '''
            if epoch > 50:
                if validation_loss > previous_loss:
                    trigger_times += 1
                    print('TRIGGER TIME: ', trigger_times)
                    if trigger_times >= self.MAX_PATIENCE:
                        print("EARLY STOPPING")
                        break
                else:
                    print('trigger times: 0')
                    trigger_times = 0
                previous_loss = validation_loss
            '''

        print('TRAIN COMPLETE')
        avg_loss = train_loss / epochs
        return avg_loss

    def evaluate(self, validation_dataset):
        validation_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in validation_dataset:
                inputs = sample['inputs'].to(self.device)
                labels = sample['outputs'].to(self.device)
                predictions = self.model(inputs)

                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)

                sample_loss = self.loss_funct(predictions, labels)
                validation_loss += sample_loss.tolist()
        return validation_loss / len(validation_dataset)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            predictions = torch.argmax(logits, -1)
            return logits, predictions


if __name__ == '__main__':

    torch.manual_seed(28)

    TRAIN_PATH = '../../../data/train.tsv'
    DEV_PATH = '../../../data/dev.tsv'
    DEVICE = 'cuda:0'

    # create Dataset

    dataset = NerDataset(TRAIN_PATH)
    devset = NerDataset(DEV_PATH)
    # create Vocab
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    dev_loader = DataLoader(devset, batch_size=128, shuffle=False)

    params = Params(vocab_size=len(dataset.vocab),
                    tag_size=len(dataset.vocab_label))
    ner_model = NerModel(params).to(DEVICE)

    trainer = Trainer(model=ner_model,
                      loss_funct=nn.CrossEntropyLoss(ignore_index=dataset.vocab_label['<pad>']),
                      optimizer=optim.SGD(ner_model.parameters(), lr=0.001, momentum=0.9),
                      device=DEVICE,
                      )
    trainer.train(train_loader, dev_loader, epochs=700)
    torch.save(ner_model.state_dict(), '../../../model/model_weights/base_line_700_NoStop.pth')


