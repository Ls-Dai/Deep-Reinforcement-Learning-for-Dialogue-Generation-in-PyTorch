import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from seq2seq import Seq2Seq
from dataloading import Data, PAD_IDX
from utils import reverse


class Trainer():

    def __init__(self, model, lr=0.001):
        self.lr = lr
        self.model = model
        self.loss_function = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optim = optim.Adam(self.model.parameters(), lr)

    def train(self, data, epoch=10):
        for ep in range(epoch):
            # for i, batch in enumerate(tqdm(data.train_iter)):
            for batch in data.train_iter:
                self.optim.zero_grad()
                batch_size, seq_len = batch.resp[0].size()
                softmax = self.model(batch.merged_hist, batch.resp)
                loss = self.loss_function(
                    softmax.contiguous().view(batch_size*seq_len, -1), batch.resp[0].view(-1))
                loss.backward()
                self.optim.step()
            print("[epoch %d] loss = %.4f" % (ep+1, loss))
        return self.model

    def save_model(self, state_dict_name='model.bin'):
        torch.save(self.model.state_dict(), state_dict_name)


if __name__ == '__main__':
    datadir = 'data'
    # device = torch.device('cuda')
    device = torch.device('cpu')
    embed_size=300

    data = Data(datadir, device, batch_size=64, use_glove=False)
    vocab_size = len(data.vocab)

    model = Seq2Seq(vocab_size, embed_size, embedding_weight=data.vocab.vectors).to(device)
    trainer = Trainer(model)
    model = trainer.train(data, epoch=100)

    # eval
    for batch in data.test_iter:
        softmax = model(batch.merged_hist, batch.resp)
        _, argmax = softmax.max(dim=2)
        print(reverse(batch.resp[0], data.vocab))
        print(reverse(argmax, data.vocab))
        break
    trainer.save_model('seq2seq.bin')

