import torch

from dataloading import Data
from utils import reverse, truncate

if __name__ == '__main__':
    datadir = 'data'
    device = torch.device('cuda')

    data = Data(datadir, device, batch_size=2, use_glove=False)

    # check batch
    for batch in data.train_iter:
        print(batch)
        print('-----> batch has 4 attributes: hist1, hist2, hist_merged, resp\n')

        print(batch.hist1[0], batch.hist1[1])
        print('-----> hist1 is a tuple of data tensor and lengths\n')

        hist1, lengths = truncate(batch.hist1, 'sos')
        print(reverse(hist1, data.vocab))
        print('-----> use truncate to get rid of sos/eos token in a batch, use reverse to denumericalize\n')
        break

    # check vocab
    print(len(data.vocab))
    print(data.vocab.itos)
    print('-----> vocab len, itos')
