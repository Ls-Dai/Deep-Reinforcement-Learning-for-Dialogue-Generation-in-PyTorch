import os
import logging

import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator


MAXLEN = 22
logger = logging.getLogger(__name__)

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
# TODO: <SEP> needed?


class Data(object):
    def __init__(self, data_dir, device, batch_size, use_glove=False):
        self.device = device
        self.train_path = os.path.join(data_dir, 'train/formatted_dialogues_train.txt')
        self.test_path = os.path.join(data_dir, 'test/formatted_dialogues_test.txt')
        self.build(use_glove, batch_size)
        self.report_stats()

    def report_stats(self):
        logger.info('data size... {} / {} / {}'.\
                    format(len(self.train), len(self.val), len(self.test)))
        logger.info('vocab size... {}'.format(len(self.vocab)))

    def build(self, use_glove, batch_size):
        logger.info('building field, dataset, vocab, dataiter...')
        HIST, RESP = self.build_field(maxlen=MAXLEN)
        self.train, self.val, self.test = self.build_dataset(HIST, RESP)
        sources = [self.train.hist, self.train.resp,
                   self.val.hist, self.val.resp]
        self.vocab = self.build_vocab(HIST, RESP, sources, use_glove)
        self.train_iter, self.valid_iter, self.test_iter =\
            self.build_iterator(self.train, self.val, self.test, batch_size)

    def build_field(self, maxlen=None):
        HIST = Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        eos_token='<eos>', tokenize='toktok')
        RESP = Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        eos_token='<eos>', tokenize='toktok')
        return HIST, RESP

    def build_dataset(self, HIST, RESP):
        train_val = TabularDataset(path=self.train_path, format='tsv',
                                fields=[('hist', HIST), ('resp', RESP)])
        train, val = train_val.split(split_ratio=0.8)
        test = TabularDataset(path=self.train_path, format='tsv',
                                fields=[('hist', HIST), ('resp', RESP)])
        filtered = 0
        for data in [train, val, test]: # merging hist1 and hist2
            data.fields['merged_hist'] = data.fields['hist']
            before = len(data.examples)
            data.examples = [ex for ex in data if all([hasattr(ex, 'hist'),
                                                       hasattr(ex, 'resp')])]
            after = len(data.examples)
            filtered += (before - after)
            for ex in data.examples:
                setattr(ex, 'merged_hist', ex.hist)
        logger.info('number of examples filtered: {}'.format(filtered))
        return train, val, test

    def build_vocab(self, HIST, RESP, sources, use_glove=False):
        v = 'glove.6B.300d' if use_glove else None
        HIST.build_vocab(sources, max_size=30000, vectors=v)

        HIST.vocab.itos.insert(2, '<sos>')
        from collections import defaultdict
        stoi = defaultdict(lambda: 0)
        stoi.update({tok: i for i, tok in enumerate(HIST.vocab.itos)})
        HIST.vocab.stoi = stoi
        RESP.vocab = HIST.vocab
        return HIST.vocab

    def build_iterator(self, train, val, test, batch_size=32):
        train_iter, valid_iter, test_iter = \
        BucketIterator.splits((train, val, test), batch_size=batch_size,
                              sort_key=lambda ex: (len(ex.merged_hist),
                                                   len(ex.resp)),
                              sort_within_batch=True, repeat=False,
                              device=self.device)
        return train_iter, valid_iter, test_iter

if __name__ == '__main__':
    datadir = 'data'
    
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    data = Data(datadir, device, batch_size=2, use_glove=False)

    for batch in data.train_iter:
        print(batch)
    print(data.vocab.itos)