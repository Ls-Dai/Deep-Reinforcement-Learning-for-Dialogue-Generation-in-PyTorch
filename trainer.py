import os
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nlgeval import NLGEval

from seq2seq import Seq2Seq
from dataloading import PAD_IDX
from utils import truncate

logger = logging.getLogger(__name__)


# TODO: Stats, EarlyStopper, Simulator -> train_utils.py
# TODO: running average
class Stats:
    def __init__(self, records):
        self.records = records
        self.reset_stats()

    def reset_stats(self):
        self.stats = {name: [] for name in self.records}

    def record_stats(self, *args):
        assert len(self.records) == len(args)
        for name, loss in zip(self.records, args):
            self.stats[name].append(loss.item())

    def report_stats(self, epoch, step='N/A'):
        to_report = {}
        for name in self.records:
            to_report[name] = np.mean(self.stats[name])
        logger.info('stats at epoch {} step {}:\n'\
                    .format(epoch, step) + str(to_report))


class EarlyStopper:
    def __init__(self, patience, metric):
        self.patience = patience
        self.metric = metric # 'Bleu_1', ..., 'METEOR', 'ROUGE_L'
        self.count = 0
        self.best_score = defaultdict(lambda: 0)
        self.is_improved = False

    def stop(self, cur_score):
        if self.best_score[self.metric] > cur_score[self.metric]:
            self.is_improved = False
            if self.count <= self.patience:
                self.count += 1
                logger.info('Counting early stop patience... {}'\
                            .format(self.count))
                return False
            else:
                logger.info('Early stopping patience exceeded.\
                            Stopping training...')
                return True # halt training
        else:
            self.is_improved = True
            self.count = 0
            self.best_score = cur_score
            return False


class BaseTrainer:
    def __init__(self, model, data, lr,  clip, records, savedir):
        self.model = model
        self.data = data
        # TODO: implement get_optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.clip = clip
        self.stats = Stats(records)
        self.savedir = savedir

    def _compute_loss(self):
        raise NotImplementedError

    def _run_epoch(self, epoch, sort_key=None, verbose=True):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    # TODO: save_checkpoint like in DME code
    def save_model(self, epoch, savedir='models/'):
        logger.info('saving model in {}'.format(savedir))
        if not os.path.isdir(self.savedir):
            os.mkdir(self.savedir)
        filename = self.model.name + '_epoch{}.pt'.format(epoch)
        savedir = os.path.join(self.savedir, filename)
        torch.save(self.model.state_dict(), savedir)
        return savedir

    # TODO: evaluate and write to file!
    def evaluate(self, data_type, epoch):
        pass


class SupervisedTrainer(BaseTrainer):
    def __init__(self, model, data, backward=False, lr=0.001, clip=5,
                 records=None, savedir='models/'):
        super().__init__(model, data, lr, clip, records, savedir)
        self.backward = backward
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def _compute_loss(self, batch):
        if self.backward:
            logits = self.model(batch.resp, batch.hist2)
            target, _ = truncate(batch.hist2, 'sos')
        else:
            logits = self.model(batch.merged_hist, batch.resp)
            logits = logits[:, :-1, :]  # truncate 'eos' token
            target, _ = truncate(batch.resp, 'sos')
        B, L, _ = logits.size()
        loss = self.criterion(logits.contiguous().view(B*L, -1),
                              target.contiguous().view(-1))
        return loss

    def _run_epoch(self, epoch, sort_key=None, verbose=True):
        if sort_key is not None:
            self.data.train_iter.sort_key = sort_key
        for step, batch in enumerate(self.data.train_iter, 1):
            loss = self._compute_loss(batch)
            self.stats.record_stats(loss)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            # report train stats on a regular basis
            if verbose and (step % 100 == 0):
                self.stats.report_stats(epoch, step=step)

    def train(self, num_epoch, verbose=True):
        if self.backward: # to use packed_sequence...
            sort_key = lambda ex: (len(ex.resp), len(ex.hist2))
        else:
            sort_key = None
        for epoch in range(1, num_epoch+1, 1):
            self._run_epoch(epoch, sort_key, verbose)
            savedir = self.save_model(epoch)
        return {'savedir': savedir, 'stats': self.stats.stats}


class RLTrainer(BaseTrainer):
    def __init__(self, model, data, reward_func, lr=0.001, clip=5, turn=3,
                 records=None, savedir='models/',patience=3, metric='Bleu_1'):
        super().__init__(model, data, lr, clip, records, savedir)
        # TODO: variable name - criterion?
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX,
                                             reduction='none') # helper
        self.early_stopper = EarlyStopper(patience, metric)
        self.evaluator = NLGEval(no_skipthoughts=True, no_glove=True)
        #self.simulator = Simulator(model, reward_func, turn)

    def _compute_loss(self, batch):
        # calculate loss with simulator
        # ex)
        #    rewards = simulator.simulate(batch)
        #    loss = rewards * self.criterion(a, b)
        return loss

    def _run_epoch(self):
        # compute loss for every step
        # ex)
        #   for batch in iter:
        #       loss = self._compute_loss(batch)
        #       loss.backward()
        pass

    def train(self, num_epoch):

        for epoch in range(1, num_epoch+1, 1):
            self._run_epoch()

            # evaluate at the end of every epoch
            with torch.no_grad():
                pass

            # early stopping

        # save model to a file
        self.save_model(epoch)

        # report results on test data at the end of training

        return

