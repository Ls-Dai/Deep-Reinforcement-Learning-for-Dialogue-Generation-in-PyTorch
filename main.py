import logging
from setproctitle import setproctitle

import torch

from dataloading import Data
from seq2seq import Seq2Seq
from rewards import get_mutual_information
from trainer import SupervisedTrainer, RLTrainer

setproctitle("(hwijeen) RL dialogue")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


if  __name__ == "__main__":
    DATA_DIR = 'data/'
    DEVICE = torch.device('cuda:0')
    EPOCH = 1
    SUPERVISED_FORWARD = 'models/forward_epoch1.pt'
    SUPERVISED_BACKWARD = 'models/backward_epoch1.pt'
    MUTUAL_INFORMATION = None

    EMBEDDING = 300
    HIDDEN = 500
    data = Data(DATA_DIR, DEVICE, batch_size=64, use_glove=False)
    VOCAB_SIZE = len(data.vocab)

    ########################################################################
    # if MODE == 'SUPERVISED PRETRAIN'
    # supervised learning
    if SUPERVISED_FORWARD is None or SUPERVISED_BACKWARD is None:
        seq2seq_for = Seq2Seq(VOCAB_SIZE, EMBEDDING, HIDDEN, name='forward')\
            .to(DEVICE)
        seq2seq_back = Seq2Seq(VOCAB_SIZE, EMBEDDING, HIDDEN, name='backward')\
            .to(DEVICE)
        trainer_for = SupervisedTrainer(seq2seq_for, data, lr=0.001,
                                        records=['NLLLoss'])
        trainer_back = SupervisedTrainer(seq2seq_back, data, lr=0.001,
                                         records=['NLLLoss'], backward=True)
        results_for = trainer_for.train(num_epoch=EPOCH, verbose=True)
        results_back = trainer_back.train(num_epoch=EPOCH, verbose=True)
        SUPERVISED_FORWARD = results_for['savedir']
        SUPERVISED_BACKWARD = results_back['savedir']
    ########################################################################

    ########################################################################
    # elif MODE == 'RL PRETRAIN'
    # rl with mutual information
    if MUTUAL_INFORMATION is None:
        seq2seq_rl = Seq2Seq.load(SUPERVISED_FORWARD, VOCAB_SIZE, EMBEDDING,
                                  HIDDEN, name='mutual information')
        mi = get_mutual_information(SUPERVISED_FORWARD, SUPERVISED_BACKWRD,
                                        VOCAB_SIZE, EMBEDDING, HIDDEN)
        trainer_rl = RLTrainer(seq2seq_rl, data, mi, lr=0.001, clip=5,
                               records=['Mutual Informtion'])
    # rl with other rewards

    ########################################################################

    # elif MODE == 'RL'

