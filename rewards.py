import torch

from dataloading import PAD_IDX
from seq2seq import Seq2Seq

class BaseReward:
    def __init__(self):
        pass

    def __call__(self):
        pass

# same as semantic coherence
class MutualInformation(BaseReward):
    def __init__(self, net_forward, net_backward):
        self.net_forward = net_forward
        self.net_backward = net_backward
        self.dist = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='none') # helper

    # TODO: y is generated sequence, not golden data
    @staticmethod
    def _calculate_prob(net, dist, x):
        with torch.no_grad():
            logits = net(x, y)
            generated = net.decode(x)
            generated = generated.view(-1)
            B, L, _ = logits.size()
            logits = logits.view(B*L, -1)
            # TODO: check dim=1
            logprob = (dist(logits, generated).view(B, L).sum(dim=1)
                       / lengths).mean()
        return logprob

    def __call__(self, batch):
        logprob_forward = self._calculate_prob(self.net_forward, self.dist,
                                               batch.merged_hist)
        logprob_backward = self._calculate_prob(self.net_backward, self.dist,
                                               batch.resp)
        return logprob_forward + logprob_backward

class EaseAnswering(BaseReward):
    def __init__(self, net):
        self.net = net
        self.S = ["I don't know what you're talking about.",
		"I don't know.", "You don't know.",
		"You know what I mean.",
		"I know what you mean.",
		"You know what I'm saying.",
		"You don't know anything."]

# TODO specify network name with func args
def get_mutual_information(forward_path, backward_path, *args, **kwargs):
    net_forward = Seq2Seq.load(forward_path, *args, **kwargs)
    net_backward = Seq2Seq.load(backward_path, *args, **kwargs)
    return MutualInformation(net_forward, net_backward)
