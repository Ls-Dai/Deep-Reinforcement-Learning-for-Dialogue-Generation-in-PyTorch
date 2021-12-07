import argparse
import torch
from seq2seq import Seq2Seq
from dataloading import Data, EOS_IDX
from utils import reverse, truncate, concat
from nltk.tokenize.toktok import ToktokTokenizer
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda')


class Simulator:

    def __init__(self, model, state_dict=None, reward_func=None, criterion=None):
        self.agentA = model
        self.agentB = model
        if state_dict is not None:
            self.agentA.load_state_dict(torch.load(state_dict))
            self.agentB.load_state_dict(torch.load(state_dict))
        self.agent = [self.agentA, self.agentB]
        self.agent_name = ["agent A", "agent B"]
        self.reward_func = reward_func
        self.criterion = criterion

    def simulate(self, batch, reward_func, turn=3):
        input_message = batch.hist1
        history1 = input_message
        reward = 0
        for t in range(turn):
            agent = self.agent[(t % 2)]
            logits_matrix, decoder_out = agent.generate(input_message)  # type of decoder out = [data, lenght]
            reward += reward_func(logits_matrix)

            history2 = decoder_out
            input_message = concat(history1, history2)
            history1 = history2
        return reward

    def demo_a2a(self, _data, turn=5):  # demo for agent-to-agent
        print("======================== DEMO ===========================")
        input_message = input("Enter input message: ")
        print("      input message: %s" % input_message)
        input_tokens = input_message.split() + ['<eos>']
        input_tensor = torch.tensor([_data.vocab.stoi[i] for i in input_tokens]).to(device)
        input_batch = (input_tensor.unsqueeze(0), torch.LongTensor([input_tensor.size(0)]))
        history1 = input_batch
        for t in range(turn):
            agent = self.agent[(t % 2)]
            _, decoder_out = agent.generate(input_batch)  # type of decoder out = [data, lenght]
            out_message = reverse(decoder_out[0], _data.vocab)
            print("   (turn %d) %s: %s" % ((t+1), self.agent_name[(t % 2)], out_message[0]))
            # init for next turn
            history2 = decoder_out
            input_batch = concat(history1, history2)
            history1 = history2

    def demo_u2a(self, _data):  # demo for user-to-agent
        tokenizer = ToktokTokenizer()
        agent = self.agent[0]
        print("======================== DEMO ===========================")
        input_message = input("     (User): ")
        if input_message.lower == "bye":
            print("    (Agent): Bye!")
            return
        input_tokens = tokenizer.tokenize(input_message) + ['<eos>']
        input_tensor = torch.tensor([_data.vocab.stoi[i] for i in input_tokens]).to(device)
        input_batch = (input_tensor.unsqueeze(0), torch.LongTensor([input_tensor.size(0)]))
        while True:
            _, decoder_out = agent.generate(input_batch)  # type of decoder out = [data, lenght]
            decoded_message = reverse(decoder_out[0][:, :-1], _data.vocab)
            print("    (Agent): %s" % decoded_message[0])
            # init for next turn
            history1 = decoder_out

            input_message = input("     (User): ")
            if input_message.lower() == "bye":
                print("    (Agent): Bye!")
                break
            input_tokens = input_message.split() + ['<eos>']
            input_tensor = torch.tensor([_data.vocab.stoi[i] for i in input_tokens]).to(device)
            input_batch = (input_tensor.unsqueeze(0), torch.LongTensor([input_tensor.size(0)]))
            history2 = input_batch
            input_batch = concat(history1, history2)

    def debug(self, data, turn=3, sample_num=10):
        logger.info("Debugging ...")
        for i, batch in enumerate(data.train_iter):
            assert len(batch) == 1, 'batch size must be 1 in debugging'
            input_message = batch.hist1
            history1 = input_message[0]
            logger.info(" sample %d ", (i+1))
            for t in range(turn):
                agent = self.agent[(t % 2)]
                logits_matrix = agent.generate(input_message)
                _, decoder_out = logits_matrix.max(dim=2)
                logger.info("[turn %d] IN : " + " ".join(reverse(input_message[0], data.vocab)), (t+1))
                logger.info("[turn %d] OUT: " + " ".join(reverse(decoder_out, data.vocab)), (t+1))

                history2 = decoder_out
                input_message = torch.cat([history1, history2], dim=1)
                input_message = [input_message, torch.LongTensor([input_message.size(1)])]
                history1 = history2
            if (i+1) == sample_num:
                break


if __name__ == '__main__':
    datadir = 'data'
    device = torch.device('cuda')
    embed_size = 300

    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dict', default='forward_epoch1.pt')
    parser.add_argument('--mode', default='user2agent', choices=['agent2agent', 'user2agent'])
    args = parser.parse_args()

    data = Data(datadir, device, batch_size=1, use_glove=False)
    vocab_size = len(data.vocab)

    model = Seq2Seq(vocab_size, embed_size, embedding_weight=data.vocab.vectors).to(device)
    simulator = Simulator(model, args.state_dict)
    if args.mode == 'agent2agent':
        simulator.demo_a2a(data)
    elif args.mode == 'user2agent':
        simulator.demo_u2a(data)

