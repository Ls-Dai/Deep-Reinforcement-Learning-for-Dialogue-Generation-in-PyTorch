import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import numpy as np
from scipy.spatial import distance

from loss import mask_nll_loss
from seq2seq import *
from dataloading import *
from utils import *

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


def rl(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, batch_size, teacher_forcing_ratio):
    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    #print_losses = []
    response = []

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, n_total = mask_nll_loss(
                decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            #print_losses.append(mask_loss.item() * nTotal)
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, n_total = mask_nll_loss(
                decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            #print_losses.append(mask_loss.item() * nTotal)

            #ni or decoder_output
            response.append(topi)

    return loss, max_target_len, response


def ease_of_answering(input_variable, lengths, dull_responses, mask, max_target_len, encoder, decoder, batch_size, teacher_forcing_ratio):
    NS=len(dull_responses)
    r1=0
    for d in dull_responses:
        d, mask, max_target_len = output_var(d, voc)
        newD, newMask = transform_tensor_to_same_shape_as(d, input_variable.size())
        #tar, mask, max_target_len = convertTarget(d)
        forward_loss, forward_len, _ = rl(input_variable, lengths, newD, newMask, max_target_len, encoder, decoder, batch_size, teacher_forcing_ratio)
        # log (1/P(a|s)) = CE  --> log(P(a | s)) = - CE
        if forward_len > 0:
            r1 -= forward_loss / forward_len
    if len(dull_responses) > 0:
        r1 = r1 / NS
    return r1


def information_flow(responses):
    r2=0
    if(len(responses) > 2):
        #2 representations obtained from the encoder for two consecutive turns pi and pi+1
        h_pi = responses[-3]
        h_pi1 = responses[-1]
        # length of the two vector might not match
        min_length = min(len(h_pi), len(h_pi+1))
        h_pi = h_pi[:min_length]
        h_pi1 = h_pi1[:min_length]
        #cosine similarity 
        #cos_sim = 1 - distance.cosine(h_pi, h_pi1)
        cos_sim = 1 - distance.cdist(h_pi.cpu().numpy(), h_pi1.cpu().numpy(), 'cosine')
        #Handle negative cos_sim
        if np.any(cos_sim <= 0):
            r2 = - cos_sim
        else:
            r2 = - np.log(cos_sim)
        r2 = np.mean(r2)
    return r2

def semantic_coherence(input_variable, lengths, target_variable, mask, max_target_len, forward_encoder, forward_decoder, backward_encoder, backward_decoder, batch_size, teacher_forcing_ratio):
    #print("IN R3:")
    #print("Input_variable :", input_variable.shape)
    #print("Lengths :", lengths.shape)
    #print("Target_variable :", target_variable.shape)
    #print("Mask :", mask.shape)
    #print("Max_Target_Len :", max_target_len)
    r3 = 0
    forward_loss, forward_len, _ = rl(input_variable, lengths, target_variable, mask, max_target_len, forward_encoder, forward_decoder, batch_size, teacher_forcing_ratio)
    ep_input, lengths_trans = convert_response(target_variable, batch_size)
    #print("ep_input :", ep_input.shape)
    #print("Lengths transformed :", lengths_trans.shape)
    ep_target, mask_trans, max_target_len_trans = convert_target(target_variable, batch_size)
    #print("ep_target :", ep_target.shape)
    #print("mask transformed :", mask_trans.shape)
    #print("max_target_len_trans :", max_target_len_trans)
    backward_loss, backward_len, _ = rl(ep_input, lengths_trans, ep_target, mask_trans, max_target_len_trans, backward_encoder, backward_decoder, batch_size, teacher_forcing_ratio)
    if forward_len > 0:
        r3 += forward_loss / forward_len
    if backward_len > 0:
        r3+= backward_loss / backward_len
    return r3

l1=0.25
l2=0.25
l3=0.5

dull_responses = ["i do not know what you are talking about.", "i do not know.", "you do not know.", "you know what i mean.", "i know what you mean.", "you know what i am saying.", "you do not know anything."]

MIN_COUNT = 3
MAX_LENGTH = 15

def calculate_rewards(input_var, lengths, target_var, mask, max_target_len, forward_encoder, forward_decoder, backward_encoder, backward_decoder, batch_size, teacher_forcing_ratio):
    #rewards per episode
    ep_rewards = []
    #indice of current episode
    ep_num = 1
    #list of responses
    responses = []
    #input of current episode
    ep_input = input_var
    #target of current episode
    ep_target = target_var
    
    #ep_num bounded -> to redefine (MEDIUM POST)
    while (ep_num <= 10):
        
        print(ep_num)
        #generate current response with the forward model
        _, _, curr_response = rl(ep_input, lengths, ep_target, mask, max_target_len, forward_encoder, forward_decoder, batch_size, teacher_forcing_ratio)
        
        #Break if :
        # 1 -> dull response
        # 2 -> response is less than MIN_LENGTH
        # 3 -> repetition ie curr_response in responses
        if(len(curr_response) < MIN_COUNT):# or (curr_response in dull_responses) or (curr_response in responses)):
            break
            
            
        #We can add the response to responses list
        #curr_response = torch.LongTensor(curr_response).view(-1, 1)
        #transform curr_response size
        #target = torch.zeros(960, 1)
        #target[:15, :] = curr_response
        #curr_response = target
        #print(curr_response.size())
        #curr_response = torch.reshape(curr_response, (15, 64))
        #print(curr_response.size())
        #curr_response = curr_response.to(device)
        #responses.append(curr_response) 
        
        
        #Ease of answering
        r1 = ease_of_answering(ep_input, lengths, dull_responses, mask, max_target_len, forward_encoder, forward_decoder, batch_size, teacher_forcing_ratio)
        
        #Information flow
        r2 = information_flow(responses)
        
        #Semantic coherence
        r3 = semantic_coherence(ep_input, lengths, target_var, mask, max_target_len, forward_encoder, forward_decoder, backward_encoder, backward_decoder, batch_size, teacher_forcing_ratio)
        
        #Final reward as a weighted sum of rewards
        r = l1*r1 + l2*r2 + l3*r3
        
        #Add the current reward to the list
        ep_rewards.append(r.detach().cpu().numpy())
        
        #We can add the response to responses list
        curr_response, lengths = convert_response(curr_response, batch_size)
        curr_response = curr_response.to(device)
        responses.append(curr_response)
        
        #Next input is the current response
        ep_input = curr_response
        #Next target -> dummy
        ep_target = torch.zeros(MAX_LENGTH,batch_size,dtype=torch.int64)
        #ep_target = torch.LongTensor(torch.LongTensor([0] * MAX_LENGTH)).view(-1, 1)
        ep_target = ep_target.to(device)
        
        #Turn off the teacher forcing  after first iteration -> dummy target
        teacher_forcing_ratio = 0
        ep_num +=1
        
    #Take the mean of the episodic rewards
    return np.mean(ep_rewards) if len(ep_rewards) > 0 else 0 


def training_rl_loop(model_name, voc, pairs, batch_size, forward_encoder, forward_encoder_optimizer, forward_decoder, forward_decoder_optimizer, backward_encoder, backward_encoder_optimizer, backward_decoder, backward_decoder_optimizer,teacher_forcing_ratio,n_iteration, print_every, save_every, save_dir):

    dull_responses = ["i do not know what you are talking about.", "i do not know.", "you do not know.", "you know what i mean.", "i know what you mean.", "you know what i am saying.", "you do not know anything."]

    # Load batches for each iteration
    training_batches = [batch_2_train_data(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]
    
    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    
    
    #Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        print("Iteration", iteration)
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        ##MODIFS HERE
        # Zero gradients the optimizer
        forward_encoder_optimizer.zero_grad()
        forward_decoder_optimizer.zero_grad()
        
        backward_encoder_optimizer.zero_grad()
        backward_decoder_optimizer.zero_grad()
        
        #Forward
        forward_loss, forward_len, _ = rl(input_variable, lengths, target_variable, mask, max_target_len, forward_encoder, forward_decoder, batch_size, teacher_forcing_ratio)
        
        #Calculate reward
        reward = calculate_rewards(input_variable, lengths, target_variable, mask, max_target_len, forward_encoder, forward_decoder, backward_encoder, backward_decoder, batch_size, teacher_forcing_ratio)
        
        #Update forward seq2seq with loss scaled by reward
        loss = forward_loss * reward
        
        loss.backward()
        forward_encoder_optimizer.step()
        forward_decoder_optimizer.step()

        # Run a training iteration with batch
        print_loss += loss / forward_len
        
        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0
            
        #SAVE CHECKPOINT TO DO
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name)#, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


if __name__ == "__main__":
    #forward_encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    #forward_decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    #forward_encoder = forward_encoder.to(device)
    #forward_decoder = forward_decoder.to(device)

    # device choice
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    corpus_name = "train"
    corpus = os.path.join("data", corpus_name)
    datafile = os.path.join(corpus, "formatted_dialogues_train.txt")

    # Load/Assemble voc and pairs
    save_dir = os.path.join("data", "save")
    voc, pairs = load_prepare_data(corpus, corpus_name, datafile, save_dir)
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)
    pairs = trim_rare_words(voc, pairs, min_count=3)

    # Example for validation
    small_batch_size = 5
    batches = batch_2_train_data(
        voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)

    # Configure models
    model_name = 'cb_model'
    attn_model = 'dot'
    # attn_model = 'general'
    # attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = None
    checkpoint_iter = 10000  # 4000
    # loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                            '{}_checkpoint.tar'.format(checkpoint_iter))
    # print(loadFilename)

    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        #checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 1000  # 4000
    print_every = 1
    save_every = 1000

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # If you have cuda, configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Run training iterations
    print("Starting Training!")

    forward_encoder = encoder
    forward_decoder = decoder
    forward_encoder = forward_encoder.to(device)
    forward_decoder = forward_decoder.to(device)

    backward_encoder = EncoderRNN(
        hidden_size, embedding, encoder_n_layers, dropout)
    backward_decoder = LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    backward_encoder = backward_encoder.to(device)
    backward_decoder = backward_decoder.to(device)


    #Configure RL model

    model_name='RL_model_seq'
    n_iteration = 10000
    print_every=100
    save_every=500
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    teacher_forcing_ratio = 0.5

    # Ensure dropout layers are in train mode
    forward_encoder.train()
    forward_decoder.train()

    backward_encoder.train()
    backward_decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    forward_encoder_optimizer = optim.Adam(forward_encoder.parameters(), lr=learning_rate)
    forward_decoder_optimizer = optim.Adam(forward_decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    backward_encoder_optimizer = optim.Adam(backward_encoder.parameters(), lr=learning_rate)
    backward_decoder_optimizer = optim.Adam(backward_decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # If you have cuda, configure cuda to call
    for state in forward_encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in forward_decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
                
    for state in backward_encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in backward_decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
                
    # Run training iterations
    print("Starting Training!")
    training_rl_loop(model_name, voc, pairs, batch_size, forward_encoder, forward_encoder_optimizer, forward_decoder, forward_decoder_optimizer, backward_encoder, backward_encoder_optimizer, backward_decoder, backward_decoder_optimizer,teacher_forcing_ratio,n_iteration, print_every, save_every, save_dir)
