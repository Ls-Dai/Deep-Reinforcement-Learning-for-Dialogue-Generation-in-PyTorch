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


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "dailydialog"
corpus = os.path.join("data", corpus_name)


def print_lines(file, n=10):
    with open(file, 'r', encoding="utf-8") as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

# Splits each line of the file into a dictionary of fields


def load_lines(file_name):
    conversations = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split("__eou__")
            conversations.append(values)
    return conversations


# Extracts pairs of sentences from conversations
def extract_sentence_pairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        # We ignore the last line (no answer for it)
        for i in range(len(conversation) - 1):
            input_line = conversation[i].strip()
            target_line = conversation[i+1].strip()
            # Filter wrong samples (if one of the lists is empty)
            if input_line and target_line:
                qa_pairs.append([input_line, target_line])
    return qa_pairs


def create_formatted_dataset(source, target, type=None):
    # Define path to new file
    datafile = target

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    print("\nLoading conversations...")
    conversations = load_lines(source)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter,
                            lineterminator='\n')
        for pair in extract_sentence_pairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from file:")
    print_lines(datafile)


if __name__ == "__main__":
    create_formatted_dataset(
        source="data/train/dialogues_train.txt",
        target="data/train/formatted_dialogues_train.txt",
    )

    create_formatted_dataset(
        source="data/validation/dialogues_validation.txt",
        target="data/validation/formatted_dialogues_validation.txt",
    )

    create_formatted_dataset(
        source="data/test/dialogues_test.txt",
        target="data/test/formatted_dialogues_test.txt",
    )
