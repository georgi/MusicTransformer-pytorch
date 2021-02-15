import random
import utils
import numpy as np
import torch
from random import randrange, gauss
import note_seq
from tqdm.notebook import tqdm
from note_seq.sequences_lib import (
    stretch_note_sequence, 
    transpose_note_sequence,
    NegativeTimeError
)
        
def train_test_split(dataset, split=0.90):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


def load_seq_files(folder):
    res = []
    files = tqdm(list(utils.find_files_by_extensions(folder, ['.pb'])))
    for fname in files:
        with open(fname, 'rb') as f:
            ns = note_seq.NoteSequence()
            ns.ParseFromString(f.read())
            res.append(ns)
    return res


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, seq_length, midi_encoder, time_augment, transpose_augment):
        self.sequences = sequences
        self.seq_length = seq_length
        self.midi_encoder = midi_encoder
        self.time_augment = time_augment
        self.transpose_augment = transpose_augment
        
    def __len__(self):
        return len(self.sequences)

    # def batch(self, batch_size, length, mode='train'):
    #     batch_data = [
    #         self._get_seq(seq, length, mode)
    #         for seq in random.sample(self.sequences[mode], k=batch_size)
    #     ]
    #     return np.array(batch_data)  # batch_size, seq_len
    
    def augment(self, ns):
        if self.transpose_augment > 0:
            transpose = randrange(-self.transpose_augment, self.transpose_augment)
            ns = transpose_note_sequence(ns, transpose)[0]
        if self.time_augment > 0:
            try:
                stretch_factor = gauss(1.0, self.time_augment)
                ns = stretch_note_sequence(ns, stretch_factor)
            except NegativeTimeError:
                pass
        # velocity_factor = gauss(1.0, 0.2)
        # for note in ns.notes:
        #     note.velocity = max(1, min(127, int(note.velocity * velocity_factor)))
        return ns

    def encode(self, ns):
        return self.midi_encoder.encode_note_sequence(ns)

    def __getitem__(self, idx):
        return self._get_seq(self.sequences[idx])

    def _get_seq(self, ns):
        data = np.array(self.encode(self.augment(ns)))
            
        if len(data) > self.seq_length:
            start = random.randrange(0, len(data) - self.seq_length)
            data = data[start:start + self.seq_length]
        elif len(data) < self.seq_length:
            data = np.append(data, self.midi_encoder.token_eos)
            while len(data) < self.seq_length:
                data = np.append(data, self.midi_encoder.token_pad)

        assert(len(data) == self.seq_length)

        return data