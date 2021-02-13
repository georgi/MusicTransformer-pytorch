import random
import numpy as np
from random import randrange, gauss
import note_seq
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


def load_seq_files(files):
    res = []
    for fname in files:
        with open(fname, 'rb') as f:
            ns = note_seq.NoteSequence()
            ns.ParseFromString(f.read())
            res.append(ns)
    return res


class Data:
    def __init__(self, sequences, midi_encoder):
        self.sequences = sequences
        self.midi_encoder = midi_encoder
        
    def __len__(self):
        return sum(len(s) for s in self.sequences.values()) 

    def batch(self, batch_size, length, mode='train'):
        batch_data = [
            self._get_seq(seq, length, mode)
            for seq in random.sample(self.sequences[mode], k=batch_size)
        ]
        return np.array(batch_data)  # batch_size, seq_len

    def slide_seq2seq_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length + 1, mode)
        assert(data.shape == (batch_size, length + 1))
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y
    
    def augment(self, ns):
        stretch_factor = gauss(1.0, 0.5)
        velocity_factor = gauss(1.0, 0.2)
        transpose = randrange(-5, 7)
        ns = stretch_note_sequence(ns, stretch_factor)
        for note in ns.notes:
            note.velocity = max(1, min(127, int(note.velocity * velocity_factor)))
        return transpose_note_sequence(ns, transpose, in_place=True)[0]

    def _get_seq(self, ns, max_length, mode):
        if mode == 'train':
            try:
                data = self.midi_encoder.encode_note_sequence(self.augment(ns))
            except NegativeTimeError:
                data = self.midi_encoder.encode_note_sequence(ns)
        else:
            data = self.midi_encoder.encode_note_sequence(ns)
            
        if len(data) > max_length:
            start = random.randrange(0, len(data) - max_length)
            data = data[start:start + max_length]
        elif len(data) < max_length:
            data = np.append(data, self.midi_encoder.token_eos)
            while len(data) < max_length:
                data = np.append(data, self.midi_encoder.token_pad)

        assert(len(data) == max_length)

        return data