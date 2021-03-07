import random
import torch
import os
from random import randrange, uniform
from torch.utils.data import DataLoader
from note_seq.sequences_lib import (
    stretch_note_sequence,
    transpose_note_sequence,
    NegativeTimeError
)
from note_seq.protobuf import music_pb2
from utils import find_files_by_extensions
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def process_midi(seq, max_seq, token_pad):
    if len(seq) <= max_seq:
        x = torch.full((max_seq, ), token_pad, dtype=torch.long)
        tgt = torch.full((max_seq, ), token_pad, dtype=torch.long)
        x[:len(seq)] = seq
        tgt[:len(seq) - 1] = seq[1:]
    else:
        try:
            start = random.randint(0, len(seq) - max_seq - 1)
        except ValueError:
            start = 0
        end = start + max_seq + 1
        data = seq[start:end]
        x = data[:max_seq]
        tgt = data[1:max_seq + 1]
    return x, tgt


def train_test_split(dataset, split=0.90):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


def save_sequence(ns, path):
    with open(path, 'wb') as f:
        f.write(ns.SerializeToString())


def load_sequence(fname):
    with open(fname, 'rb') as f:
        ns = music_pb2.NoteSequence()
        ns.ParseFromString(f.read())
        return ns


def convert_midi_to_proto(midi_encoder, src, dest_dir):
    res = []
    for i, ns in enumerate(midi_encoder.load_midi(src)):
        fname = os.path.join(dest_dir, os.path.basename(src) + f'-{i}.pb')
        save_sequence(ns, fname)
        res.append(fname)
    return res


def convert_midi_to_proto_folder(midi_encoder, src_dir, dest_dir, max_workers=10):
    files = list(find_files_by_extensions(src_dir, ['.mid', '.midi']))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        res = []
        futures = [
            executor.submit(
                convert_midi_to_proto, midi_encoder, f, dest_dir)
            for f in files
        ]
        for future in tqdm(futures):
            res.extend(future.result())


def data_loaders(
    midi_encoder,
    data_dir,
    batch_size,
    max_seq,
    time_augment,
    transpose_augment,
    num_workers=8
):
    data_files = list(find_files_by_extensions(data_dir, ['.pb']))
    train_files, valid_files = train_test_split(data_files)

    train_data = SequenceDataset(
        sequences=train_files,
        seq_length=max_seq,
        midi_encoder=midi_encoder,
        time_augment=time_augment,
        transpose_augment=transpose_augment
    )
    valid_data = SequenceDataset(
        sequences=valid_files,
        seq_length=max_seq,
        midi_encoder=midi_encoder,
        time_augment=0,
        transpose_augment=0
    )

    train_loader = DataLoader(train_data, batch_size, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size, num_workers=num_workers)

    return train_loader, valid_loader


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequences,
        seq_length,
        midi_encoder,
        time_augment,
        transpose_augment
    ):
        self.sequences = sequences
        self.seq_length = seq_length
        self.midi_encoder = midi_encoder
        self.time_augment = time_augment
        self.transpose_augment = transpose_augment

    def __len__(self):
        return len(self.sequences)

    def augment(self, ns):
        if self.transpose_augment > 0:
            transpose = randrange(-self.transpose_augment,
                                  self.transpose_augment)
            ns = transpose_note_sequence(ns, transpose)[0]
        if self.time_augment > 0:
            try:
                stretch_factor = uniform(
                    1.0 - self.time_augment,
                    1.0 + self.time_augment
                )
                ns = stretch_note_sequence(ns, stretch_factor)
            except NegativeTimeError:
                pass
        return ns

    def encode(self, ns):
        return self.midi_encoder.encode_note_sequence(ns)

    def __getitem__(self, idx):
        return self._get_seq(self.sequences[idx])

    def _get_seq(self, file):
        ns = load_sequence(file)
        data = torch.tensor(self.encode(self.augment(ns)))
        data = process_midi(data, self.seq_length, 0)
        return data
