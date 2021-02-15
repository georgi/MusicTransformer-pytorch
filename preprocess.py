import utils
import os
import sys
import pickle
import note_seq
from note_seq.midi_io import midi_to_note_sequence
import pretty_midi
from note_seq.sequences_lib import (
    quantize_note_sequence_absolute,
    stretch_note_sequence,
    transpose_note_sequence,
    apply_sustain_control_changes,
)
from note_seq import PerformanceOneHotEncoding

class MidiEncoder:
   
    def __init__(self, num_velocity_bins, min_pitch, max_pitch, steps_per_quarter=None, steps_per_second=None):
        self._steps_per_second = steps_per_second
        self._steps_per_quarter = steps_per_quarter
        self._num_velocity_bins = num_velocity_bins
        self._encoding = PerformanceOneHotEncoding(
            num_velocity_bins=num_velocity_bins,
            min_pitch=min_pitch,
            max_pitch=max_pitch
        )
        self.num_reserved_ids = 3
        self.vocab_size = self._encoding.num_classes + self.num_reserved_ids + 1
        self.token_pad = 0
        self.token_sos = 1
        self.token_eos = 2

    def encode_note_sequence(self, ns):
        if self._steps_per_quarter:
            performance = note_seq.MetricPerformance(
                note_seq.quantize_note_sequence(ns, self._steps_per_quarter),
                num_velocity_bins=self._num_velocity_bins,
            )
        else:
            performance = note_seq.Performance(
                note_seq.quantize_note_sequence_absolute(
                    ns, 
                    self._steps_per_second),
                    num_velocity_bins=self._num_velocity_bins
                )

        event_ids = [self._encoding.encode_event(event) + 
                     self.num_reserved_ids
                     for event in performance]

        event_ids = [i for i in event_ids if i > 0]

        assert(max(event_ids) < self.vocab_size)
        assert(min(event_ids) >= 0)

        return [self.token_sos] + event_ids


    def decode_ids(self, ids):
        assert(max(ids) < self.vocab_size)
        assert(min(ids) >= 0)

        if self._steps_per_quarter:
            performance = note_seq.MetricPerformance(
                steps_per_quarter=self._steps_per_quarter,
                num_velocity_bins=self._num_velocity_bins
            )
        else:
            performance = note_seq.Performance(
                steps_per_second=self._steps_per_second,
                num_velocity_bins=self._num_velocity_bins
            )

        for i in ids:
            if i >= self.num_reserved_ids:
                performance.append(self._encoding.decode_event(i - self.num_reserved_ids))

        return performance.to_sequence()


def convert_midi_to_proto(path, dest_dir):
    midi = pretty_midi.PrettyMIDI(path)
    for i, inst in enumerate(midi.instruments):
        num_distinct_pitches = sum([i > 5 for i in inst.get_pitch_class_histogram()])
        if inst.is_drum or num_distinct_pitches < 5 or len(inst.notes) < 30:
            midi.instruments.remove(inst)
    ns = midi_to_note_sequence(midi)
    ns = apply_sustain_control_changes(ns)
    del ns.control_changes[:]
    out_file = os.path.join(dest_dir, os.path.basename(path)) + '.pb'
    with open(out_file, 'wb') as f:
        f.write(ns.SerializeToString())
        



if __name__ == '__main__':
    midi_root = sys.argv[1]
    data_dir = sys.argv[2]  
    midi_paths = list(find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(data_dir, exist_ok=True)

    for file in os.listdir(data_dir):
        os.unlink(os.path.join(data_dir, file))

    # Convert all MIDI files into internal format
    for path in midi_paths:
        try:
            preprocess.convert_midi_to_proto(path, data_dir)
        except Exception as e:
            print(e)
