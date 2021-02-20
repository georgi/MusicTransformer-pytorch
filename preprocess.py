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
from utils import find_files_by_extensions
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor


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
        self.num_reserved_ids = 5
        self.vocab_size = self._encoding.num_classes + self.num_reserved_ids + 1
        self.token_pad = 0
        self.token_sos = 1
        self.token_eos = 2
        self.token_bar = 3
        self.token_beat = 4

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

        event_ids = [self.token_sos, self.token_bar]
        current_step = 0
        ts = ns.time_signatures[0]
        steps_per_beat = ts.numerator
        steps_per_bar = ts.numerator * ts.denominator

        def emit_metric_events():
            if current_step % steps_per_bar == 0:
                event_ids.append(midi_encoder.token_bar)
            elif current_step % steps_per_beat == 0:
                event_ids.append(midi_encoder.token_beat)

        for event in performance:
            if event.event_type == note_seq.PerformanceEvent.TIME_SHIFT:
                for _ in range(event.event_value):
                    current_step += 1
                    emit_metric_events() 
            id = midi_encoder._encoding.encode_event(event) + \
                midi_encoder.num_reserved_ids
            if id > 0:
                event_ids.append(id)

        assert(max(event_ids) < self.vocab_size)
        assert(min(event_ids) >= 0)

        return event_ids + [self.token_eos]


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



def load_sequence(path, min_pitch, max_pitch):
    midi = pretty_midi.PrettyMIDI(path)
    for i, inst in enumerate(midi.instruments):
        if inst.is_drum:
            midi.instruments.remove(inst)
    ns = midi_to_note_sequence(midi)
    ns = apply_sustain_control_changes(ns)
    ns, _ = transpose_note_sequence(ns, 0, min_pitch, max_pitch)
    del ns.control_changes[:]
    return ns


def convert_midi_to_proto(path, dest_dir, min_pitch, max_pitch):
    ns = load_sequence(path, min_pitch, max_pitch)
    out_file = os.path.join(dest_dir, os.path.basename(path)) + '.pb'
    with open(out_file, 'wb') as f:
        f.write(ns.SerializeToString())
        

def convert_midi_folder(midi_root, data_dir, min_pitch, max_pitch):
    files = list(find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(data_dir, exist_ok=True)

    def convert(path):
        try:
            convert_midi_to_proto(path, data_dir, min_pitch, max_pitch)
        except Exception as e:
            print(e)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(convert, f) for f in files]
        for future in tqdm(futures):
            future.result()
        


if __name__ == '__main__':
    midi_root = sys.argv[1]
    data_dir = sys.argv[2]  
    convert_midi_folder(midi_root, data_dir)
