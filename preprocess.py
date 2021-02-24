import note_seq
from note_seq.midi_io import midi_to_note_sequence
import pretty_midi
from note_seq.sequences_lib import (
    transpose_note_sequence,
    apply_sustain_control_changes,
)
from note_seq import PerformanceOneHotEncoding
from utils import find_files_by_extensions
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor


class MidiEncoder:

    def __init__(
        self, 
        num_velocity_bins, 
        min_pitch=21,
        max_pitch=108, 
        steps_per_quarter=None, 
        steps_per_second=None, 
        encode_metrics=True
    ):
        self.steps_per_second = steps_per_second
        self.steps_per_quarter = steps_per_quarter
        self.num_velocity_bins = num_velocity_bins
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.encoding = PerformanceOneHotEncoding(
            num_velocity_bins=num_velocity_bins,
            min_pitch=min_pitch,
            max_pitch=max_pitch
        )
        self.encode_metrics = encode_metrics
        self.num_reserved_ids = 5
        self.vocab_size = self.encoding.num_classes + self.num_reserved_ids + 1
        self.token_pad = 0
        self.token_sos = 1
        self.token_eos = 2
        self.token_bar = 3
        self.token_beat = 4

    def encode_note_sequence(self, ns):
        if self.steps_per_quarter:
            performance = note_seq.MetricPerformance(
                note_seq.quantize_note_sequence(ns, self.steps_per_quarter),
                num_velocity_bins=self.num_velocity_bins,
            )
        else:
            performance = note_seq.Performance(
                note_seq.quantize_note_sequence_absolute(
                    ns,
                    self.steps_per_second),
                num_velocity_bins=self.num_velocity_bins
            )

        event_ids = [self.token_sos]
        current_step = 0
        ts = ns.time_signatures[0]
        steps_per_beat = ts.numerator
        steps_per_bar = ts.numerator * ts.denominator

        def emit_metric_events():
            if current_step % steps_per_bar == 0:
                event_ids.append(self.token_bar)
            elif current_step % steps_per_beat == 0:
                event_ids.append(self.token_beat)
                
        if self.encode_metrics:
            emit_metric_events()

        for event in performance:
            if event.event_type == note_seq.PerformanceEvent.TIME_SHIFT:
                for _ in range(event.event_value):
                    current_step += 1
                    if self.encode_metrics:
                        emit_metric_events()
            id = self.encoding.encode_event(event) + self.num_reserved_ids
            if id > 0:
                event_ids.append(id)

        assert(max(event_ids) < self.vocab_size)
        assert(min(event_ids) >= 0)

        return event_ids + [self.token_eos]

    def decode_ids(self, ids):
        assert(max(ids) < self.vocab_size)
        assert(min(ids) >= 0)

        if self.steps_per_quarter:
            performance = note_seq.MetricPerformance(
                steps_per_quarter=self.steps_per_quarter,
                num_velocity_bins=self.num_velocity_bins
            )
        else:
            performance = note_seq.Performance(
                steps_per_second=self.steps_per_second,
                num_velocity_bins=self.num_velocity_bins
            )

        for i in ids:
            if i >= self.num_reserved_ids:
                performance.append(self.encoding.decode_event(
                    i - self.num_reserved_ids))

        return performance.to_sequence()

    def load_midi(self, path):
        midi = pretty_midi.PrettyMIDI(path)
        for i, inst in enumerate(midi.instruments):
            if inst.is_drum:
                midi.instruments.remove(inst)
        ns = midi_to_note_sequence(midi)
        ns = apply_sustain_control_changes(ns)
        ns, _ = transpose_note_sequence(ns, 0, self.min_pitch, self.max_pitch)
        del ns.control_changes[:]
        return ns
    
    def load_midi_folder(self, folder):
        files = list(find_files_by_extensions(folder, ['.mid', '.midi']))
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.load_midi, f) for f in files]
            return [future.result() for future in tqdm(futures)]

