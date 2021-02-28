import os
from note_seq.midi_io import (
    midi_to_note_sequence,
    note_sequence_to_midi_file
)
from collections import defaultdict
import numpy as np
from note_seq.protobuf import music_pb2
import pretty_midi
from note_seq.sequences_lib import (
    is_quantized_sequence,
    apply_sustain_control_changes,
    split_note_sequence_on_silence,
    split_note_sequence_on_time_changes,
    quantize_note_sequence,
    quantize_note_sequence_absolute
)
from note_seq import (
    PerformanceOneHotEncoding,
    Performance,
)
from utils import find_files_by_extensions
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor


class Event:
    NOTE_ON = 'note_on'
    NOTE_OFF = 'note_off'
    TIME_SHIFT = 'time_shift'
    PROGRAM = 'program'

    def __init__(self, event_type, event_value=0, instrument=0):
        self.event_type = event_type
        self.instrument = instrument
        self.event_value = event_value

        if event_type == Event.TIME_SHIFT:
            assert(event_value > 0 and event_value <= Encoding.MAX_SHIFT)

        if event_type == Event.PROGRAM:
            assert(event_value >= 0 and event_value <= 127)

        assert(event_type in (Event.NOTE_ON, Event.NOTE_OFF,
                              Event.TIME_SHIFT, Event.PROGRAM))
        assert(instrument >= 0 and instrument < Encoding.MAX_INSTRUMENTS)

    def __repr__(self):
        return f"<Event {self.event_type} {self.event_value} {self.instrument}>"


class Encoding:
    MIN_NOTE = 0
    MAX_NOTE = 127
    MAX_SHIFT = 64
    MAX_INSTRUMENTS = 4

    def __init__(self):
        self._event_ranges = [
            (Event.TIME_SHIFT, 0, 1, Encoding.MAX_SHIFT),
        ]
        for i in range(Encoding.MAX_INSTRUMENTS):
            self._event_ranges.extend([
                (Event.PROGRAM, i, 0, 127),
                (Event.NOTE_ON, i, Encoding.MIN_NOTE, Encoding.MAX_NOTE),
                (Event.NOTE_OFF, i, Encoding.MIN_NOTE, Encoding.MAX_NOTE),
            ])

    @ property
    def num_classes(self):
        return sum(max_value - min_value + 1
                   for _, _, min_value, max_value in self._event_ranges)

    def encode_event(self, event):
        offset = 0
        for event_type, inst, min_value, max_value in self._event_ranges:
            if event.event_type == event_type and event.instrument == inst:
                return offset + event.event_value - min_value
            offset += max_value - min_value + 1

        raise ValueError('Unknown event type: %s' % event.event_type)

    def decode_event(self, index):
        offset = 0
        for event_type, inst, min_value, max_value in self._event_ranges:
            if offset <= index <= offset + max_value - min_value:
                return Event(
                    event_type=event_type,
                    event_value=min_value + index - offset,
                    instrument=inst
                )
            offset += max_value - min_value + 1

        raise ValueError('Unknown event index: %s' % index)


class MIDIEncoder:

    def remove_duplicate_notes(self, ns):
        notes = set()
        dupes = []
        for note in ns.notes:
            key = f"{note.quantized_start_step}_{note.pitch}"
            if key in notes:
                dupes.append(note)
            else:
                notes.add(key)
        for note in dupes:
            ns.notes.remove(note)
        return ns

    def quantize(self, ns):
        return self.remove_duplicate_notes(
            quantize_note_sequence(ns, self.steps_per_quarter)
        )

    def split_and_quantize(self, ns):
        if self.steps_per_quarter:
            return [
                self.quantize(i)
                for i in split_note_sequence_on_time_changes(ns)
                if len(i.notes) > 5
            ]
        else:
            return split_note_sequence_on_silence(ns)

    def load_midi_folder(self, folder, max_workers=20):
        files = list(find_files_by_extensions(folder, ['.mid', '.midi']))
        res = []
        if max_workers == 0:
            for f in tqdm(files):
                res.extend(self.load_midi(f))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.load_midi, f) for f in files]
                for future in tqdm(futures):
                    res.extend(future.result())
        return res


class MIDIMetricEncoder(MIDIEncoder):
    def __init__(
        self,
        encoding=None,
        steps_per_quarter=4,
    ):
        if encoding is None:
            encoding = Encoding()
        self.steps_per_quarter = steps_per_quarter
        self.encoding = encoding
        self.num_reserved_ids = 3
        self.vocab_size = self.encoding.num_classes + self.num_reserved_ids
        self.token_pad = 0
        self.token_sos = 1
        self.token_eos = 2

    def encode_note_sequence(self, ns):
        assert(is_quantized_sequence(ns))

        programs = {}

        pitch_by_instr = defaultdict(list)
        for note in ns.notes:
            if not note.is_drum:
                pitch_by_instr[note.instrument].append(note.pitch)

        mean_pitch = [
            (np.median(v), i)
            for i, v in pitch_by_instr.items()
        ]

        instruments = [v for k, v in sorted(mean_pitch, reverse=True)]

        if len(instruments) > 0:
            instruments = instruments[-1:] + instruments[:-1]

        sorted_notes = sorted(ns.notes, key=lambda note: (
            note.start_time, note.pitch))

        # Sort all note start and end events.
        onsets = [(note.quantized_start_step, idx, False)
                  for idx, note in enumerate(sorted_notes)]
        offsets = [(note.quantized_end_step, idx, True)
                   for idx, note in enumerate(sorted_notes)]
        note_events = sorted(onsets + offsets)

        current_step = 0
        events = []
        for step, idx, is_offset in note_events:
            if step > current_step:
                while step > current_step + Encoding.MAX_SHIFT:
                    events.append(Event(Event.TIME_SHIFT, Encoding.MAX_SHIFT))
                    current_step += Encoding.MAX_SHIFT
                events.append(
                    Event(Event.TIME_SHIFT, int(step - current_step)))
                current_step = step
            event_type = Event.NOTE_OFF if is_offset else Event.NOTE_ON
            note = sorted_notes[idx]
            if note.is_drum:
                events.append(Event(event_type, note.pitch, 0))
            else:
                instr_index = instruments.index(note.instrument) + 1
                if instr_index < Encoding.MAX_INSTRUMENTS:
                    if instr_index not in programs and note.program:
                        programs[instr_index] = note.program
                        events.append(
                            Event(Event.PROGRAM, note.program, instr_index))
                    events.append(Event(event_type, note.pitch, instr_index))

        return [self.token_sos] + [
            self.encoding.encode_event(event) + self.num_reserved_ids
            for event in events
        ] + [self.token_eos]

    def decode_ids(self, ids):
        assert(max(ids) < self.vocab_size)
        assert(min(ids) >= 0)

        sequence = music_pb2.NoteSequence()
        sequence.ticks_per_quarter = 220
        qpm = 120.0
        seconds_per_step = 60.0 / (self.steps_per_quarter * qpm)
        step = 0
        velocity = 100
        max_note_duration = 10
        programs = {}

        # Map pitch to list because one pitch may be active multiple times.
        pitch_start_steps = defaultdict(list)
        for i, idx in enumerate(ids):
            if idx < self.num_reserved_ids:
                continue
            event = self.encoding.decode_event(idx - self.num_reserved_ids)
            key = (event.instrument, event.event_value)
            if event.event_type == Event.NOTE_ON:
                pitch_start_steps[key].append(step)
            elif event.event_type == Event.NOTE_OFF:
                if pitch_start_steps[key]:
                    # Create a note for the pitch that is now ending.
                    pitch_start_step = pitch_start_steps[key][0]
                    pitch_start_steps[key] = (
                        pitch_start_steps[event.event_value][1:]
                    )
                    if step == pitch_start_step:
                        continue
                    note = sequence.notes.add()
                    note.start_time = pitch_start_step * seconds_per_step
                    note.end_time = step * seconds_per_step
                    if note.end_time - note.start_time > max_note_duration:
                        note.end_time = note.start_time + max_note_duration
                    note.pitch = event.event_value
                    note.velocity = velocity
                    note.instrument = event.instrument
                    if event.instrument in programs:
                        note.program = programs[event.instrument]
                    note.is_drum = event.instrument == 0
                    if note.end_time > sequence.total_time:
                        sequence.total_time = note.end_time
            elif event.event_type == Event.TIME_SHIFT:
                step += event.event_value
            elif event.event_type == Event.PROGRAM:
                programs[event.instrument] = event.event_value
        return sequence

    def load_midi(self, path):
        try:
            midi = pretty_midi.PrettyMIDI(path)
        except Exception as e:
            print("Failed to load MIDI file", path, e)
            return []
        ns = midi_to_note_sequence(midi)
        del ns.control_changes[:]
        return self.split_and_quantize(ns)


class MIDIPerformanceEncoder(MIDIEncoder):

    def __init__(
        self,
        num_velocity_bins,
        steps_per_second,
    ):
        super(MIDIPerformanceEncoder, self).__init__()
        self.steps_per_second = steps_per_second
        self.num_reserved_ids = 4
        self.token_pad = 0
        self.token_sos = 1
        self.token_eos = 2
        self.token_bar = 3
        self.num_velocity_bins = num_velocity_bins
        self.encoding = PerformanceOneHotEncoding(num_velocity_bins)
        self.vocab_size = self.encoding.num_classes + self.num_reserved_ids

    def encode_note_sequence(self, ns):
        performance = Performance(
            quantize_note_sequence_absolute(
                ns,
                self.steps_per_second),
            num_velocity_bins=self.num_velocity_bins
        )

        event_ids = [self.token_sos]

        for event in performance:
            id = self.encoding.encode_event(event) + self.num_reserved_ids
            if id > 0:
                event_ids.append(id)

        assert(max(event_ids) < self.vocab_size)
        assert(min(event_ids) >= 0)

        return event_ids + [self.token_eos]

    def decode_ids(self, ids):
        assert(max(ids) < self.vocab_size)
        assert(min(ids) >= 0)

        performance = Performance(
            steps_per_second=self.steps_per_second,
            num_velocity_bins=self.num_velocity_bins
        )

        for i in ids:
            if i >= self.num_reserved_ids:
                performance.append(
                    self.encoding.decode_event(i - self.num_reserved_ids)
                )

        return performance.to_sequence()

    def load_midi(self, path):
        try:
            midi = pretty_midi.PrettyMIDI(path)
        except Exception as e:
            print("Failed to load MIDI file", path, e)
            return []
        ns = midi_to_note_sequence(midi)
        ns = apply_sustain_control_changes(ns)
        # after applying sustain, we don't need control changes anymore
        del ns.control_changes[:]
        return split_note_sequence_on_silence(ns)


if __name__ == '__main__':
    midi_encoder = MIDIMetricEncoder(Encoding(), steps_per_quarter=4)
    midi_dir = "/Users/mmg/uni/midi/final_fantasy/"
    # midi_encoder.load_midi_folder(midi_dir, 0)
    for f in os.listdir(midi_dir):
        print(f)
        midi_file = os.path.join(midi_dir, f)
        ns = midi_encoder.load_midi(midi_file)
        if len(ns) > 0:
            ids = midi_encoder.encode_note_sequence(ns[0])
            out = midi_encoder.decode_ids(ids)
            note_sequence_to_midi_file(out, os.path.join('out', f))
