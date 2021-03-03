import os
from note_seq.midi_io import (
    midi_to_note_sequence,
    note_sequence_to_midi_file
)
from collections import defaultdict
from note_seq.protobuf import music_pb2
import pretty_midi
from note_seq.sequences_lib import (
    is_quantized_sequence,
    apply_sustain_control_changes,
    split_note_sequence_on_silence,
    split_note_sequence_on_time_changes,
    quantize_note_sequence,
    quantize_note_sequence_absolute,
    steps_per_bar_in_quantized_sequence
)
from note_seq import (
    PerformanceOneHotEncoding,
    Performance,
)


class Event:
    PAUSE = 'pause'
    NOTE_ON = 'note_on'
    NOTE_OFF = 'note_off'
    TIME_SHIFT = 'time_shift'

    def __init__(self, event_type, event_value=0):
        self.event_type = event_type
        self.event_value = event_value

        assert(event_type in (Event.NOTE_ON, Event.NOTE_OFF, Event.PAUSE,
                              Event.TIME_SHIFT))

    def __repr__(self):
        return f"<Event {self.event_type} {self.event_value}>"


class Encoding:
    MIN_NOTE = 0
    MAX_NOTE = 127
    MAX_TIMESHIFT = 64

    def __init__(self):
        self._event_ranges = [
            (Event.PAUSE, 0, 0),
            (Event.NOTE_ON, Encoding.MIN_NOTE, Encoding.MAX_NOTE),
            (Event.NOTE_OFF, Encoding.MIN_NOTE, Encoding.MAX_NOTE),
        ]
        # self._event_ranges.append((Event.TIME_SHIFT, 1, Event.MAX_TIMESHIFT))

    @ property
    def num_classes(self):
        return sum(max_value - min_value + 1
                   for _, min_value, max_value in self._event_ranges)

    def encode_event(self, event):
        offset = 0
        for event_type,  min_value, max_value in self._event_ranges:
            if event.event_type == event_type:
                return offset + event.event_value - min_value
            offset += max_value - min_value + 1

        raise ValueError('Unknown event type: %s' % event.event_type)

    def decode_event(self, index):
        offset = 0
        for event_type, min_value, max_value in self._event_ranges:
            if offset <= index <= offset + max_value - min_value:
                return Event(
                    event_type=event_type,
                    event_value=min_value + index - offset,
                )
            offset += max_value - min_value + 1

        raise ValueError('Unknown event index: %s' % index)


class MIDIEncoder:

    def split_and_quantize(self, ns):
        return [
            quantize_note_sequence(i, self.steps_per_quarter)
            for i in split_note_sequence_on_time_changes(ns)
            if len(i.notes) > 16
        ]


class MIDIMetricEncoder(MIDIEncoder):
    def __init__(
        self,
        encoding=None,
        steps_per_quarter=4,
        steps_per_bar=16,
        num_instruments=4,
    ):
        if encoding is None:
            encoding = Encoding()
        self.steps_per_quarter = steps_per_quarter
        self.steps_per_bar = steps_per_bar
        self.num_instruments = num_instruments
        self.encoding = encoding
        self.vocab_size = self.encoding.num_classes

    def get_instruments(self, ns):
        pitch_by_instr = defaultdict(list)
        for note in ns.notes:
            if not note.is_drum:
                pitch_by_instr[note.instrument].append(note.pitch)

        max_pitch = [
            (max(v), i)
            for i, v in pitch_by_instr.items()
            if len(v) > 10
        ]
        min_pitch = [
            (min(v), i)
            for i, v in pitch_by_instr.items()
            if len(v) > 10
        ]

        lead_inst = [v for k, v in sorted(max_pitch, reverse=True)]
        bass_inst = [v for k, v in sorted(min_pitch, reverse=False)]

        instruments = bass_inst[:1] + lead_inst

        if len(instruments) > 0:
            return instruments[:self.num_instruments]
        else:
            return [0] * self.num_instruments

    def encode_note_sequence_to_events(self, ns, num_bars=16):
        assert(is_quantized_sequence(ns))
        instruments = self.get_instruments(ns)
        notes = defaultdict(dict)

        # Only one note per instrument
        for note in ns.notes:
            if note.instrument in instruments:
                instrument = instruments.index(note.instrument)
                notes[instrument][note.quantized_start_step] = (
                    note.pitch, Event.NOTE_ON)
                notes[instrument][note.quantized_end_step] = (
                    note.pitch, Event.NOTE_OFF)

        events = []
        for bar in range(num_bars):
            for instrument in range(self.num_instruments):
                for bar_step in range(self.steps_per_bar):
                    step = bar * self.steps_per_bar + bar_step
                    if step in notes[instrument]:
                        pitch, event_type = notes[instrument][step]
                        events.append(Event(event_type, pitch))
                    else:
                        events.append(Event(Event.PAUSE))
        return events

    def encode_events(self, events):
        return [self.encoding.encode_event(event) for event in events]

    def encode_note_sequence(self, ns, num_bars=16):
        ids = self.encode_events(
            self.encode_note_sequence_to_events(ns, num_bars))
        assert(max(ids) < self.vocab_size)
        assert(min(ids) >= 0)
        return ids

    def decode_ids(self, ids, bpm=120.0, velocity=100, max_note_steps=8):
        assert(max(ids) < self.vocab_size)
        assert(min(ids) >= 0)

        sequence = music_pb2.NoteSequence()
        sequence.ticks_per_quarter = 220
        seconds_per_step = 60.0 / (self.steps_per_quarter * bpm)
        step = 0
        num_bars = len(ids) // self.steps_per_bar
        active_note = [None] * self.num_instruments

        def add_note(instrument, start_step, end_step, pitch):
            note = sequence.notes.add()
            note.start_time = start_step * seconds_per_step
            note.end_time = end_step * seconds_per_step
            note.pitch = pitch
            note.instrument = instrument
            note.program = 0
            note.velocity = velocity
            note.is_drum = False
            if note.end_time > sequence.total_time:
                sequence.total_time = note.end_time

        def end_active_note(instrument, step):
            start_step, pitch = active_note[instrument]
            end_step = min(step, start_step + max_note_steps)
            add_note(instrument, start_step, end_step, pitch)
            active_note[instrument] = None

        for bar in range(num_bars):
            bar_offset = bar * self.steps_per_bar * self.num_instruments
            for instrument in range(self.num_instruments):
                inst_offset = bar_offset + instrument * self.steps_per_bar
                for bar_step in range(self.steps_per_bar):
                    index = inst_offset + bar_step
                    step = bar * self.steps_per_bar + bar_step
                    if index >= len(ids):
                        break
                    event = self.encoding.decode_event(ids[index])
                    if event.event_type == Event.NOTE_ON:
                        if active_note[instrument]:
                            end_active_note(instrument, step)
                        active_note[instrument] = (step, event.event_value)
                    elif event.event_type == Event.NOTE_OFF:
                        if active_note[instrument]:
                            end_active_note(instrument, step)

        for instrument in range(self.num_instruments):
            if active_note[instrument]:
                end_active_note(instrument, num_bars *
                                self.steps_per_bar * self.num_instruments)

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
    midi_encoder = MIDIMetricEncoder(
        Encoding(), steps_per_quarter=2, steps_per_bar=8, num_instruments=2)
    midi_dir = "/Users/mmg/uni/midi/final_fantasy/"
    for f in os.listdir(midi_dir):
        # for f in ['ff1matya.mid']:
        midi_file = os.path.join(midi_dir, f)
        ns = midi_encoder.load_midi(midi_file)
        if len(ns) > 0:
            steps_per_bar = steps_per_bar_in_quantized_sequence(ns[0])
            if steps_per_bar == 8:
                print(f)
                ids = midi_encoder.encode_note_sequence(ns[0], 8)
                out = midi_encoder.decode_ids(ids)
                note_sequence_to_midi_file(out, os.path.join('out', f))
