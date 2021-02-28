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
    transpose_note_sequence,
    apply_sustain_control_changes,
    split_note_sequence_on_silence,
    split_note_sequence_on_time_changes,
    quantize_note_sequence,
    quantize_note_sequence_absolute
)
from note_seq.chord_inference import infer_chords_for_sequence
from note_seq import (
    PerformanceOneHotEncoding,
    TriadChordOneHotEncoding,
    Performance,
    PerformanceEvent
)
from utils import find_files_by_extensions
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor


class Event:
    PAD = 'pad'
    BAR = 'bar'
    START = 'start'
    END = 'end'
    CHORD = 'chord'
    NOTE_ON = 'note_on'
    NOTE_OFF = 'note_off'
    TIME_SHIFT = 'time_shift'
    BASS = 'bass'
    MELODY = 'melody'
    OTHER = 'other'

    def __init__(self, event_type, event_value=0, instrument=None):
        self.event_type = event_type
        self.instrument = instrument
        self.event_value = event_value

        if event_type == Event.TIME_SHIFT:
            assert(event_value >= 0 and event_value < 128)

        if event_type == Event.CHORD:
            assert(event_value >= 0 and event_value <= 48)

        if instrument == Event.BASS:
            self.event_value = event_value % 12

        if instrument == Event.MELODY:
            self.event_value = (event_value - 48) % 48

        if instrument == Event.OTHER:
            self.event_value = (event_value - 36) % 48

        assert(event_type in (Event.PAD, Event.BAR, Event.START,
                              Event.END, Event.NOTE_ON, Event.NOTE_OFF,
                              Event.TIME_SHIFT))
        if instrument:
            assert(instrument in (Event.BASS, Event.MELODY, Event.OTHER))

    @ property
    def pitch(self):
        if self.instrument == Event.BASS:
            return self.event_value + 24

        if self.instrument == Event.MELODY:
            return self.event_value + 48

        if self.instrument == Event.OTHER:
            return self.event_value + 36

    def __repr__(self):
        return f"<Event {self.event_type} {self.event_value} {self.instrument}>"


class Encoding:

    def __init__(self, max_shift_steps=64):
        self._event_ranges = [
            (Event.PAD, None, 0, 1),
            (Event.START, None, 0, 1),
            (Event.END, None, 0, 1),
            (Event.BAR, None, 0, 1),
            (Event.CHORD, None, 0, 48),
            (Event.NOTE_ON, Event.BASS, 0, 12),
            (Event.NOTE_OFF, Event.BASS, 0, 12),
            (Event.NOTE_ON, Event.MELODY, 0, 48),
            (Event.NOTE_OFF, Event.MELODY, 0, 48),
            (Event.NOTE_ON, Event.OTHER, 0, 48),
            (Event.NOTE_OFF, Event.OTHER, 0, 48),
            (Event.TIME_SHIFT, None, 1, max_shift_steps)
        ]
        self._max_shift_steps = max_shift_steps

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

    def load_midi_folder(self, folder, max_workers=8):
        files = list(find_files_by_extensions(folder, ['.mid', '.midi']))
        res = []
        if max_workers == 0:
            for f in tqdm(files):
                res.extend(self.load_midi(f))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.load_midi, f) for f in files]
                for future in tqdm(futures):
                    res.extend(future.result())
        return res


class MIDISongEncoder(MIDIEncoder):

    def __init__(
        self,
        steps_per_quarter=None,
    ):
        super(MIDISongEncoder, self).__init__()
        self.steps_per_quarter = steps_per_quarter
        self.encoding = Encoding()
        self.vocab_size = self.encoding.num_classes

    def infer_chords(self, ns):
        try:
            infer_chords_for_sequence(ns)
        except Exception:
            pass
        return ns

    def encode_note_sequence(self, ns, max_shift_steps=1000):
        assert(is_quantized_sequence(ns))

        chords = {}
        for a in ns.text_annotations:
            if a.annotation_type == 0:
                chords[a.quantized_step] = a.text

        pitch_by_instr = defaultdict(list)
        for note in ns.notes:
            if not note.is_drum:
                pitch_by_instr[note.instrument].append(note.pitch)

        mean_pitch = [
            np.mean(v)
            for v in pitch_by_instr.values()
        ]

        if len(mean_pitch) > 0:
            melody_inst = list(pitch_by_instr.keys())[np.argmax(mean_pitch)]
            bass_inst = list(pitch_by_instr.keys())[np.argmin(mean_pitch)]
        else:
            melody_inst = 0
            bass_inst = 1

        sorted_notes = sorted(ns.notes, key=lambda note: (
            note.start_time, note.pitch))

        # Sort all note start and end events.
        onsets = [(note.quantized_start_step, idx, False)
                  for idx, note in enumerate(sorted_notes)]
        offsets = [(note.quantized_end_step, idx, True)
                   for idx, note in enumerate(sorted_notes)]
        note_events = sorted(onsets + offsets)

        current_step = 0
        events = [
            Event(Event.START)
        ]

        for step, idx, is_offset in note_events:
            if step > current_step:
                # Shift time forward from the current step to this event.
                while step > current_step + max_shift_steps:
                    # We need to move further than the maximum shift size.
                    events.append(Event(Event.TIME_SHIFT, max_shift_steps))
                    current_step += max_shift_steps
                events.append(
                    Event(Event.TIME_SHIFT, int(step - current_step)))
                current_step = step
            if step in chords:
                value = self.chord_encoding.encode_event(chords[step])
                events.append(Event(Event.CHORD, value))
            # Add a performance event for this note on/off.
            event_type = Event.NOTE_OFF if is_offset else Event.NOTE_ON
            note = sorted_notes[idx]

            if note.instrument == bass_inst:
                events.append(Event(event_type, note.pitch, Event.BASS))
            elif note.instrument == melody_inst:
                events.append(Event(event_type, note.pitch, Event.MELODY))
            elif not note.is_drum:
                events.append(Event(event_type, note.pitch, Event.OTHER))

        events.append(Event(Event.END))

        return [
            self.encoding.encode_event(event)
            for event in events
        ]

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

        # Map pitch to list because one pitch may be active multiple times.
        pitch_start_steps = defaultdict(list)
        for i, idx in enumerate(ids):
            event = self.encoding.decode_event(idx)
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
                    note.pitch = event.pitch
                    note.velocity = velocity
                    if event.instrument == Event.BASS:
                        note.instrument = 0
                        note.program = 32
                    if event.instrument == Event.MELODY:
                        note.instrument = 1
                        note.program = 1
                    if event.instrument == Event.OTHER:
                        note.instrument = 2
                        note.program = 16
                    note.is_drum = False
                    if note.end_time > sequence.total_time:
                        sequence.total_time = note.end_time
            elif event.event_type == Event.TIME_SHIFT:
                step += event.event_value
        return sequence

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
        # self.remove_out_of_bound_notes(ns)
        # ns = quantize_note_sequence(ns, self.steps_per_quarter)
        return [
            self.infer_chords(quantize_note_sequence(
                i, self.steps_per_quarter))
            for i in split_note_sequence_on_time_changes(ns)
        ]


class MIDIPerformanceEncoder(MIDIEncoder):

    def __init__(
        self,
        num_velocity_bins,
        steps_per_second=None,
    ):
        super(MIDIPerformanceEncoder, self).__init__()
        self.num_reserved_ids = 4
        self.token_pad = 0
        self.token_sos = 1
        self.token_eos = 2
        self.token_bar = 3
        self.num_velocity_bins = num_velocity_bins
        self.vocab_size = self.encoding.num_classes + self.num_reserved_ids
        self.encoding = PerformanceOneHotEncoding(num_velocity_bins)

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
    midi_encoder = MIDISongEncoder(4)
    midi_dir = "/Users/mmg/uni/midi/beatles/"
    for f in os.listdir(midi_dir):
        print(f)
        midi_file = os.path.join(midi_dir, f)
        midi = pretty_midi.PrettyMIDI(midi_file)
        ns = midi_to_note_sequence(midi)
        ns = split_note_sequence_on_time_changes(ns)[0]
        ns = quantize_note_sequence(ns, 4)
        ids = midi_encoder.encode_note_sequence(ns)
        out = midi_encoder.decode_ids(ids)
        note_sequence_to_midi_file(out, os.path.join('out', f))
