# CODEALPHA
# Task#2: Music Generation with AI
# Objective: Create an AI-powered music generation system capable of composing original music. Utilize deep learning techniques like Recurrent Neural Networks (RNNs) or Generative Adversarial Networks (GANs) to generate music sequences.

# This is a Console Based Music generator App.
# *********************************** AI_MUSIC_GENERATOR *******************************************

import os
import numpy as np
import tensorflow as tf
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from pickle import load

# Path for MIDI songs
MIDI_DIR = "midi_songs"

# Check if the folder has MIDI files
def get_notes():
    notes = []
    if not os.path.exists(MIDI_DIR) or len(os.listdir(MIDI_DIR)) == 0:
        raise FileNotFoundError(f"The '{MIDI_DIR}' folder is empty or missing. Add some MIDI files!")

    for file in os.listdir(MIDI_DIR):
        if file.endswith(".mid"):
            midi = converter.parse(os.path.join(MIDI_DIR, file))
            print(f"Parsing {file}")

            notes_to_parse = None
            try:  # Handle multi-track MIDI files
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except Exception:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# Prepare input sequences
def prepare_sequences(notes, n_vocab):
    sequence_length = 100
    pitch_names = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output)
    return network_input, network_output

# Build LSTM model
def create_model(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Generate music
def generate_music(model, network_input, pitch_names, n_vocab):
    start = np.random.randint(0, len(network_input) - 1)
    int_to_note = dict((number, note) for number, note in enumerate(pitch_names))

    pattern = network_input[start]
    prediction_output = []

    for note_index in range(500):  # Generate 500 notes
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

# Create MIDI file
def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = [note.Note(int(n)) for n in notes_in_chord]
            for n in notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')
    print("MIDI file created as 'output.mid'!")

# Main program
try:
    print("Getting notes...")
    notes = get_notes()

    print("Preparing sequences...")
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)

    print("Building model...")
    model = create_model(network_input, n_vocab)

    print("Training model (This may take some time)...")
    model.fit(network_input, network_output, epochs=10, batch_size=64)

    print("Generating music...")
    pitch_names = sorted(set(item for item in notes))
    prediction_output = generate_music(model, network_input, pitch_names, n_vocab)

    print("Creating MIDI file...")
    create_midi(prediction_output)

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")
