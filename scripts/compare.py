import mido
from mido import MidiFile
from difflib import SequenceMatcher
import os

from music21 import converter, note
import numpy as np

from collections import Counter
import numpy as np

from music21 import chord

import numpy as np
from scipy.spatial.distance import cdist

def extract_notes(midi_path):
    """
    Extracts a list of (note, time) tuples from a MIDI file.
    Time is cumulative ticks to preserve order.
    """
    if not os.path.isfile(midi_path):
        raise FileNotFoundError(f"File not found: {midi_path}")
    if not midi_path.lower().endswith(".mid") and not midi_path.lower().endswith(".midi"):
        raise ValueError("File must be a MIDI (.mid/.midi) file")

    mid = MidiFile(midi_path)
    notes = []
    current_time = 0

    for track in mid.tracks:
        time_accum = 0
        for msg in track:
            time_accum += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append((msg.note, time_accum))
    return notes

def midi_similarity(file1, file2):
    """
    Returns a similarity percentage between two MIDI files based on note sequences.
    """
    try:
        notes1 = extract_notes(file1)
        notes2 = extract_notes(file2)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 0.0

    # Convert to strings for sequence matching
    seq1 = " ".join(f"{n}-{t}" for n, t in notes1)
    seq2 = " ".join(f"{n}-{t}" for n, t in notes2)

    matcher = SequenceMatcher(None, seq1, seq2)
    return round(matcher.ratio() * 100, 2)


## method 2: pitch similarity based on MIDI note numbers

def extract_pitches(midi_file):
    score = converter.parse(midi_file)
    pitches = []
    for n in score.recurse().notes:
        if isinstance(n, note.Note):
            pitches.append(n.pitch.midi)
    return pitches

def levenshtein(seq1, seq2):
    dp = np.zeros((len(seq1)+1, len(seq2)+1))
    
    for i in range(len(seq1)+1):
        dp[i][0] = i
    for j in range(len(seq2)+1):
        dp[0][j] = j

    for i in range(1, len(seq1)+1):
        for j in range(1, len(seq2)+1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[-1][-1]

def pitch_similarity(midi1, midi2):
    p1 = extract_pitches(midi1)
    p2 = extract_pitches(midi2)

    dist = levenshtein(p1, p2)
    max_len = max(len(p1), len(p2))

    similarity = (1 - dist / max_len) * 100
    return similarity

## method 3: rhythmic similarity based on note durations
def extract_notes_with_duration(midi_path):
    score = converter.parse(midi_path)
    return [(n.pitch.midi, n.duration.quarterLength)
            for n in score.recurse().notes]

def similarity_pitch_rhythm(midi1, midi2):
    seq1 = extract_notes_with_duration(midi1)
    seq2 = extract_notes_with_duration(midi2)

    dist = levenshtein(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    return (1 - dist / max_len) * 100

## histogram similarity based on pitch class distribution
def pitch_histogram(midi_path):
    pitches = extract_pitches(midi_path)
    counts = Counter(pitches)

    vec = np.zeros(128)
    for p, c in counts.items():
        vec[p] = c

    return vec / np.sum(vec)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def similarity_histogram(midi1, midi2):
    h1 = pitch_histogram(midi1)
    h2 = pitch_histogram(midi2)

    return cosine_similarity(h1, h2) * 100


## 5. Chord based similarity

def extract_chords(midi_path):
    score = converter.parse(midi_path)
    chords = []

    for c in score.chordify().recurse():
        if isinstance(c, chord.Chord):
            chords.append(tuple(sorted(p.midi for p in c.pitches)))

    return chords

def similarity_chords(midi1, midi2):
    c1 = extract_chords(midi1)
    c2 = extract_chords(midi2)

    dist = levenshtein(c1, c2)
    max_len = max(len(c1), len(c2))
    return (1 - dist / max_len) * 100

## 6. Time series alignment using Dynamic Time Warping (DTW) 

def notes_to_vector(midi_path):
    score = converter.parse(midi_path)
    return np.array([n.pitch.midi for n in score.recurse().notes])

def dtw_similarity(midi1, midi2):
    s1 = notes_to_vector(midi1)
    s2 = notes_to_vector(midi2)

    dist_matrix = cdist(s1.reshape(-1,1), s2.reshape(-1,1), metric='euclidean')

    dp = np.full(dist_matrix.shape, np.inf)
    dp[0,0] = dist_matrix[0,0]

    for i in range(1, len(s1)):
        dp[i,0] = dist_matrix[i,0] + dp[i-1,0]
    for j in range(1, len(s2)):
        dp[0,j] = dist_matrix[0,j] + dp[0,j-1]

    for i in range(1, len(s1)):
        for j in range(1, len(s2)):
            dp[i,j] = dist_matrix[i,j] + min(
                dp[i-1,j],
                dp[i,j-1],
                dp[i-1,j-1]
            )

    dist = dp[-1,-1]
    max_dist = np.max(dp)

    return (1 - dist / max_dist) * 100

## Combine all methods into a single similarity score (optional)

def combined_similarity(m1, m2):
    s1 = pitch_similarity(m1, m2)
    s2 = similarity_pitch_rhythm(m1, m2)
    s3 = similarity_histogram(m1, m2)
    s4 = similarity_chords(m1, m2)

    return (0.3*s1 + 0.3*s2 + 0.2*s3 + 0.2*s4)


# Example usage:
if __name__ == "__main__":
    base_folder = "C:\\Users\\ghekiereb\\data\\MuziekVoorspellen\\data\\raw\\bach"
    file_a = base_folder + "\\bwv2.6.mid"
    file_b = base_folder + "\\bwv1.6.mid"
    similarity = midi_similarity(file_a, file_b)
    print(f"Similarity: {similarity}%")

    # Example
    print(f"2. Pitch similarity: {pitch_similarity(file_a, file_b)}%")
    print(f"3. Pitch + rhythm similarity: {similarity_pitch_rhythm(file_a, file_b)}%")
    print(f"4. Histogram similarity: {similarity_histogram(file_a, file_b)}%")
    print(f"5. Chord similarity: {similarity_chords(file_a, file_b)}%")
    print(f"6. DTW similarity: {dtw_similarity(file_a, file_b)}%")

    print(f"Combined similarity: {combined_similarity(file_a, file_b)}%")