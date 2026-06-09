Functions to compare to midi files



Available under 



1. MIDI similarity : midi\_similarity(file\_a, file\_b)

2\. Pitch similarity : pitch\_similarity(file\_a, file\_b)

3\. Pitch + rhythm similarity : similarity\_pitch\_rhythm(file\_a, file\_b)

4\. Histogram similarity : similarity\_histogram(file\_a, file\_b)

5\. Chord similarity : similarity\_chords(file\_a, file\_b)

6\. DTW similarity : dtw\_similarity(file\_a, file\_b)



Combined similarity : combined\_similarity(file\_a, file\_b)



Brief explanation

1. midi\_similarity(file1, file2) computes how similar two MIDI files are by comparing their note sequences.
In brief:

&#x20;  1. Extract notes:It calls extract\_notes on both files to get lists of (note, time) pairs (note pitch + when it occurs).

&#x20;  2. Convert to sequences: Each list is turned into a string like "60-0 64-480 67-960" so the order and timing are encoded.

&#x20;  3. Compare sequences: It uses SequenceMatcher (a text comparison tool) to measure how similar the two strings are.

&#x20;  4. Return result: the similarity score is returned as a percentage (0–100%).



👉 Overall: it measures how closely the two MIDI files match in terms of which notes are played and when, by treating them as sequences and comparing them like text.





2\. 🎵 Pitch similarity — 22.4%



Compares the sequence of notes (pitches only)

Ignores timing and rhythm

Low score → melodies are quite different



👉 Your result:

→ Only \~22% of the melody matches → melodically quite different



3\. ⏱️ Pitch + rhythm similarity — 14.7%



Compares pitch AND note durations

More strict than pitch-only



👉 Your result:

→ Even lower → melody + timing differ a lot

→ They not only use different notes, but also different rhythms



4\. 📊 Histogram similarity — 61.1%



Compares which notes are used overall (not order)

Like comparing the “musical palette” or key/scale



👉 Your result:

→ \~61% → similar tonal style / key usage

→ They use similar notes, just in different order



5\. 🎹 Chord similarity — 0.61%



Compares harmonic structure (chords)



👉 Your result:

→ Almost 0% → completely different harmony

→ Likely different chord progressions or texture



6\. 📈 DTW similarity — 71.8%



Uses Dynamic Time Warping (flexible alignment)

Allows stretching/compressing timing

Best for overall similarity



👉 Your result:

→ \~72% → moderately to strongly similar overall

→ Suggests:



Similar musical contour or structure

But maybe played at different speeds or phrasing

