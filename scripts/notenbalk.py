from music21 import converter, instrument, stream, note, chord
from music21 import configure
from music21 import environment
configure.run()


def midi_naar_notenbalk(midi_pad):
    # Lees MIDI bestand
    mid = converter.parse(midi_pad)

    # Partitioneer per instrument/stem
    parts = instrument.partitionByInstrument(mid)

    score = stream.Score()

    if parts:
        # Als er verschillende instrumenten/stemmen zijn
        for part in parts.parts:
            p = stream.Part()
            for element in part.recurse():
                if isinstance(element, note.Note):
                    p.append(element)
                elif isinstance(element, chord.Chord):
                    p.append(element)
            score.append(p)
    else:
        # Als er geen aparte instrumenten zijn
        p = stream.Part()
        for element in mid.flat:
            if isinstance(element, note.Note) or isinstance(element, chord.Chord):
                p.append(element)
        score.append(p)

    return score


if __name__ == "__main__":
    print(environment.Environment()['musicxmlPath'])
    midi_file = "voorbeeld.mid"  # <-- zet hier je MIDI bestand
    base_folder = "C:\\Users\\ghekiereb\\data\\MuziekVoorspellen\\data\\raw\\bach"
    midi_file = base_folder + "\\bwv1.6.mid"
    score = midi_naar_notenbalk(midi_file)

    # Toon als notenbalk (vereist MuseScore of ander notation programma)
    score.show()