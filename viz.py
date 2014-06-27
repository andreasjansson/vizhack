import numpy as np
from scipy.signal import argrelmax, medfilt, medfilt2d
from scipy.spatial.distance import cityblock, pdist

def get_top_pitch(analysis, n):
    pitches = np.array([s['pitches'] for s in analysis['segments']])
    index = np.argsort(np.sum(pitches, 0))[::-1][n]
    return medfilt(pitches[:, index], 21)

def get_top_changing_timbres(analysis, n):
    timbres = np.array([s['timbre'] for s in analysis['segments']])
    timbres = (timbres - np.min(timbres, 0)) / (np.max(timbres, 0) - np.min(timbres, 0))
    return medfilt(timbres[:, n], 21)

def get_loudness(analysis):
    loudness = np.array([s['loudness_max'] for s in analysis['segments']])
    loudness = medfilt(loudness, 21)
    loudness = (loudness - np.min(loudness)) / (np.max(loudness) - np.min(loudness))
    return loudness

def get_loudness_delta(analysis):
    loudness = np.array([s['loudness_max'] for s in analysis['segments']])
    loudness = medfilt(loudness, 21)
    return loudness[20:] - loudness[:-20]

def get_pitch_cleanliness(analysis):
    pitches = np.array([s['pitches'] for s in analysis['segments']])
    return medfilt(kurtosis(pitches.T), 21)

def get_pitch_cleanliness_delta(analysis):
    cleanliness = pitch_cleanliness(analysis)
    return cleanliness[20:] - cleanliness[:-20]    

def get_confidence(analysis):
    confidences = np.array([s['confidence'] for s in analysis['segments']])
    return medfilt(confidences, 21)

def get_bar_change(analysis):
    segment_starts = np.array([s['start'] for s in analysis['segments']])
    bar_starts = np.array([s['start'] for s in analysis['bars']])
    bar_change = np.zeros(len(segment_starts))
    bsi = 0
    for i, segment_start in enumerate(segment_starts):
        if segment_start >= bar_starts[bsi]:
            bsi += 1
            bar_change[i] = 1
            if bsi == len(bar_starts):
                break
    return bar_change

def get_section_change(analysis):
    segment_starts = np.array([s['start'] for s in analysis['segments']])
    section_starts = np.array([s['start'] for s in analysis['sections']])
    section_change = np.zeros(len(segment_starts))
    csi = 0
    for i, segment_start in enumerate(segment_starts):
        if segment_start >= section_starts[csi]:
            csi += 1
            section_change[i] = 1
            if csi == len(section_starts):
                break
    return section_change

def get_duration(analysis):
    durations = np.array([s['duration'] for s in analysis['segments']])
    return medfilt(durations, 21)

class Participant(object):

    def __init__(self, name, pitch, timbre):
        self.name = name
        self.position = None
        self.pitch = pitch
        self.timbre = timbre

    def __repr__(self):
        return str(self.__dict__)

def initialize_participants(n_participants):
    pitch_indices = np.random.choice(np.arange(n_participants), n_participants, replace=False)
    timbre_indices = np.random.choice(np.arange(n_participants), n_participants, replace=False)
    participants = []
    for i, (p, t) in enumerate(zip(pitch_indices, timbre_indices)):
        participants.append(Participant(i, p, t))
    return participants

def calculate_pitch_diffs(t, top_pitches, participants):
    pitches = [top_pitches[i][t] for i in range(len(participants))]
    dists = np.zeros((len(pitches), len(pitches)))
    max_dist_i = None
    max_dist = 0.5
    min_dist_i = None
    min_dist = 0.5
    for i in range(len(pitches)):
        for j in range(len(pitches)):
            dist = pitches[i] - pitches[j]
            dists[i, j] = dist
            if dist < min_dist and dist > 0:
                min_dist = dist
                min_dist_i = (i, j)
            elif dist > max_dist:
                max_dist = dist
                max_dist_i = (i, j)
            
    return np.abs(dists), min_dist_i, max_dist_i

def move_towards(p1, p2):
    return p1.name, 'move towards', p2.name

def move_away(p1, p2):
    return p1.name, 'move away from', p2.name

def change_speed(speed):
    if speed < 1:
        return 'everybody', 'stop'
    if speed < 2:
        return 'everybody', 'move in slow motion'
    if speed < 3:
        return 'everybody', 'walk'
    if speed < 4:
        return 'everybody', 'walk briskly'
    else:
        return 'everybody', 'do the twist'

def change_mood(p, mood):
    if mood < 1:
        mood_name = 'timid'
    elif mood < 2:
        mood_name = 'curious'
    elif mood < 3:
        mood_name = 'confident'
    elif mood < 4:
        mood_name = 'slick'
    elif mood <= 5:
        mood_name = 'aggressive'
    return p.name, 'be %s' % mood_name

def instruction(t, ins):
    print t, ins
    return t, ins

def script(n_participants, analysis):
    participants = initialize_participants(n_participants)

    top_pitches = [get_top_pitch(analysis, i) for i in range(len(participants))]
    top_changing_timbres = [get_top_changing_timbres(analysis, i) for i in range(len(participants))]
    loudness = get_loudness(analysis)
    loudness_delta = get_loudness_delta(analysis)
    pitch_cleanliness = get_pitch_cleanliness(analysis)
    pitch_cleanliness_delta = get_pitch_cleanliness_delta(analysis)
    confidence = get_confidence(analysis)
    bar_change = get_bar_change(analysis)
    section_change = get_section_change(analysis)
    duration = get_duration(analysis)

    time_since_last_speed_change = 0
    current_speed = None
    prev_section = 0
    prev_moods = np.zeros(n_participants)
    times_since_last_mood_change = np.zeros(n_participants)

    instructions = []

    for t in range(len(analysis['segments'])):
        ts = analysis['segments'][t]['start']

        if section_change[t]:
            section = prev_section + 1

        speed = loudness[t] * 5
        if ((time_since_last_speed_change > 20 or section != prev_section)
            and current_speed != int(speed)
            and t != len(analysis['segments']) - 1):
            instructions.append(instruction(ts, change_speed(speed)))
            current_speed = int(speed)
            time_since_last_speed_change = 0
        time_since_last_speed_change += 1

        mood = np.array([top_changing_timbres[i] for i in range(n_participants)])
        mood *= 5
        mood = mood.astype(int)

        for i in range(n_participants):
            if (times_since_last_mood_change[i] > 40
                and prev_moods[i] != mood[i][t]):
                instructions.append(instruction(ts, change_mood(participants[i], mood[i][t])))
                prev_moods[i] = mood[i][t]
                times_since_last_mood_change[i] = 0
        times_since_last_mood_change += 1

        pitch_diffs, min_i, max_i = calculate_pitch_diffs(t, top_pitches, participants)
        if np.random.rand() < 0.03 and min_i:
            instructions.append(instruction(ts, move_towards(participants[min_i[0]], participants[min_i[1]])))
            instructions.append(instruction(ts, move_towards(participants[min_i[1]], participants[min_i[0]])))
        if np.random.rand() < 0.03 and max_i:
            instructions.append(instruction(ts, move_away(participants[max_i[0]], participants[max_i[1]])))
            instructions.append(instruction(ts, move_away(participants[max_i[1]], participants[max_i[0]])))

        prev_section = section

    instructions.append(instruction(ts, change_speed(0)))

    with open('instructions.json', 'w') as f:
        json.dump(instructions, f)
