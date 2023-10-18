import pysubs2

def get_speakers(rttm):
    speaker_ts = []
    speakers = set()
    with open(rttm, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(' ')
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speakers.add(line_list[11])
            speaker_ts.append([s, e, line_list[11]])
    return speaker_ts, speakers

def calculate_overlap_percentage(a_start, a_end, b_start, b_end):
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)

    if overlap_start <= overlap_end:
        overlap_duration = overlap_end - overlap_start
        a_duration = a_end - a_start
        b_duration = b_end - b_start
        overlap_percentage = max(overlap_duration / min(a_duration, b_duration), overlap_duration / max(a_duration, b_duration))
        return overlap_percentage

    return 0.0

def find_max_overlap_or_closest(start, end, array, getter=lambda item: (item[0], item[1])):
    max_overlap_percentage = 0.0
    max_overlap_element = None
    closest_element = None
    closest_distance = float('inf')
    for item in array:
        a_start, a_end = getter(item)
        overlap_percentage = calculate_overlap_percentage(a_start, a_end, start, end)
        if overlap_percentage > max_overlap_percentage:
            max_overlap_percentage = overlap_percentage
            max_overlap_element = item
        b_midpoint = (start + end) / 2
        a_midpoint = (a_start + a_end) / 2
        distance = abs(b_midpoint - a_midpoint)
        if distance < closest_distance:
            closest_distance = distance
            closest_element = item
    return max_overlap_percentage, closest_element if max_overlap_percentage == 0.0 else max_overlap_element

def merge_speakers_sub(subs_file, speaker_ts, names_dict):
    subs = pysubs2.load(subs_file, encoding="utf-8")
    current_speaker = None
    speakers = {}
    draft = []
    warnings = []
    for line in subs:
        ws, we, wrd = line.start, line.end, line.text
        max_overlap_percentage, element = find_max_overlap_or_closest(ws, we, speaker_ts)
        speaker_code = element[2]
        if max_overlap_percentage < 0.5:
            speaker_name = names_dict.get(speaker_code, speaker_code)
            warnings.append(f"[%.2fs -> %.2fs] %s (%.2f %%):\"%s\" \n" % (ws/1000, we/1000, speaker_name, max_overlap_percentage * 100, wrd))
        speakers.setdefault(speaker_code, []).append(wrd)
        if current_speaker is not None and current_speaker != speaker_code:
            speaker_name = names_dict.get(current_speaker, current_speaker)
            draft.append(f"{speaker_name}: \"{' '.join(speakers[current_speaker])}\" \n")
            speakers[current_speaker] = []
        current_speaker = speaker_code

    if current_speaker is not None:
        speaker_name = names_dict.get(current_speaker, current_speaker)
        draft.append(f"{speaker_name}: \"{' '.join(speakers[current_speaker])}\" \n")

    return ''.join(draft), ''.join(warnings)