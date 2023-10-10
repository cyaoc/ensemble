import pysubs2

rttm = 'erd.16k.rttm'
sub = 'erd.16k.zh.srt'

speaker_ts = []
with open(rttm, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(' ')
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split('_')[-1])])

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

def print_speaker_words(speaker, words, names):
    if speaker in names:
        return f"{names[speaker]}: \"{words}\" \n"
    else:
        return f"speaker[{speaker}]: \"{words}\" \n"

subs = pysubs2.load(sub, encoding="utf-8")
current_speaker = None
speakers = {}
names = ["顾三乐","姚超","赵雷","严蓓琦","朱珺","徐冰"]
draft = []

for line in subs:
    ws, we, wrd = line.start, line.end, line.text
    max_overlap_percentage, element = find_max_overlap_or_closest(ws, we, speaker_ts)
    if max_overlap_percentage < 0.5:
      print("Warning: max_overlap_percentage:%.2fs, [%.2fs -> %.2fs]「%s」" % (max_overlap_percentage, ws/1000, we/1000, wrd))
    speaker_index = element[2]
    speakers.setdefault(speaker_index, []).append(wrd)
    if current_speaker is not None and current_speaker != speaker_index:
        speaker_name = names[current_speaker] if 0 <= current_speaker < len(names) else f"speaker[{current_speaker}]"
        draft.append(f"{speaker_name}: \"{' '.join(speakers[current_speaker])}\" \n")
        speakers[current_speaker] = []
    current_speaker = speaker_index

if current_speaker is not None:
    speaker_name = names[current_speaker] if 0 <= current_speaker < len(names) else f"speaker[{current_speaker}]"
    draft.append(f"{speaker_name}: \"{' '.join(speakers[current_speaker])}\" \n")

content = ''.join(draft)
print(content)
with open("output.txt", "w") as f:
    f.write(content)

# import openai

# openai.api_type = "azure"
# openai.api_base = "https://chatjp.openai.azure.com/"
# openai.api_version = "2023-07-01-preview"
# openai.api_key = "f006e34f87144625b7a4b04f1eef4339"
# engine = "gpt4-32k"
# model = "gpt-4-32k"
# max_tokens = 32 * 1024

# import tiktoken

# prompt = "I want you to act as a conference summarization assistant. I will provide you with Chinese transcripts of the conference. You need to correct homophones and generate Chinese summaries that emphasize what each person did this week and what tasks they will undertake in the future. Here is the conference content:"

# encoding = tiktoken.encoding_for_model(model)
# system_tokens = len(encoding.encode(prompt))
# tokens_every_message = 3
# tokens_res_assistant = 3
# pre_tokens = len(encoding.encode("system")) + len(encoding.encode("user")) + tokens_res_assistant + 2 * tokens_every_message
# total_tokens = len(encoding.encode(content)) + pre_tokens + system_tokens
# print(f"This request consumed {total_tokens} tokens.")
# max_res_tokens = max_tokens - total_tokens -1

# if max_res_tokens < 0:
#     print("Warning: This request has exceeded the maximum processing limit of GPT. Please change the model.")
#     exit()
# if max_res_tokens < 400:
#     print("Warning: The response for this request is fewer than 400 tokens. Please take note.")

# completion = openai.ChatCompletion.create(
#   engine=engine,
#   messages = [
#     {"role": "system", "content": prompt},
#     {"role": "user", "content": content}
#   ],
#   temperature=0.7,
#   max_tokens=max_res_tokens,
#   top_p=0.95,
#   frequency_penalty=0,
#   presence_penalty=0,
#   stop=None)
# result  = completion.choices[0].message.content
# print(result)