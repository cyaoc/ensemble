from faster_whisper import WhisperModel
import whisperx
import torch
import os
import pysubs2
from tqdm import tqdm
from time import sleep

def transcribe_to_srt(audio_file, whisper_model="medium", compute_type="float16", beam_size=1, min_silence_duration_ms=500, initial_prompt="", is_align=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
    segments, info = model.transcribe(audio_file, beam_size=beam_size, word_timestamps=False,
                                  vad_filter=True, vad_parameters=dict(min_silence_duration_ms=min_silence_duration_ms),
                                  initial_prompt=initial_prompt)
    whisper_results = []
    total_duration = info.duration
    duration = 0
    with tqdm(total=total_duration, unit=" seconds") as pbar:
        for segment in segments:
            whisper_results.append(segment._asdict())
            segment_duration = segment.end - segment.start
            duration += segment_duration
            pbar.update(segment_duration)
        sleep(0.1)
        pbar.update(total_duration-duration)
    del model
    torch.cuda.empty_cache()
    if is_align:
        alignment_model, metadata = whisperx.load_align_model(language_code=info.language, device=device)
        result_aligned = whisperx.align(whisper_results, alignment_model, metadata, audio_file, device)
        whisper_results = result_aligned["segments"]
        del alignment_model
        torch.cuda.empty_cache()
    subs = pysubs2.load_from_whisper(whisper_results)
    text = subs.to_string(format_="srt")
    file_name, _ = os.path.splitext(audio_file)
    srt = f"{file_name}.{info.language}.srt"
    subs.save(srt)
    return text, srt
