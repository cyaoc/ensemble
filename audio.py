import demucs.separate
from pydub import AudioSegment
import ffmpeg
import os
import shlex
import shutil

sample_rate = 16000

def preprocess_audio(file_path, target_dBFS=-5, vocals_flg=True):
    file_dir = os.path.dirname(file_path)
    temp_outputs = os.path.join(file_dir, "temp_outputs")
    file_name, _ = os.path.splitext(file_path)
    tmp_file = file_path
    if vocals_flg:
        demucs.separate.main(shlex.split(f'-n htdemucs --two-stems=vocals "{file_path}" -o "{temp_outputs}"'))
        tmp_file = os.path.join(temp_outputs, "htdemucs", os.path.basename(file_name), "vocals.wav")
    output_file = f"{file_name}.16k.wav"
    ffmpeg.input(tmp_file).output(output_file, acodec='pcm_s16le', ar=sample_rate, ac=1).overwrite_output().run()
    if os.path.exists(temp_outputs):
        shutil.rmtree(temp_outputs)

    audio = AudioSegment.from_wav(output_file)
    if int(audio.dBFS) != target_dBFS:
        audio = audio.apply_gain(target_dBFS - audio.dBFS)
        audio.export(output_file, format="wav")
    return output_file

def get_dBFS(file_path):
    audio = AudioSegment.from_wav(file_path)
    return int(audio.dBFS)