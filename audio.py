import demucs.separate
from pydub import AudioSegment
import ffmpeg
import os
import shlex
import shutil
import glob
import re

sample_rate = 16000

def separate(file, outdir):
    demucs.separate.main(shlex.split(f'-n htdemucs --two-stems=vocals "{file}" -o "{outdir}"'))
    basename_without_ext = os.path.splitext(os.path.basename(file))[0]
    return os.path.join(outdir, "htdemucs", basename_without_ext, "vocals.wav")

def preprocess_audio(file_path, target_dBFS=-5, vocals_flg=True):
    file_dir = os.path.dirname(file_path)
    file_name, _ = os.path.splitext(file_path)
    output_file = f"{file_name}_preprocess.wav"
    ffmpeg.input(file_path).output(output_file, acodec='pcm_s16le', ar=sample_rate, ac=1).overwrite_output().run()
    audio = AudioSegment.from_wav(output_file)
    if int(audio.dBFS) != target_dBFS:
        audio = audio.apply_gain(target_dBFS - audio.dBFS)
        audio.export(output_file, format="wav")

    if vocals_flg:
      temp_outputs = os.path.join(file_dir, "temp_outputs")
      os.makedirs(temp_outputs, exist_ok=True)
      basename_without_ext = os.path.splitext(os.path.basename(output_file))[0]
      ffmpeg.input(output_file).output(os.path.join(temp_outputs, f"{basename_without_ext}.%d.wav"), f="segment", segment_time=1800, c="copy").overwrite_output().run()
      output_files = glob.glob(os.path.join(temp_outputs, f"{basename_without_ext}.*.wav"))
      output_files.sort(key=lambda f: int(os.path.basename(f).split('.')[-2]))

      result_list_comprehension = [separate(file, temp_outputs) for file in output_files]
      inputs = {f: ffmpeg.input(f) for f in result_list_comprehension}
      ffmpeg.concat(*inputs.values(), v=0, a=1).output(output_file, acodec='pcm_s16le', ar=sample_rate, ac=1).overwrite_output().run()
      shutil.rmtree(temp_outputs)
    return output_file

def get_dBFS(file_path):
    audio = AudioSegment.from_wav(file_path)
    return int(audio.dBFS)