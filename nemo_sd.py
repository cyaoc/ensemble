import os
import json
import wget
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import torch

import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def diarize(audio_file, config_type):
    file_dir = os.path.dirname(audio_file)
    file_name, _ = os.path.splitext(audio_file)
    rttm = f'{file_name}.rttm'
    meta = {
        'audio_filepath': f'{audio_file}',
        'offset': 0,
        'duration':  None,
        'label': "infer",
        'text': "-",
        'num_speakers': None,
        'rttm_filepath': f'{rttm}',
        'uniq_id': ""
    }
    print(audio_file)
    print(rttm)
    manifest = os.path.join(file_dir, 'manifest.json')
    print(manifest)
    if not os.path.exists(manifest):
        with open(manifest, 'w') as f:
            f.write(json.dumps(meta))   

    mode_config = os.path.join(file_dir, f'diar_infer_{config_type}.yaml')
    print(mode_config)
    if not os.path.exists(mode_config):
        config_url = f'https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_{config_type}.yaml'
        mode_config = wget.download(config_url, file_dir)
        print(mode_config)
    config = OmegaConf.load(mode_config)
    config.num_workers = 1

    config.diarizer.manifest_filepath = manifest
    config.diarizer.out_dir = os.path.join(file_dir, 'diarized')
    print(config.diarizer.out_dir)
    config.diarizer.vad.model_path = 'vad_multilingual_marblenet'
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.speaker_embeddings.model_path = 'titanet_large'
    config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic'

    config.diarizer.oracle_vad = False
    config.diarizer.clustering.parameters.oracle_num_speakers=False
    model = NeuralDiarizer(cfg=config)
    model.diarize()
    del model
    torch.cuda.empty_cache()

    list_files(file_dir)

    with open(rttm, 'r') as file:
        text = file.read()
    return text, rttm
