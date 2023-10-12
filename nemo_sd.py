import os
import json
import wget
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import torch
import os
import shutil

def diarize(audio_file, config_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    file_dir = os.path.dirname(audio_file)
    file_path_no_ext, _ = os.path.splitext(audio_file)
    file_name = os.path.basename(file_path_no_ext)
    work_home = os.path.join(file_dir, "nemo")
    if not os.path.exists(work_home):
        os.makedirs(work_home)

    meta = {
        'audio_filepath': f'{audio_file}',
        'offset': 0,
        'duration':  None,
        'label': "infer",
        'text': "-",
        'num_speakers': None,
        'rttm_filepath': None,
        'uniq_id': ""
    }
    manifest = os.path.join(work_home, 'manifest.json')
    if not os.path.exists(manifest):
        with open(manifest, 'w') as f:
            f.write(json.dumps(meta))   
    mode_config = os.path.join(work_home, f'diar_infer_{config_type}.yaml')
    if not os.path.exists(mode_config):
        config_url = f'https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_{config_type}.yaml'
        mode_config = wget.download(config_url, work_home)

    config = OmegaConf.load(mode_config)
    config.num_workers = 1
    config.diarizer.manifest_filepath = manifest
    config.diarizer.out_dir = os.path.join(work_home, 'diarized')
    config.diarizer.vad.model_path = 'vad_multilingual_marblenet'
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.speaker_embeddings.model_path = 'titanet_large'
    config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic'
    config.diarizer.oracle_vad = False
    config.diarizer.clustering.parameters.oracle_num_speakers=False
    model = NeuralDiarizer(cfg=config).to(device)
    model.diarize()
    del model
    torch.cuda.empty_cache()

    source_file = os.path.join(work_home, f"diarized/pred_rttms/{file_name}.rttm")
    if os.path.exists(source_file):
        shutil.move(source_file, file_dir) 
        shutil.rmtree(work_home)

    rttm = os.path.join(file_dir, f"{file_name}.rttm")
    with open(rttm, 'r') as file:
        text = file.read()
    return text, rttm
