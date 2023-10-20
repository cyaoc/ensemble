import gradio as gr
from audio import preprocess_audio, get_dBFS
from whisper import transcribe_to_srt
from nemo_sd import diarize_to_rttm
from merge import get_speakers, merge_speakers_sub
from llm import engine_building, engine_loading

def preprocess(file_path, target_dBFS, vocals_flg):
    output_file = preprocess_audio(file_path, target_dBFS, vocals_flg)
    return output_file, output_file

def transcribe(audio_file, whisper_model, compute_type, beam_size, min_silence_duration_ms, initial_prompt, is_align):
    text, srt = transcribe_to_srt(audio_file, whisper_model, compute_type, beam_size, min_silence_duration_ms, initial_prompt, is_align)
    return text, srt, srt

def diarize(audio_file, config_type):
    text, rttm_file = diarize_to_rttm(audio_file, config_type)
    return text, rttm_file, rttm_file

def show_speakers(rttm_file):
    _, speakers = get_speakers(rttm_file.name)
    speakers_list = list(speakers)
    return { "headers":speakers_list, "data":[speakers_list] }

def meger_text(rttm_file, subs_file, speakers):
    speaker_ts, _ = get_speakers(rttm_file.name)
    content, warning = merge_speakers_sub(subs_file.name, speaker_ts, speakers.to_dict('records')[0])
    return gr.Dropdown(["all"] + speakers.iloc[0].values.tolist()), content, warning, content

def speaker_filter(speakers_selector, cached_text):
    parts = cached_text.split('\n')
    filtered_text = [part for part in parts if part.startswith(speakers_selector + ':')]
    return '\n'.join(filtered_text)

def save_file(text):
    file = "output.txt"
    with open(file, 'w') as f:
        f.write(text)
    return file, file

def generate(input_file,api_base,api_key,api_version,engine,embed_model_name,embed_deployment_name,embed_model_api_version):
    llm_engine, zip = engine_building(input_file.name,api_base,api_key,api_version,engine,embed_model_name,embed_deployment_name,embed_model_api_version)
    return zip, gr.Button(visible=False), gr.Tabs.update(selected=1),llm_engine

def load_datas(input_datas_file,api_base,api_key,api_version,engine,embed_model_name,embed_deployment_name,embed_model_api_version):
    llm_engine = engine_loading(input_datas_file.name,api_base,api_key,api_version,engine,embed_model_name,embed_deployment_name,embed_model_api_version)
    return llm_engine

def user(message, history):
    return "", history + [[message, None]]

def bot(history,llm_engine):
    user_message = history[-1][0]
    response = llm_engine.query(user_message)
    history.append([None, str(response)])
    return history

def send_to_other_tab(info, target_tab):
    return info, gr.Tabs.update(selected=target_tab)

with gr.Blocks() as demo:
    gr.Markdown("# 没想好名字的AI工具")
    with gr.Tabs() as tabs:
        with gr.TabItem("预处理", id=0):
            gr.Markdown("将音频或视频转换为采样速率16k的wav文件，可进行人声分离及说话分贝值调整")
            with gr.Row():
                with gr.Column():
                    raw_audio_input = gr.Audio(label="Input Audio", type="filepath")
                    decibel = gr.Slider(-50, 0, step=1, label = "分贝", info="声音过轻可在此处调节")
                    vocals_flg = gr.Checkbox(value=True, label="人声分离", info="去除背景音")
                    preprocess_audio_btn = gr.Button("预处理")
                with gr.Column():
                    cached_preprocess = gr.State()
                    pre_audio_output = gr.Audio(label="Output Audio", type="filepath")
                    with gr.Row():
                        pre_to_transcription_btn = gr.Button("发送到语音转录")
                        pre_to_speaker_recognition_btn = gr.Button("发送到说话人识别")
                raw_audio_input.upload(get_dBFS, inputs=raw_audio_input, outputs=decibel)
                preprocess_audio_btn.click(preprocess, inputs=[raw_audio_input,decibel,vocals_flg], outputs=[pre_audio_output,cached_preprocess])
        with gr.TabItem("语音转录", id=1):
            gr.Markdown("音频文件转录成文字")
            with gr.Row():
                with gr.Column():
                    wav_audio_input = gr.Audio(label="Input Audio",info="推荐使用采样速率16K的音频文件", type="filepath")
                    whisper_models = gr.Dropdown(["medium", "large-v2"], value="medium", label="Models", info="选择转录模型")
                    compute_type = gr.Radio(["float16", "float32"], value="float16", label="compute_type", info="单精度或双精度")
                    beam_size = gr.Slider(1, 10, step=1, value=5, label = "beam_size")
                    vad_parameters = gr.Slider(100, 10000, step=100, value=2000, label = "vad_min_silence_duration_ms")
                    initial_prompt = gr.Textbox(label="initial_prompt")
                    is_align_flg = gr.Checkbox(label="对齐", info="wav2vec2模型")
                    transcribe_btn = gr.Button("转录")
                with gr.Column():
                    cached_srt = gr.State()
                    subs_preview = gr.Textbox(label="字幕预览", show_copy_button=True)
                    subs_file = gr.File(label="字幕文件",file_types=['.str','.ass'])
                    with gr.Row():
                        send_srt_to_merge_btn = gr.Button("发送到合并信息")
                        send_srt_to_llm_btn = gr.Button("发送到LLM知识库")
                transcribe_btn.click(transcribe, inputs=[wav_audio_input,whisper_models,compute_type, beam_size,vad_parameters,initial_prompt,is_align_flg], outputs=[subs_preview, subs_file, cached_srt])
        with gr.TabItem("说话人分类", id=2):
            gr.Markdown("在多人会话中，将不同说话人进行分类")
            with gr.Row():
                with gr.Column():
                    source_audio_input = gr.Audio(label="Input Audio", type="filepath")
                    config_type = gr.Dropdown(["general", "meeting", "telephonic"], value="telephonic", label="配置类型", info="预配置模版")
                    diarize_btn = gr.Button("分类")
                with gr.Column():
                    cached_rttm = gr.State()
                    rttm_preview = gr.Textbox(label="说话人分类预览", show_copy_button=True)
                    rttm_file = gr.File(label="rttm文件", file_types=['.rrtm'])
                    send_rttm_to_merge_btn = gr.Button("发送到合并信息")
            diarize_btn.click(diarize, inputs=[source_audio_input,config_type], outputs=[rttm_preview,rttm_file,cached_rttm])        
        with gr.TabItem("合并信息", id=3):
            gr.Markdown("将说话人和字幕文件进行匹配")
            with gr.Row():
                with gr.Column():
                    rttm_file_input = gr.File(label="rttm文件", file_types=['.rttm'])
                    subs_file_input = gr.File(label="字幕文件",file_types=['.srt'])
                    speakers = gr.Dataframe(label="说话人列表",row_count=(1, "fixed"))
                    merge_btn = gr.Button("合并")
                with gr.Column():
                    speakers_selector = gr.Dropdown(["all"], value="all", label="说话人选择")
                    text_preview = gr.Textbox(label="合成预览", show_copy_button=True)
                    warning_show = gr.Textbox(label="匹配度警告")
                    cached_text = gr.State()
                    make_file_btn = gr.Button("生成文件")
                with gr.Column():
                    cached_output = gr.State()
                    text_file = gr.File(label="合并文件", file_types=['text'])
                    send_output_to_llm_btn = gr.Button("发送到LLM知识库")
                rttm_file_input.upload(show_speakers, inputs=rttm_file_input, outputs=speakers)
                merge_btn.click(meger_text, inputs=[rttm_file_input,subs_file_input,speakers], outputs=[speakers_selector, text_preview, warning_show, cached_text])
                speakers_selector.select(speaker_filter, inputs=[speakers_selector, cached_text], outputs=text_preview)
                make_file_btn.click(save_file, inputs=cached_text, outputs=[text_file,cached_output])
        with gr.TabItem("LLM知识库", id=4):
            gr.Markdown("你可以直接和你的上传文件进行对话")
            with gr.Row():
                with gr.Column(scale=1):
                    llm_selector = gr.Dropdown(["Azure OpenAI"], value="Azure OpenAI", label="LLM选择")
                    api_base = gr.Textbox(label="api_base")
                    api_key = gr.Textbox(label="api_key", type="password")
                    api_version = gr.Textbox(label="api_version")
                    engine = gr.Textbox(label="engine")
                    embed_model_name = gr.Textbox(label="embed_model_name", value="text-embedding-ada-002")
                    embed_deployment_name = gr.Textbox(label="embed_deployment_name")
                    embed_model_api_version = gr.Textbox(label="embed_model_api_version", value="2023-05-15")
                    with gr.Tabs() as files:
                        with gr.TabItem("未处理文件", id=0):
                            input_file = gr.File(label="私人文件", file_types=['text','.srt'])
                            generation_btn = gr.Button("生成知识库")
                        with gr.TabItem("知识库文件", id=1):
                            input_datas_file = gr.File(label="知识库文件", file_types=['.zip'])
                            load_btn = gr.Button("加载")
                with gr.Column(scale=3):
                    llm_engine = gr.State()
                    llm_index = gr.State()
                    chatbot = gr.Chatbot()
                    msg = gr.Textbox(label="Input message")
                    clear = gr.Button("Clear")
                generation_btn.click(generate, inputs=[input_file,api_base,api_key,api_version,engine,embed_model_name,embed_deployment_name,embed_model_api_version], outputs=[input_datas_file,load_btn,files, llm_engine])
                load_btn.click(load_datas,inputs=[input_datas_file,api_base,api_key,api_version,engine,embed_model_name,embed_deployment_name,embed_model_api_version], outputs=llm_engine)
                msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, [chatbot, llm_engine], chatbot)
                clear.click(lambda: None, None, chatbot, queue=False)
    pre_to_transcription_btn.click(send_to_other_tab, inputs=[cached_preprocess, gr.State(value=1)], outputs=[wav_audio_input,tabs])
    pre_to_speaker_recognition_btn.click(send_to_other_tab, inputs=[cached_preprocess, gr.State(value=2)], outputs=[source_audio_input,tabs])
    send_srt_to_merge_btn.click(send_to_other_tab, inputs=[cached_srt, gr.State(value=3)], outputs=[subs_file_input,tabs])
    send_srt_to_llm_btn.click(send_to_other_tab, inputs=[cached_srt, gr.State(value=4)], outputs=[input_file,tabs])
    send_rttm_to_merge_btn.click(send_to_other_tab, inputs=[cached_rttm, gr.State(value=3)], outputs=[rttm_file_input,tabs])
    send_output_to_llm_btn.click(send_to_other_tab, inputs=[cached_output, gr.State(value=4)], outputs=[input_file,tabs])
demo.queue(max_size=50).launch(debug=True, share=True, inline=False)