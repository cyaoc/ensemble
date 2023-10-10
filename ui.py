import gradio as gr
from audio import preprocess_audio, get_dBFS
from whisper import transcribe_to_srt

# def show_buttons(*btns):
#     return list(map(lambda _: gr.Button(visible=True), btns))

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
                    pre_audio_output = gr.Audio(label="Output Audio", type="filepath")
                    with gr.Row():
                        pre_to_transcription_btn = gr.Button("发送到语音转录")
                        pre_to_speaker_recognition_btn = gr.Button("发送到说话人识别")
                raw_audio_input.upload(get_dBFS, inputs=raw_audio_input, outputs=decibel)
                preprocess_audio_btn.click(preprocess_audio, inputs=[raw_audio_input,decibel,vocals_flg], outputs=pre_audio_output)
                # pre_audio_output.change(show_buttons, inputs=[pre_to_transcription_btn, pre_to_speaker_recognition_btn], outputs=[pre_to_transcription_btn, pre_to_speaker_recognition_btn])
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
                    is_align_flg = gr.Checkbox(value=True, label="对齐", info="不勾选可以加快生成速度")
                    transcribe_btn = gr.Button("转录")
                with gr.Column():
                    subs_preview = gr.Textbox(label="字幕预览", show_copy_button=True)
                    subs_file = gr.File(label="字幕文件",file_types=['.str','.ass'])
                    with gr.Row():
                        send_to_merge = gr.Button("发送到说话人合并")
                        send_to_llm = gr.Button("Send to llm")
                transcribe_btn.click(transcribe_to_srt, inputs=[wav_audio_input,whisper_models,compute_type, beam_size,vad_parameters,initial_prompt,is_align_flg], outputs=[subs_preview,subs_file])
            pre_to_transcription_btn.click(send_to_other_tab, inputs=[pre_audio_output, gr.State(value=1)], outputs=[wav_audio_input,tabs])
        with gr.TabItem("说话人分类", id=2):
            gr.Markdown("在多人会话中，将不同说话人进行分类")
            with gr.Row():
                with gr.Column():
                    source_audio_input = gr.Audio(label="Input 16K Audio", type="filepath")
                    config_type = gr.Dropdown(["general", "meeting", "telephonic"], value="telephonic", label="配置类型", info="预配置模版")
                with gr.Column():
                    speaks_show = gr.Number(label="说话人数")
                    rttm_show = gr.Textbox(label="分类结果")
                    send_to_merge = gr.Button("Send to merge")
                    
        with gr.TabItem("合并", id=3):
            gr.Markdown("将不同的说话人和字幕文件进行匹配")
            speaks_show = gr.Textbox(label="说话人列表")
        with gr.TabItem("LLM知识库", id=4):
            gr.Markdown("你可以直接和你的上传文件进行对话")

demo.launch()