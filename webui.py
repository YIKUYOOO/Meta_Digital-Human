import gradio as gr
from gradio_client import Client, file
import os
import shutil
import subprocess

# GPT-SoVITS API 地址
GPT_SOVITS_API_URL = "http://localhost:9872/"

# 初始化 API 客户端
client = Client(GPT_SOVITS_API_URL)

# 获取 GPT-SoVITS 服务器上可用的 SoVITS 语音模型和 GPT 语音模型
def get_available_models():
    result = client.predict(api_name="/change_choices")
    if not isinstance(result, tuple) or len(result) < 2:
        raise ValueError(f"API 返回格式不正确: {result}")
    
    # 解析 SoVITS 语音模型
    sovits_models = [model[0] for model in result[0]["choices"]]
    
    # 解析 GPT 语音模型
    gpt_models = [model[0] for model in result[1]["choices"]]

    print("🔹 可用的 SoVITS 语音模型:", sovits_models)
    print("🔹 可用的 GPT 语音模型:", gpt_models)

    return sovits_models, gpt_models

# 切换 SoVITS 和 GPT 语音模型
def change_models(sovits_model, gpt_model):
    client.predict(
        sovits_path=sovits_model,
        prompt_language="中文",
        text_language="中文",
        api_name="/change_sovits_weights"
    )
    print(f"已切换 SoVITS 语音模型: {sovits_model}")
    
    client.predict(
        gpt_path=gpt_model,
        api_name="/change_gpt_weights"
    )
    print(f"已切换 GPT 语音模型: {gpt_model}")

# 通过 GPT-SoVITS API 生成语音
def generate_speech(ref_audio_path, text, sovits_model, gpt_model, prompt_text, prompt_language, how_to_cut, top_k, top_p, temperature, ref_free, speed, if_freeze, sample_steps):
    # 切换模型
    change_models(sovits_model, gpt_model)

    # 生成语音
    print("正在向 GPT-SoVITS 发送 TTS 请求...\n\n")
    
    result = client.predict(
        ref_wav_path=file(ref_audio_path),
        prompt_text=prompt_text,
        prompt_language=prompt_language,
        text=text,
        text_language="中文",
        how_to_cut=how_to_cut,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        ref_free=ref_free,
        speed=speed,
        if_freeze=if_freeze,
        inp_refs=None,
        sample_steps=sample_steps,
        api_name="/get_tts_wav"
    )

    # 打印返回的路径并检查文件
    print(f"API 返回的音频路径: {result}")

    if not os.path.exists(result):
        raise ValueError(f"API 生成的音频文件不存在: {result}")

    # 目标路径
    output_audio_path = os.path.join("output", "generated_audio.wav")
    shutil.copy(result, output_audio_path)
    print(f"语音合成完成: {output_audio_path}")

    # 生成视频的目标路径
    output_video_path = os.path.join("E:/Project/Digital_Human/Wav2Lip/results", "result_voice.mp4")

    # Wav2Lip 所需环境路径
    wav2lip_env_path = "E:/Project/Digital_Human/Wav2Lip/env"
    wav2lip_script_path = "E:/Project/Digital_Human/Wav2Lip/inference.py"
    wav2lip_checkpoint_path = "E:/Project/Digital_Human/Wav2Lip/models/wav2lip_gan.pth"
    input_video_path = "E:\Project\Digital_Human\input\input.mp4"

    # 组装完整的 shell 命令
    wav2lip_command = f'''
    conda run -p "{wav2lip_env_path}" python "{wav2lip_script_path}" --checkpoint_path "{wav2lip_checkpoint_path}" --face "{input_video_path}" --audio "{output_audio_path}"
    '''

    # 调用 Wav2Lip 进行视频生成
    try:
        process = subprocess.run(wav2lip_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 打印输出日志，方便调试
        print(f"🔹 Wav2Lip 输出: {process.stdout}")
        print(f"🔹 Wav2Lip 错误: {process.stderr}")

        if process.returncode == 0:
            print(f"Wav2Lip 视频生成成功: {output_video_path}")
        else:
            print(f"Wav2Lip 失败，错误代码: {process.returncode}")

    except subprocess.CalledProcessError as e:
        print(f"调用 Wav2Lip 生成视频失败: {e}")
        
    return output_audio_path, output_video_path

# 获取可用模型并初始化
sovits_models, gpt_models = get_available_models()

# Gradio 界面
def gradio_interface(ref_audio, text, sovits_model, gpt_model, prompt_text, prompt_language, how_to_cut, top_k, top_p, temperature, ref_free, speed, if_freeze, sample_steps):
    return generate_speech(ref_audio, text, sovits_model, gpt_model, prompt_text, prompt_language, how_to_cut, top_k, top_p, temperature, ref_free, speed, if_freeze, sample_steps)

# 构建 Gradio 界面
with gr.Blocks() as demo:
    with gr.Row():
        # 下拉选择模型
        sovits_model_dropdown = gr.Dropdown(sovits_models, label="选择 SoVITS 模型")
        gpt_model_dropdown = gr.Dropdown(gpt_models, label="选择 GPT 模型")
    
    with gr.Row():
        # 上传参考音频文件
        ref_audio_input = gr.Audio(label="上传参考音频", type="filepath")
        text_input = gr.Textbox(label="输入要合成的文本", placeholder="请输入文本")
    
    with gr.Row():
        # 额外参数的输入框
        prompt_text_input = gr.Textbox(label="输入提示文本", value="你好，我是你的数字人助手！")
        prompt_language_input = gr.Dropdown(["中文", "英文"], label="选择提示语言", value="中文")
        how_to_cut_input = gr.Textbox(label="如何切割文本", value="凑四句一切")
        top_k_input = gr.Slider(minimum=1, maximum=50, label="Top-k", value=15)
        top_p_input = gr.Slider(minimum=0.0, maximum=1.0, label="Top-p", value=1.0)
        temperature_input = gr.Slider(minimum=0.0, maximum=2.0, label="temperature", value=1.0)
        ref_free_input = gr.Checkbox(label="开启无参考文本模式", value=False)
        speed_input = gr.Slider(minimum=0.5, maximum=2.0, label="语速", value=1.0)
        if_freeze_input = gr.Checkbox(label="是否直接对上次合成结果调整语速和音色。防止随机性。", value=False)
        sample_steps_input = gr.Slider(minimum=1, maximum=50, label="采样步数", value=8)

    with gr.Row():
        # 生成按钮
        generate_button = gr.Button("生成数字人语音与唇动")
    
    with gr.Row():
        # 输出音频和视频
        output_audio = gr.Audio(label="生成的音频")
        output_video = gr.Video(label="生成的视频")

    # 设置生成按钮的点击事件
    generate_button.click(
        gradio_interface,
        inputs=[
            ref_audio_input, 
            text_input, 
            sovits_model_dropdown, 
            gpt_model_dropdown,
            prompt_text_input, 
            prompt_language_input, 
            how_to_cut_input, 
            top_k_input, 
            top_p_input, 
            temperature_input, 
            ref_free_input, 
            speed_input, 
            if_freeze_input, 
            sample_steps_input
        ],
        outputs=[output_audio, output_video]
    )

# 启动 Web UI
demo.launch()
