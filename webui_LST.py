import gradio as gr
from gradio_client import Client, file
import os
import shutil
import subprocess
import time
import cv2

# GPT-SoVITS API 地址
GPT_SOVITS_API_URL = "http://localhost:9872/"
client = Client(GPT_SOVITS_API_URL)

# Wav2Lip 相关路径
wav2lip_env_path = "E:/Project/Digital_Human/Wav2Lip/env"
wav2lip_script_path = "E:/Project/Digital_Human/Wav2Lip/inference.py"
wav2lip_checkpoint_path = "E:/Project/Digital_Human/Wav2Lip/models/wav2lip_gan.pth"
input_video_dir = "E:/Project/Digital_Human/input"
output_video_path = "E:/Project/Digital_Human/Wav2Lip/results/result_voice.mp4"

# 确保 input 文件夹存在
os.makedirs(input_video_dir, exist_ok=True)

# 获取可用的视频文件列表
def get_video_files():
    return [f for f in os.listdir(input_video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

# 录制视频功能（录制完成后刷新下拉列表）
def record_video():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_name = f"recorded_{int(time.time())}.avi"
    video_path = os.path.join(input_video_dir, video_name)
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

    print("正在录制视频, 按 'q' 退出...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Recording', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"录制完成: {video_path}")

    return gr.update(choices=get_video_files(), value=video_name)

# 获取可用模型
def get_available_models():
    result = client.predict(api_name="/change_choices")
    if not isinstance(result, tuple) or len(result) < 2:
        raise ValueError(f"API 返回格式不正确: {result}")

    sovits_models = [model[0] for model in result[0]["choices"]]
    gpt_models = [model[0] for model in result[1]["choices"]]

    print("可用的 SoVITS 语音模型:", sovits_models)
    print("可用的 GPT 语音模型:", gpt_models)

    return sovits_models, gpt_models

# 切换模型
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

# 生成语音并调用 Wav2Lip 生成视频
def generate_speech(ref_audio_path, text, sovits_model, gpt_model, prompt_text, prompt_language, how_to_cut, top_k, top_p, temperature, ref_free, speed, if_freeze, sample_steps, input_video):
    change_models(sovits_model, gpt_model)

    print("正在向 GPT-SoVITS 发送 TTS 请求...\n")
    
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

    output_audio_path = os.path.join("output", "generated_audio.wav")
    shutil.copy(result, output_audio_path)
    print(f"语音合成完成: {output_audio_path}")

    input_video_path = os.path.join(input_video_dir, input_video)

    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"输入视频文件不存在: {input_video_path}")

    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    wav2lip_command = [
        "conda", "run", "-p", wav2lip_env_path,
        "python", wav2lip_script_path,
        "--checkpoint_path", wav2lip_checkpoint_path,
        "--face", input_video_path,
        "--audio", output_audio_path,
        "--outfile", output_video_path
    ]

    print("正在运行 Wav2Lip ...")
    subprocess.run(wav2lip_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) 

    wait_time = 0
    while not os.path.exists(output_video_path):
        time.sleep(1)
        wait_time += 1
        print(f"等待视频生成... {wait_time}s")
        if wait_time > 300:
            print("超时: Wav2Lip 可能生成失败")
            break

    if os.path.exists(output_video_path):
        print(f"Wav2Lip 视频生成成功: {output_video_path}")
    else:
        print(f"Wav2Lip 生成失败, 文件 {output_video_path} 未找到")

    return output_audio_path, output_video_path

# 获取可用模型并初始化
sovits_models, gpt_models = get_available_models()

# Gradio 界面
with gr.Blocks() as demo:
    
    with gr.Row():
        gr.Text("请注意录制的视频需包含清晰人脸和说话场景")
        
    with gr.Row():
        record_button = gr.Button("🎥 录制视频")
        input_video_dropdown = gr.Dropdown(get_video_files(), label="选择输入视频")

    with gr.Row():
        sovits_model_dropdown = gr.Dropdown(sovits_models, label="选择 SoVITS 模型")
        gpt_model_dropdown = gr.Dropdown(gpt_models, label="选择 GPT 模型")
    
    with gr.Row():
        ref_audio_input = gr.Audio(label="上传参考音频", type="filepath")
        text_input = gr.Textbox(label="输入要合成的文本", placeholder="请输入文本")

    with gr.Row():
        prompt_text_input = gr.Textbox(label="输入提示文本", value="你好，我是你的数字人助手！")
        ref_free_input = gr.Checkbox(label="开启无参考文本模式", value=False)
        prompt_language_input = gr.Dropdown(["中文", "英文"], label="选择提示语言", value="中文")
        how_to_cut_input = gr.Textbox(label="如何切割文本", value="凑四句一切")
        top_k_input = gr.Slider(minimum=1, maximum=50, label="Top-k", value=15)
        top_p_input = gr.Slider(minimum=0.0, maximum=1.0, label="Top-p", value=1.0)
        temperature_input = gr.Slider(minimum=0.0, maximum=2.0, label="temperature", value=1.0)
    
    with gr.Row():
        if_freeze_input = gr.Checkbox(label="是否直接对上次合成结果调整语速和音色", value=False)
        speed_input = gr.Slider(minimum=0.5, maximum=2.0, label="语速", value=1.0)
        sample_steps_input = gr.Slider(minimum=1, maximum=50, label="采样步数", value=8)

    with gr.Row():
        generate_button = gr.Button("生成数字人语音与视频")
    
    with gr.Row():
        output_audio = gr.Audio(label="生成的音频")
        output_video = gr.Video(label="生成的视频")
    record_button.click(record_video, outputs=input_video_dropdown)
    generate_button.click(
        generate_speech,
        inputs=[ref_audio_input, text_input, sovits_model_dropdown, gpt_model_dropdown, prompt_text_input, prompt_language_input, how_to_cut_input, top_k_input, top_p_input, temperature_input, ref_free_input, speed_input, if_freeze_input, sample_steps_input, input_video_dropdown],
        outputs=[output_audio, output_video]
    )

demo.launch()
