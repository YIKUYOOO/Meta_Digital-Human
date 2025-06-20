from gradio_client import Client, file
import requests
import os
import ffmpeg
import shutil

# load GPT-SoVITS 
GPT_SOVITS_API_URL = "http://localhost:9872/"

# 初始化 API 客户端
client = Client(GPT_SOVITS_API_URL)

#获取 GPT-SoVITS 服务器上可用的 SoVITS 语音模型和 GPT 语音模型
def get_available_models():

    result = client.predict(api_name="/change_choices")

    if not isinstance(result, tuple) or len(result) < 2:
        raise ValueError(f" API 返回格式不正确: {result}\n")

    # 解析 SoVITS 语音模型
    sovits_models = [model[0] for model in result[0]["choices"]]
    
    # 解析 GPT 语音模型
    gpt_models = [model[0] for model in result[1]["choices"]]

    print("🔹 可用的 SoVITS 语音模型:", sovits_models)
    print("🔹 可用的 GPT 语音模型:", gpt_models)

    return sovits_models, gpt_models

#选择 SoVITS 和 GPT 语音模型
def change_models(sovits_model, gpt_model):

    client.predict(
        sovits_path=sovits_model,
        prompt_language="中文",
        text_language="中文",
        api_name="/change_sovits_weights"
    )
    print(f"已切换 SoVITS 语音模型: {sovits_model}\n")

    client.predict(
        gpt_path=gpt_model,
        api_name="/change_gpt_weights"
    )
    print(f"已切换 GPT 语音模型: {gpt_model}\n")


#通过 GPT-SoVITS API 生成语音
def generate_speech(ref_audio_path, text, output_audio_path):

    print("正在向 GPT-SoVITS 发送 TTS 请求...\n\n")
    
    result = client.predict(
        ref_wav_path=file(ref_audio_path),
        prompt_text="你好，我是你的数字人助手！",
        prompt_language="中文",
        text=text,
        text_language="中文",
        how_to_cut="凑四句一切",
        top_k=15,
        top_p=1,
        temperature=1,
        ref_free=False,
        speed=1,
        if_freeze=False,
        inp_refs=None,
        sample_steps="8",
        api_name="/get_tts_wav"
    )

    print(f"API 返回的音频路径: {result}\n")

    # 检查 API 返回路径是否真实存在
    if not os.path.exists(result):
        raise ValueError(f"API 生成的音频文件不存在: {result}\n")

    # 复制到目标路径
    shutil.copy(result, output_audio_path)
    print(f"语音合成完成: {output_audio_path}\n")

    return output_audio_path

#运行测试
if __name__ == "__main__":
    # 获取模型列表
    sovits_models, gpt_models = get_available_models()

    # 选择模型
    selected_sovits_model = sovits_models[1]
    selected_gpt_model = gpt_models[1]

    # 切换模型
    change_models(selected_sovits_model, selected_gpt_model)

    # 生成语音
    ref_audio_path =r"E:\Project\Digital_Human\audio (1).wav"
    text_to_speak = "你好，我是你的数字人助手！"
    output_audio_path = "E:\\Project\\Digital_Human\\generated.wav"

    generate_speech(ref_audio_path, text_to_speak, output_audio_path)
