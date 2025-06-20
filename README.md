# Meta_Digital-Human

本项目实现了一个基于 GPT-SoVITS + Wav2Lip 的数字人语音合成与唇动视频生成系统，通过 Gradio 构建 Web UI 界面，支持上传参考音频和输入文本，实时合成对应语音，并将其与输入视频进行唇形同步，生成数字人短视频。

推荐使用以下命令为Wav2Lip创建环境：

    conda create -n wav2lip python=3.8
    conda activate wav2lip
    pip install -r Wav2Lip/requirements.txt

创建本项目环境：

    pip install -r requirements.txt

启动UI：

    python webui_LST.py
