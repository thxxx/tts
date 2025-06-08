mkdir ckpts
wget https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/model_1200000.pt?download=true
wget https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/vocab.txt?download=true
mv model_1200000.pt?download=true ckpts/model_1200000.pt
mv vocab.txt?download=true ckpts/vocab.txt
git clone https://github.com/NVIDIA/BigVGAN
mv BigVGAN src/third_party/BigVGAN