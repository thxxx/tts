apt-get update
apt-get install -y ffmpeg
git config --global user.email zxcv05999@naver.com
git config --global user.name thxxx
apt-get install -y libsndfile1-dev
pip install -e .
pip install supabase
pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
pip install git+https://github.com/descriptinc/audiotools
pip install jiwer ffprobe ffmpeg-python

pip uninstall -y torchvision
pip install -U transformers
