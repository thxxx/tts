# huggingface-cli login
from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="/home/khj6051/tts_sfx/src/weights_1203_prepend_vocos_clone_qwen/model_4.pt",
    path_in_repo="f5tts_clone_qwen_4.pt",
    repo_id="Daniel777/stereo48k",
    repo_type="model"
)