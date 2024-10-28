#Model Download
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/whisper-large-v3')
print(model_dir)