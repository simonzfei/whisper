#Model Download
from modelscope import snapshot_download
model_dir = snapshot_download('mirror013/speaker-diarization-3.1')
print(model_dir)