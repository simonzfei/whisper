import os
import whisper
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
import concurrent.futures


cache_dir = "/root/autodl-tmp/whisper/env"
if os.path.exists(cache_dir):
    os.system(f"rm -rf {cache_dir}")

# Create a new cache directory
os.makedirs(cache_dir, exist_ok=True)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",cache_dir="/data/zfei/code/whisper/model_pyannote/")
output_dir = '/data/zfei/code/whisper/results/pyannote-whisper'

def process_audio(file_path):
    model = whisper.load_model("large")
    asr_result = model.transcribe(file_path, initial_prompt="????")
    diarization_result = pipeline(file_path)
    final_result = diarize_text(asr_result, diarization_result)

    output_file = os.path.join(output_dir, os.path.basename(file_path)[:-4] + '.txt')
    with open(output_file, 'w') as f:
        for seg, spk, sent in final_result:
            line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}\n'
            f.write(line)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

wave_dir = '/data/zfei/code/whisper/materials'
# ?????????wav???
wav_files = [os.path.join(wave_dir, file) for file in os.listdir(wave_dir) if file.endswith('.wav')]


# ????wav??
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(process_audio, wav_files)

print('????!')