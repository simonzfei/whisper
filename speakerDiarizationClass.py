# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/24 下午8:17
@Auth ： simonzfei
@File ：speakerDiarizationClass.py
@IDE ：PyCharm
@Motto：thinking coding 
"""


import os
import whisper
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
import concurrent.futures


class WhisperDiarization():
    def __init__(self, model_path, diarization_cache_dir, output_dir, wave_dir):
        self.model_path = model_path
        self.output_dir = output_dir
        self.wave_dir = wave_dir

        # Load Whisper model
        # self.whisper_model = whisper.load_model(self.model_path)

        # Load  model
        self.pyannote_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            cache_dir=diarization_cache_dir
        )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_audio(self, file_path):
        # whisper_model = whisper.load_model(self.model_path)
        whisper_model = whisper.load_model("large-v3-turbo")

        print(f"{'-' * 20}'starting process' {'-' * 20} ")
        # Transcribe
        # asr_result = self.whisper_model.transcribe(file_path, initial_prompt="语音转换")
        #asr_result = whisper_model.transcribe(file_path)
        asr_result = whisper_model.transcribe(file_path, initial_prompt="语音转换")
        print(f"asr_result:\nType: {type(asr_result)}\nContent: {asr_result}")

        #  speaker diarization
        diarization_result = self.pyannote_pipeline(file_path)

        print(f"diarization_result:\nType: {type(diarization_result)}\nContent: {diarization_result}")

        # Combine Whisper transcription and PyAnnote diarization results
        final_result = diarize_text(asr_result, diarization_result)

        # Save results
        output_file = os.path.join(self.output_dir, os.path.basename(file_path)[:-4] + '.txt')
        with open(output_file, 'w') as f:
            for seg, spk, sent in final_result:
                line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}\n'
                f.write(line)

    def process_all_files(self, max_workers=3):
        # wav_files = [os.path.join(self.wave_dir, file) for file in os.listdir(self.wave_dir) if file.endswith('.wav')]
        wav_files = [os.path.join(self.wave_dir, file) for file in os.listdir(self.wave_dir) ]
        print(wav_files)
        # Process each audio file using multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_audio, wav_files)

        print('处理完成！')


# Example usage
if __name__ == "__main__":
    model_path = "/home/f/.cache/whisper/tiny.pt"
    #model_path = "/home/f/.cache/whisper/medium.pt"
    #model_path = "large.pt"

    #diarization_cache_dir = "/media/f/E/project/git/whisper/restults/sd_model"
    #output_dir = "/media/f/E/project/git/whisper/restults/speaker_diarization_large_wavmp31025"
    #wave_dir = "/media/f/E/project/git/whisper/audio" /media/f/E/project/git/whisper/wavs/test

    #diarization_cache_dir = "/media/f/E/project/git/whisper/restults/sd_model"
    #output_dir = "/media/f/E/project/git/whisper/restults/speaker_diarization_large_1025-mp3Towav"
    #wave_dir = "/media/f/E/project/git/whisper/wavs"


    diarization_cache_dir = "/media/f/E/project/git/whisper/restults/sd_model"
    output_dir = "/media/f/E/project/git/whisper/restults/speaker_diarization_large-v3-turbo_1025"
    wave_dir = "/media/f/E/project/git/whisper/wavs"

    wd = WhisperDiarization(model_path, diarization_cache_dir, output_dir, wave_dir)
    wd.process_all_files(max_workers=1)
