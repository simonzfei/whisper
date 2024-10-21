import whisper
import arrow

# ????????????????
def excute(model_name,file_path,start_time):
    model = whisper.load_model(model_name)
    result = model.transcribe(file_path)
    for segment in result["segments"]:
        now = arrow.get(start_time)
        start = now.shift(seconds=segment["start"]).format("YYYY-MM-DD HH:mm:ss")
        end = now.shift(seconds=segment["end"]).format("YYYY-MM-DD HH:mm:ss")
        print("?"+start+"->" +end+"?:"+segment["text"])

if __name__ == '__main__':
    excute("base","1001.mp3","2022-10-24 16:23:00")

