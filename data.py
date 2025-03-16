import os
import json
from datasets import Dataset, Audio

def load_data():
    # 라벨 JSON 파일 경로 (쉼표 누락 없이 작성)
    label_dirs = [
        "/data/seungmin/training/labels/eng/arch/",
        "/data/seungmin/training/labels/eng/comp/",
        "/data/seungmin/training/labels/eng/elec/",
        "/data/seungmin/validation/labels/eng/arch/",
        "/data/seungmin/validation/labels/eng/comp/",
        "/data/seungmin/validation/labels/eng/elec/"
    ]
    
    data = []
    
    for label_dir in label_dirs:
        # 라벨 경로에 따라 오디오 베이스 경로 결정
        if "training" in label_dir:
            audio_base = "/data/seungmin/training/"
        elif "validation" in label_dir:
            audio_base = "/data/seungmin/validation/"
        else:
            audio_base = ""
            
        # 해당 디렉토리 내 모든 JSON 파일 순회
        for root, _, files in os.walk(label_dir):
            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                    # 음성 길이 확인 (문자열을 실수형으로 변환)
                    speech_length_str = json_data.get("01_dataset", {}).get("9_speech_length", "0")
                    try:
                        speech_length = float(speech_length_str)
                    except ValueError:
                        speech_length = 0.0
                    
                    # 음성 길이가 10초 이상이면 해당 샘플 건너뛰기
                    if speech_length >= 10.0:
                        continue
                    # 마지막 스크립트(06_transcription)만 사용
                    transcription = json_data.get("06_transcription", {}).get("1_text", "")
                    # 음성 파일 경로는 01_dataset의 "3_src_path"에 기록됨
                    src_path = json_data.get("01_dataset", {}).get("3_src_path", "")
                    audio_full_path = os.path.join(audio_base, src_path)
                    
                    data.append({
                        "audio": audio_full_path,
                        "transcript": transcription
                    })
    
    # 데이터 리스트를 HuggingFace Dataset으로 변환
    dataset = Dataset.from_list(data)
    # "audio" 컬럼을 Audio 타입으로 캐스팅 (샘플링 레이트: 16kHz)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # 상위 10개 샘플만 선택하여 반환 나중에 주석처리 *********
    # dataset = dataset.select(range(10))
    #나중에 주석처리 *********
    
    return dataset