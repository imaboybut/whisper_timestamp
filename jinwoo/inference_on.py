import os
import io
import torch
import numpy as np
import ffmpeg
import soundfile as sf
import re
import torch
import numpy as np
import re
import io
import ffmpeg
import soundfile as sf
def pad_audio_to_target_length(audio, target_length, sampling_rate):
    """오디오 길이를 Whisper 모델의 기대 길이(30초)로 패딩하는 함수"""
    current_length = len(audio)
    target_samples = target_length * sampling_rate  # 30초 길이에 해당하는 샘플 수

    if current_length < target_samples:
        pad_width = target_samples - current_length
        audio = np.pad(audio, (0, pad_width), mode='constant', constant_values=0)

    return audio
def load_audio_wav(file_path):
    """
    soundfile을 이용해 .wav 파일을 numpy 배열과 샘플링 레이트로 변환합니다.
    """
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # 다채널 오디오를 mono로 변환
    return audio, sr
def load_audio_ffmpeg(file_path, target_sr=16000):
    """
    ffmpeg를 사용하여 mp4 파일에서 오디오를 wav 포맷으로 변환하고,
    soundfile을 이용해 numpy 배열과 샘플링 레이트를 반환합니다.
    """
    try:
        # ffmpeg로 오디오를 wav 형식으로 파이프 출력, mono로 변환
        out, err = (
            ffmpeg
            .input(file_path)
            .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=target_sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print("FFmpeg error:", e)
        raise e

    # 메모리에서 wav 데이터를 읽어 numpy array로 변환
    audio, sr = sf.read(io.BytesIO(out))
    return audio, sr

def format_time(seconds: float) -> str:
    """
    초 단위 float 값을 WebVTT 형식의 문자열 (HH:MM:SS.mmm)로 변환.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    
def split_sentences_with_timestamps(text: str, seg_start: float, seg_end: float, chunk_size: int = 10):
    """
    주어진 텍스트를 먼저 문장 단위로 분리하고, 
    만약 문장 구분 기호가 없어서 단 한 문장이라면 단어 수(chunk_size) 기준으로 강제 분할합니다.
    이후 전체 세그먼트 시간(초)을 각 부분의 문자 길이 비율로 분할하여 타임스탬프를 할당합니다.
    """
    # 우선 문장 구분 기호(마침표, 물음표, 느낌표 뒤 공백) 기준으로 분리
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    # 만약 분리 결과가 단 한 문장이면, 단어 수 기준으로 강제 분할
    if len(sentences) == 1:
        words = text.split()
        if len(words) > chunk_size:
            sentences = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    if not sentences or sentences == ['']:
        return []
    
    # 전체 문자 길이를 기준으로 각 문장의 비율로 시간 분할 (비율 계산)
    total_length = sum(len(s) for s in sentences)
    if total_length == 0:
        total_length = len(sentences)
    
    seg_duration = seg_end - seg_start
    sentence_boundaries = []
    cumulative = 0
    for s in sentences:
        s_len = len(s)
        start_time = seg_start + (cumulative / total_length) * seg_duration
        cumulative += s_len
        end_time = seg_start + (cumulative / total_length) * seg_duration
        sentence_boundaries.append({
            "start": start_time,
            "end": end_time,
            "text": s
        })
    return sentence_boundaries

def inference_on(file_path, model, processor, segment_length_sec=30):
    """
    Whisper 모델을 사용해 오디오를 텍스트로 변환한 후,
    세그먼트 내에서도 문장 단위로 타임스탬프와 함께 텍스트를 하나의 문자열로 리턴합니다.
    """
    sampling_rate = 16000

    # 오디오 로드 (wav 파일이면 load_audio_wav, 그렇지 않으면 load_audio_ffmpeg 사용)
    if file_path.endswith(".wav"):
        audio, sr = load_audio_wav(file_path)
    else:
        audio, sr = load_audio_ffmpeg(file_path, target_sr=sampling_rate)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
        
    num_samples = len(audio)
    segment_samples = segment_length_sec * sampling_rate

    print(f"Inference 시작: {file_path}")
    print("전체 오디오 길이(샘플):", num_samples)
    print("세그먼트당 샘플 수:", segment_samples)

    # forced_decoder_ids 생성
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    if not forced_decoder_ids or len(forced_decoder_ids) == 0:
        print("processor.get_decoder_prompt_ids() returned empty. Using model.config.forced_decoder_ids instead.")
        forced_decoder_ids = model.config.forced_decoder_ids

    sentence_results = []
    seg_index = 1
    for start in range(0, num_samples, segment_samples):
        end = min(start + segment_samples, num_samples)
        segment = audio[start:end]

        inputs = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs["input_features"].to(model.device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                suppress_tokens=None
            )
        transcription = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

        # 세그먼트의 시작/끝 시간 계산 (초 단위)
        seg_start = start / sampling_rate
        seg_end = end / sampling_rate

        print(f"Segment {seg_index} 완료. ({format_time(seg_start)} ~ {format_time(seg_end)})")
        seg_index += 1

        # 문장 단위로 분리 및 세부 타임스탬프 할당
        sentences = split_sentences_with_timestamps(transcription, seg_start, seg_end)
        sentence_results.extend(sentences)

    # 문장별로 "(시작 ~ 종료) 문장" 형태로 문자열 조합
    lines = []
    for sent in sentence_results:
        st = format_time(sent["start"])
        en = format_time(sent["end"])
        txt = sent["text"]
        lines.append(f"({st} ~ {en}) {txt}")

    final_transcription = "\n".join(lines)
    print("\nInference 결과:")
    print(final_transcription)

    return final_transcription