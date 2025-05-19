'''
오디오 파일을 로드하고 이를 세그먼트로 나누어 --> 모델이 추론을 수행할 수 있도록 함.
추론된 텍스트는 타임스탬프와 함께 formatted된 txt로 반환되어 저장됨
'''

# 1-inferece_on(wav-to-txt).py
import torch
import numpy as np
import re
import io
import ffmpeg
import soundfile as sf
from typing import List, Dict

def load_audio(file_path: str, target_sr: int = 16000) -> (np.ndarray, int):
    """
    .wav 는 soundfile, 그 외(mp4 등)는 ffmpeg → wav 파이프라인으로 mono 16kHz 로딩
    """
    if file_path.lower().endswith(".wav"):
        audio, sr = sf.read(file_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    else:
        out, _ = (
            ffmpeg.input(file_path)
                  .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=target_sr)
                  .run(capture_stdout=True, capture_stderr=True)
        )
        audio, sr = sf.read(io.BytesIO(out))

    # Resample 필요 시 librosa 사용
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def split_sentences_with_timestamps(
    text: str, seg_start: float, seg_end: float, chunk_size: int = 10
) -> List[Dict]:
    # (원본 로직 그대로 복사)
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    if len(sentences) == 1:
        words = text.split()
        if len(words) > chunk_size:
            sentences = [
                ' '.join(words[i:i+chunk_size])
                for i in range(0, len(words), chunk_size)
            ]
    if not sentences or sentences == ['']:
        return []

    total_length = sum(len(s) for s in sentences) or 1
    seg_duration = seg_end - seg_start

    results = []
    cumulative = 0
    for s in sentences:
        s_len = len(s)
        start = seg_start + (cumulative / total_length) * seg_duration
        cumulative += s_len
        end = seg_start + (cumulative / total_length) * seg_duration
        results.append({"start": start, "end": end, "text": s})
    return results

def inference_on(
    file_path: str,
    model,
    processor,
    segment_length_sec: int = 30
) -> str:
    sampling_rate = 16000
    audio, sr = load_audio(file_path, target_sr=sampling_rate)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    num_samples = len(audio)
    segment_samples = segment_length_sec * sampling_rate

    # forced_decoder_ids 세팅
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    if not forced_decoder_ids:
        forced_decoder_ids = model.config.forced_decoder_ids

    sentence_results = []
    for start in range(0, num_samples, segment_samples):
        end = min(start + segment_samples, num_samples)
        segment = audio[start:end]

        # WhisperProcessor 호출 (feature_extractor)
        inputs = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(model.device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2
            )

        # 디코딩
        transcription = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

        # 타임스탬프 분할
        seg_start = start / sampling_rate
        seg_end = end / sampling_rate
        sentence_results += split_sentences_with_timestamps(transcription, seg_start, seg_end)

    # 최종 문자열 포맷팅
    lines = [
        f"({format_time(s['start'])} ~ {format_time(s['end'])}) {s['text']}"
        for s in sentence_results
    ]
    return "\n".join(lines)
