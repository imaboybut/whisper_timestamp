import os
import io
import torch
import numpy as np
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
def inference_on_wav(file_path, model, processor, segment_length_sec=30):
    sampling_rate = 16000

    # .wav 파일 로드
    audio, sr = load_audio_wav(file_path)
    if sr != sampling_rate:
        raise ValueError(f"지원되지 않는 샘플링 레이트: {sr}. {sampling_rate} Hz로 변환해야 합니다.")
    
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)  # 다채널 -> 모노 변환

    num_samples = len(audio)
    segment_samples = segment_length_sec * sampling_rate
    transcripts = []

    print("전체 오디오 길이(샘플):", num_samples)
    print("세그먼트당 샘플 수:", segment_samples)

    # 2) forced_decoder_ids (언어/태스크 태그)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    # 혹시 None/빈 리스트면 model.config.forced_decoder_ids로 대체하거나, 빈 리스트 할당
    if not forced_decoder_ids:
        forced_decoder_ids = model.config.forced_decoder_ids
        if not forced_decoder_ids:
            forced_decoder_ids = []

    # 3) Whisper 타임스탬프 설정
    #   (a) no_timestamps_token_id 확인
    if not hasattr(model.config, "no_timestamps_token_id"):
        no_ts_token = getattr(processor.tokenizer, "no_timestamps_token", "<|notimestamps|>")
        model.config.no_timestamps_token_id = processor.tokenizer.convert_tokens_to_ids(no_ts_token)

    #   (b) 첫 타임스탬프 인덱스 제한 (기본 1, 더 늘려도 됨)
    model.config.max_initial_timestamp_index = 50

    #   (c) suppress_tokens=None or 특수 토큰 제거
    #       필요하다면 model.config.suppress_tokens = [] 로 비울 수도 있음.
    #       (기존 Whisper는 특정 토큰을 억제하도록 설정되어 있을 수 있음)
    # model.config.suppress_tokens = []

    #   (d) 디코딩 길이 여유(기본값 448일 수 있는데, 부족하면 늘려볼 수 있음)
    # model.config.max_length = 1024  # 또는 generate()에서 max_new_tokens=... 지정

    # 4) 세그먼트별 추론
    for start in range(0, num_samples, segment_samples):
        end = min(start + segment_samples, num_samples)
        segment = audio[start:end]

        # 입력
        inputs = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs["input_features"].to(model.device)

        with torch.no_grad():
            # ✅ 여기서 타임스탬프 활성화를 위해:
            #   return_timestamps=True + return_dict_in_generate=True
            outputs = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                # suppress_tokens=None,
                return_dict_in_generate=True,
                return_timestamps=True,
                # max_new_tokens=1024  # 필요시 추가
            )

        # 5) 타임스탬프 모드일 때, WhisperForConditionalGeneration은
        #    outputs.sequences + outputs.segments 를 반환(가능)
        if hasattr(outputs, "segments") and outputs.segments:
            # *문장별* (세그먼트 내) 타임스탬프를 따로 추출
            seg_transcripts = []
            for seginfo in outputs.segments:
                seg_text = seginfo["text"]
                seg_transcripts.append(seg_text)
            segment_text = " ".join(seg_transcripts)
        else:
            # 타임스탬프가 전혀 안 나오면, fallback으로 그냥 디코딩
            sequences = outputs.sequences
            segment_text = processor.tokenizer.decode(sequences[0], skip_special_tokens=True)

        transcripts.append(segment_text)
        print(f"Segment {start // segment_samples + 1} 완료. 텍스트: {segment_text}")

    full_transcription = "\n".join(transcripts)
    return full_transcription

def inference_on_file(file_path, model, processor, segment_length_sec=30):
    """
    주어진 오디오 파일(MP4/WAV)을 Whisper 모델을 사용하여 문장별 타임스탬프를 포함한 텍스트로 변환합니다.
    """
    sampling_rate = 16000
    audio, sr = load_audio_ffmpeg(file_path, target_sr=sampling_rate)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)  # 다채널 -> 모노 변환

    num_samples = len(audio)
    segment_samples = segment_length_sec * sampling_rate
    full_transcript = []

    print("전체 오디오 길이(샘플):", num_samples)
    print("세그먼트당 샘플 수:", segment_samples)

    # 모델에 강제 디코딩 설정 적용 (한글로 설정)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    print("forced_decoder_ids:", forced_decoder_ids)  # 리스트 길이와 내용 확인
    # 2) None이거나 빈 리스트면 최소한 빈 리스트라도 지정 (에러 방지)
    if not forced_decoder_ids:
        forced_decoder_ids = []
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.forced_decoder_ids = forced_decoder_ids
    # 모델의 타임스탬프 관련 설정이 없을 경우 추가
    if not hasattr(model.generation_config, "no_timestamps_token_id"):
        no_ts_token = getattr(processor.tokenizer, "no_timestamps_token", "<|notimestamps|>")
        model.generation_config.no_timestamps_token_id = processor.tokenizer.convert_tokens_to_ids(no_ts_token)
    model.generation_config.max_initial_timestamp_index = 10

    for start in range(0, num_samples, segment_samples):
        end = min(start + segment_samples, num_samples)
        segment = audio[start:end]
        offset = start / sampling_rate  # 현재 세그먼트의 시작 시간(초)


        if len(segment) < segment_samples:
            segment = pad_audio_to_target_length(segment, segment_length_sec, sampling_rate)


        # 입력 생성
        inputs = processor(segment, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_features = inputs["input_features"].to(model.device)

        # Whisper 모델을 통한 추론 수행 (타임스탬프 활성화)
        with torch.no_grad():
            outputs = model.generate(
                input_features,

                return_dict_in_generate=True,
                return_timestamps=True
            )
            
                # 🚨 디버깅: 모델 출력 확인
        print("🔹 모델 출력:", outputs)

        if hasattr(outputs, "segments"):
            print(f"🔹 출력된 세그먼트 개수: {len(outputs.segments)}")
        else:
            print("❌ `outputs.segments`가 존재하지 않음!")

        # 타임스탬프 및 텍스트 추출
        if hasattr(outputs, "segments") and outputs.segments:
            for seg in outputs.segments:
                start_time = seg['start'] + offset
                end_time = seg['end'] + offset
                transcript = f"[{start_time:.2f}s - {end_time:.2f}s] {seg['text']}"
                full_transcript.append(transcript)

        print(f"Segment {start // segment_samples + 1} 완료.")

    # 전체 타임스탬프 포함된 텍스트 반환
    full_transcription = "\n".join(full_transcript)
    return full_transcription