

    

import re
import subprocess
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def made_yo_vid(max_minute = 1, input_path = "" ,input_video_path="", output_path = "", output_video_path=""):
    # 1. 파일 읽기
    file_path = input_path
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. 문장 및 타임스탬프 추출
    pattern = r"\((\d{2}:\d{2}:\d{2}\.\d{3}) ~ (\d{2}:\d{2}:\d{2}\.\d{3})\)\s*(.*)"
    matches = re.findall(pattern, text)

    # 3. 타임스탬프 변환 함수
    def time_to_seconds(t):
        h, m, s = t.split(":")
        sec, ms = s.split(".")
        return int(h)*3600 + int(m)*60 + int(sec) + int(ms)/1000

    def seconds_to_time(secs):
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        ms = int((secs - int(secs)) * 1000)
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"

    # 4. 문장 정리
    data = []
    sentences_only = []
    for start, end, sentence in matches:
        start_sec = time_to_seconds(start)
        end_sec = time_to_seconds(end)
        sentence = sentence.strip()
        data.append({
            "start_time": start,
            "end_time": end,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "sentence": sentence
        })
        sentences_only.append(sentence)

    # 5. 형태소 기반 tokenizer 정의 (TF-IDF용)
    okt = Okt()
    stopwords = ['엑스','이', '그', '저', '것', '수', '좀', '등', '때', '말', '같', '더', '듯', '우리', '여러분','엑스']
    def tokenize(text):
        tokens = okt.nouns(text)
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    # 6. TF-IDF 계산
    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    tfidf_matrix = vectorizer.fit_transform(sentences_only)
    tfidf_scores = tfidf_matrix.sum(axis=1).A1  # 문장별 TF-IDF 총합

    # 7. 중요도 점수 추가
    for i, item in enumerate(data):
        length_score = len(item["sentence"]) / 100
        item["score"] = tfidf_scores[i] + length_score

    # 8. 중요 문장 Top 1
    data_sorted = sorted(data, key=lambda x: x["score"], reverse=True)
    top1 = data_sorted[:1]
    print(top1)

    # 9. 사용자 입력된 시간 (분)으로 앞뒤 문장 선택 및 총 길이 계산
    def create_video_blocks(top1, max_time_minutes):
        max_time_seconds = max_time_minutes * 60  # 최대 영상 시간 (초)
        final_blocks = []
        
        for idx, top in enumerate(top1, 1):
            center_idx = next(i for i, d in enumerate(data) if d["start_time"] == top["start_time"])
            
            # 초기 상태
            total_time = 0
            start_idx = center_idx
            end_idx = center_idx
            block_sentences = [data[center_idx]]  # 중요 문장 포함
            
            # 앞쪽 문장 추가
            while start_idx > 0:
                new_start_time = data[start_idx - 1]["start_sec"]
                new_end_time = data[start_idx - 1]["end_sec"]
                # 총 시간 계산
                new_total_time = data[end_idx]["end_sec"] - new_start_time
                if new_total_time <= max_time_seconds:
                    block_sentences.insert(0, data[start_idx - 1])
                    start_idx -= 1
                    total_time = new_total_time  # 시간 업데이트
                else:
                    break  # 시간을 초과하면 더 이상 추가하지 않음
            
            # 뒤쪽 문장 추가
            while end_idx < len(data) - 1:
                new_start_time = data[end_idx + 1]["start_sec"]
                new_end_time = data[end_idx + 1]["end_sec"]
                # 총 시간 계산
                new_total_time = new_end_time - data[start_idx]["start_sec"]
                if new_total_time <= max_time_seconds:
                    block_sentences.append(data[end_idx + 1])
                    end_idx += 1
                    total_time = new_total_time  # 시간 업데이트
                else:
                    break  # 시간을 초과하면 더 이상 추가하지 않음
            
            block_start_time = block_sentences[0]["start_time"]
            block_end_time = block_sentences[-1]["end_time"]
            block_text = "\n".join([f"{item['sentence']}" for item in block_sentences])
            
            final_blocks.append(
                f"Top {idx} 문맥 블록\n[{block_start_time} ~ {block_end_time}]\n\n{block_text}"
            )

        return final_blocks

    # 10. 결과 생성
    #input max time
    
    max_time_minutes =  max_minute
    final_blocks = create_video_blocks(top1, max_time_minutes)
    full_context_output = "\n\n\n".join(final_blocks)

    # 11. 결과 저장
    output_path =  output_path
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_context_output)

    print(f"{max_time_minutes}분 txt 저장 완료: {output_path}")


    input_video = input_video_path
    input_txt = output_path
    output_video = output_video_path

    # 1. 중요문장.txt에서 시간 구간 추출 (하나의 문장만 포함됨)
    def extract_clip_times_from_txt(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 수정된 정규식: 초는 최대 2자리로 맞추기
        pattern = r"\[(\d{2}:\d{2}:\d{2}\.\d{3}) ~ (\d{2}:\d{2}:\d{2}\.\d{3})\]"
        return re.findall(pattern, text)

    # 2. 영상 자르기 함수
    def cut_video_ffmpeg_with_fade(input_file, start_time, end_time, output_file, fade_duration=3):
        # start_time과 end_time을 초 단위로 변환
        start_sec = time_to_seconds(start_time)
        end_sec = time_to_seconds(end_time)
        print("end sec = ", end_sec)
        
        # 페이드 효과를 적용하려면 "-vf" 필터에 fade 필터를 사용합니다.
        cmd = [
            "ffmpeg",
            "-y",  # 기존 파일 덮어쓰기
            "-ss", str(start_sec),  # 시작 시간
            "-to", str(end_sec),  # 종료 시간
            "-i", input_file,  # 입력 파일
            "-vf", f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={end_sec - start_sec - fade_duration}:d={fade_duration}",
            "-af", f"afade=t=in:st=0:d={fade_duration},afade=t=out:st={end_sec - start_sec - fade_duration}:d={fade_duration}",        
            "-c:a", "aac",  # 오디오 코덱
            "-c:v", "libx264",  # 비디오 코덱
            output_file  # 출력 파일
        ]
        subprocess.run(cmd, check=True)

    # 4. 전체 프로세스 실행 (하나의 문장만 처리)


    clip_times = extract_clip_times_from_txt(input_txt)

    start, end = clip_times[0]  # 하나의 문장의 시간 구간
    out_name = "clip1.mp4"
    cut_video_ffmpeg_with_fade(input_video, start, end, out_name)

    # 결과 영상 저장
    subprocess.run(["mv", out_name, output_video])  # 자른 파일을 최종 파일명으로 변경

    print(f"영상 편집 완료 -> {output_video}")
    print("자른 구간:", clip_times[0])