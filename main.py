import os
import time
import re
import openai
from model import load_model_and_processor
from inferece_on import inference_on
from txt_api_latex import convert_math_to_latex
from convert_latex_to_json import txt_to_json
from text2video import made_yo_vid

def process_video(input_video: str, result_dir: str , made_flag = False, max_minute = 1):
    base_name = os.path.splitext(os.path.basename(input_video))[0]

    # 1. Inference 수행
    print(f"[1/3] {base_name}: Inference 시작...")
    start = time.perf_counter()
    model, processor = load_model_and_processor()
    transcription = inference_on(input_video, model, processor, segment_length_sec=30)
    inference_path = os.path.join(result_dir, f"{base_name}.txt")
    with open(inference_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"Inference 결과 저장: {inference_path} ({time.perf_counter() - start:.2f} 초)")

    # 2. 수식 LaTeX 변환
    print(f"[2/3] {base_name}: LaTeX 변환 시작...")
    start = time.perf_counter()
    latex_output = os.path.join(result_dir, f"{base_name}-latex_real.txt")
    with open(inference_path, "r", encoding="utf-8") as f:
        content = f.read()
    segments = re.split(r"(?=\(\d{2}:\d{2}:\d{2}\.\d+\s*~)", content)
    with open(latex_output, "w", encoding="utf-8") as out:
        for seg in segments:
            if not seg.strip():
                continue
            converted = convert_math_to_latex(seg)
            out.write(converted + "\n\n")
    print(f"LaTeX 변환 결과 저장: {latex_output} ({time.perf_counter() - start:.2f} 초)")

    if made_flag == True:
            video_filename_with_ext = os.path.basename(input_video) # 원본 비디오 파일 이름 (예: "emjjk_23.mp4")
            video_base_name, video_extension = os.path.splitext(video_filename_with_ext) # ("emjjk_23", ".mp4")

            new_output_filename = f"{video_base_name}_{max_minute}분_요약영상{video_extension}" # 예: "emjjk_23_1분_요약영상.mp4"

            # 요약 영상 저장 경로를 result_dir 기준으로 설정
            final_output_video_path = os.path.join(result_dir, new_output_filename) 

            yo_vid_text_filename = f"{base_name}_yo_vid_output.txt"
            full_yo_vid_text_path = os.path.join(result_dir, yo_vid_text_filename)

            made_yo_vid(max_minute = max_minute, input_path = latex_output , input_video_path=input_video,
                        output_path =full_yo_vid_text_path,
                        output_video_path=final_output_video_path)


    # 3. JSON 변환
    print(f"[3/3] {base_name}: JSON 변환 시작...")
    start = time.perf_counter()
    json_output = os.path.join(result_dir, f"{base_name}.json")
    txt_to_json(latex_output, json_output, lecture_title=base_name)
    print(f"JSON 변환 결과 저장: {json_output} ({time.perf_counter() - start:.2f} 초)")


def main():
    # 경로 설정
    input_dir = "/data/jinwoo/target_data"
    result_dir = "/data/seungmin/test/inference/result"
    os.makedirs(result_dir, exist_ok=True)

    # OpenAI API 키
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # 모든 MP4 파일 처리
    mp4_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    if not mp4_files:
        print("처리할 MP4 파일이 없습니다.")
        return

    for mp4 in mp4_files:
        input_video = os.path.join(input_dir, mp4)
        process_video(input_video, result_dir,made_flag = True, max_minute = 1)

if __name__ == "__main__":
    main()
