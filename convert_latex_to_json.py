'''
.txt 파일을 읽어서 (각 세그먼트를) JSON 형식으로 변환.
'''
# 3-convert_latex-to-json.py

import re
import json
import os

def txt_to_json(input_path, output_path, lecture_title):
    segments = []
    # 정규식: (시작 ~ 종료) 텍스트
    pattern = re.compile(r'^\((\d{2}:\d{2}:\d{2}\.\d+)\s*~\s*(\d{2}:\d{2}:\d{2}\.\d+)\)\s*(.+)$')

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                start, end, text = m.groups()
                segments.append({
                    "start_time": start,
                    "end_time": end,
                    "text": text
                })
            else:
                # 타임스탬프가 없는 추가 줄이 있을 경우, 직전 세그먼트에 붙임
                if segments:
                    segments[-1]["text"] += " " + line
                else:
                    segments.append({
                        "start_time": None,
                        "end_time": None,
                        "text": line
                    })

    # 최종 구조: 강의명과 내용 리스트
    output_obj = {
        "lecture_title": lecture_title,
        "lecture_content": segments
    }

    # JSON으로 저장
    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(output_obj, out, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
#     # 원본 파일명과 강의명을 적절히 설정하세요
#     result_dir  = "/data/seungmin/test/inference/result"
#     input_path = os.path.join(result_dir, "emjjk_23_univ-5epoch-latex_real.txt")
#     output_path = os.path.join(result_dir, "emjjk_23_univ_json.json")
#     lecture_title   = "emjjk_23 강의"  # 필요에 따라 변경

#     if not os.path.exists(input_path):
#         print(f"Error: 파일이 없습니다: {input_path}")
#     else:
#         txt_to_json(input_path, output_path, lecture_title)
#         print(f"변환 완료: {output_path}")


