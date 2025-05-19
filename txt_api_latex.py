'''
.txt 파일을 읽고 수식 부분을 LaTeX 문법으로 변환.(txt-to-txt)
'''

# 2-txt-to-txt-with-latex.py
import os
import re
import openai
import time


# 5) 변환 함수 정의
def convert_math_to_latex(text_segment: str) -> str:
    prompt = (
        "다음 강의 스크립트에서 수식 부분만 찾아 모두 올바른 LaTeX 문법으로 바꿔. 그리고 만약 스크립트안에 수식이 없으면 그대로 출력을 보내줘."
        "너의 출력 형식은 입력 그대로에서 수식만 변환된 출력이여야 해. 만약 스크립트에 수식이 없으면 원본 상태 그대로 돌려줘." \
        "수정 전 수정 후 둘다 보여주지 말고 무조건 수정 후 내용만 (time) '수정내용' 이렇게만 보내줘  "
        "타임스탬프와 한글 나머지 텍스트는 그대로 두되, 수식은 `$...$` 또는 `$$...$$` 형태로 표기해줘.\n\n"
        f"{text_segment}"
    )
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that converts spoken math into LaTeX."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0,
        max_tokens=1500,
    )
    return resp.choices[0].message.content


