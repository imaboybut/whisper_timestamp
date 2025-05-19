

pip install torch transformers peft openai konlpy scikit-learn numpy ffmpeg-python soundfile librosa


main.py
프로젝트 메인 스크립트

model.py
Whisper 기반 PEFT LORA 모델과 프로세서를 불러오는 함수 포함

inferece_on.py
입력 영상/음성 파일에서 오디오를 추출하고 Whisper 모델로 구간별 음성 인식 후, 타임스탬프가 포함된 텍스트 형식으로 결과 반환

txt_api_latex.py
OpenAI GPT-4 API를 호출하여 인식 텍스트 내 수식 부분만 LaTeX 문법으로 변환하는 기능 포함

convert_latex_to_json.py
LaTeX 변환이 완료된 텍스트를 읽어 타임스탬프와 함께 JSON 포맷으로 변환 저장

text2video.py
TF-IDF 기반 중요 문장 추출 후, ffmpeg를 활용해 중요 문장 구간만 영상 클립으로 잘라내는 요약 영상 생성 스크립트

