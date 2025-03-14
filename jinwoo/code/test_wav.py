import os
import json

def collect_transcriptions(txt_dir, json_dir, output_txt_from_txt, output_txt_from_json):
    txt_transcriptions = []
    json_transcriptions = []
    
    # TXT 파일 처리
    for file_name in sorted(os.listdir(txt_dir)):
        if file_name.startswith("U00") and file_name.endswith(".txt"):
            file_path = os.path.join(txt_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                txt_transcriptions.append(f.read().strip())
    
    # JSON 파일 처리
    for file_name in sorted(os.listdir(json_dir)):
        if file_name.startswith("U00") and file_name.endswith(".json"):
            file_path = os.path.join(json_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                transcription = data.get("06_transcription", {}).get("1_text", "")
                if transcription:
                    json_transcriptions.append(transcription)
    
    # 병합된 TXT 파일 저장
    with open(output_txt_from_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_transcriptions))
    print(f"TXT 추론 결과가 {output_txt_from_txt}에 저장되었습니다.")
    
    with open(output_txt_from_json, "w", encoding="utf-8") as f:
        f.write("\n".join(json_transcriptions))
    print(f"JSON 전사 결과가 {output_txt_from_json}에 저장되었습니다.")


def main():
    txt_dir = "/data/jinwoo/result_trainend/"
    json_dir = "/data/seungmin/training/labels/eng/arch/C01750/"
    output_txt_from_txt = "/data/jinwoo/merged_transcriptions.txt"
    output_txt_from_json = "/data/seungmin/merged_json_transcriptions.txt"
    
    collect_transcriptions(txt_dir, json_dir, output_txt_from_txt, output_txt_from_json)

if __name__ == "__main__":
    main()