import os
import io
import torch
import numpy as np
import ffmpeg
import soundfile as sf
from transformers import TrainingArguments, Trainer
from data import load_data
from model import load_model_and_processor

from inference_on import inference_on_file
from inference_on import inference_on_wav 

import torch



def preprocess_function(batch, processor):  
    audio_inputs = processor(batch["audio"]["array"], sampling_rate=16000)
    features = audio_inputs["input_features"]
    if isinstance(features, list):
        features = np.array(features)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    labels = processor.tokenizer(batch["transcript"]).input_ids
    return {"input_features": features, "labels": labels}

def get_data_collator(processor):
    def collator(features):
        input_features = []
        for f in features:
            feat = f["input_features"]
            if isinstance(feat, list):
                feat = np.array(feat)
            if feat.ndim == 3 and feat.shape[0] == 1:
                feat = feat.squeeze(0)
            input_features.append(feat)
        labels = [f["labels"] for f in features]
        batch_inputs = processor.feature_extractor.pad({"input_features": input_features}, return_tensors="pt")
        batch_labels = processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt")
        batch_inputs["labels"] = batch_labels["input_ids"]
        return batch_inputs
    return collator



def main():
    # 1. 데이터셋 및 모델, 프로세서 로드
    # dataset = load_data() #d이건 학습할때 사용 
    
    model, processor = load_model_and_processor()
    inference_file = "/data/jinwoo/target_data/acadj_11.mp4"
    inference_file =  "/data/seungmin/training/source/eng/arch/C01750/U00000.wav"
    print("Inference 시작:", inference_file)
    full_transcription = inference_on_file(inference_file, model, processor, segment_length_sec=30)
    print("Inference 결과:")
    print(full_transcription)
    
    # result_dir = "/data/jinwoo/result_trainend"
    # os.makedirs(result_dir, exist_ok=True)
    # result_file = os.path.join(result_dir, "acadj_11_all_trained_timestamp.txt")
    # with open(result_file, "w", encoding="utf-8") as f:
    #     f.write(full_transcription)
    # print(f"Inference 결과가 {result_file} 에 저장되었습니다.")
    
    
    
    
    
    #### 여기 아래는 AI hub 데이터셋 실험 ####
    
    # model, processor = load_model_and_processor()
    # input_dir = "/data/seungmin/training/source/eng/arch/C01750/"
    # result_dir = "/data/jinwoo/result_trainend"
    # os.makedirs(result_dir, exist_ok=True)
    
    # for file_name in os.listdir(input_dir):
    #     if file_name.endswith(".wav"):
    #         file_path = os.path.join(input_dir, file_name)
    #         print("Inference 시작:", file_path)
    #         full_transcription = inference_on_wav(file_path, model, processor, segment_length_sec=30)
    #         print("Inference 결과:")
    #         print(full_transcription)
            
    #         result_file = os.path.join(result_dir, f"{file_name}_transcription.txt")
    #         with open(result_file, "w", encoding="utf-8") as f:
    #             f.write(full_transcription)
    #         print(f"Inference 결과가 {result_file} 에 저장되었습니다.")
    
    #### 여기 위는 AI hub 데이터셋 실험 ####
    
if __name__ == "__main__":
    main()






#학습 코드 짜본것 
    # print("상위 5개 데이터 샘플:")
    # for i, sample in enumerate(dataset.select(range(5))):
    #     print(f"Sample {i+1}: {sample}")
    # # 2. 전처리 함수 적용: 불필요한 컬럼 제거 후 전처리 수행
    # processed_dataset = dataset.map(
    #     lambda batch: preprocess_function(batch, processor),
    #     remove_columns=dataset.column_names,
    #     batched=False
    # )
    # print("전처리 완료")
    # # 3. 학습 관련 설정
    # training_args = TrainingArguments(
    #     output_dir="./whisper_model",
    #     per_device_train_batch_size=4,
    #     eval_strategy="no",
    #     num_train_epochs=3,
    #     logging_steps=10,
    #     save_steps=10,
    #     fp16=True,
    # )
    # print("학습 설정 완료")
    # # 4. Trainer 생성 및 학습 시작
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=processed_dataset,
    #     tokenizer=processor.feature_extractor,  # 학습 로깅에 사용
    #     data_collator=get_data_collator(processor)
    # )
    # print("학습 시작")
    # trainer.train()
    # print("학습 완료")
    # 5. Inference 수행 및 결과 저장