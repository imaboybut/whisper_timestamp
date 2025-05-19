'''
load_model_and_processor 함수 : PEFT(Parameterized Efficient Fine-Tuning) 모델과 프로세서를 로드
'''

# model.py
import torch
from peft import PeftModel, PeftConfig
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_model_and_processor(
    peft_model_id: str = "handsomemin/openai-whisper-large-v2-k12-0416-LORA-LORA"
):
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    base_model_name = peft_config.base_model_name_or_path

    processor = WhisperProcessor.from_pretrained(base_model_name)

    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(model, peft_model_id)
    model.eval()
    return model, processor
