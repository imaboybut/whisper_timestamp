from transformers import WhisperForConditionalGeneration, WhisperProcessor

def load_model_and_processor(model_name="SungBeom/whisper-small-ko"):
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    # generation_config 업데이트: no_timestamps_token_id 설정
    if not hasattr(model.generation_config, "no_timestamps_token_id"):
        try:
            no_ts_token = processor.tokenizer.no_timestamps_token
        except AttributeError:
            no_ts_token = "<|notimestamps|>"
        model.generation_config.no_timestamps_token_id = processor.tokenizer.convert_tokens_to_ids(no_ts_token)
    return model, processor