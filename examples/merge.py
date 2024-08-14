from fire import Fire
import peft
from peft import PeftModel
from utils import initialize_text_to_text_model
import os



def merge(checkpoint: str, dtype: str, merge_suffix="merged_checkpoint"):
    model_name = "meta-llama/Llama-2-7b-hf"
    model_type = "CausalLM"
    model, tokenizer = initialize_text_to_text_model(
        model_name, model_type, dtype="bf16"
    )

    if dtype in ["nf4", "int8"]:
        float_attr_list = list(model.__dict__.keys())
        float_llama_config_attr_list = list(model.config.__dict__.keys())
        del model
        model, tokenizer = initialize_text_to_text_model(
            model_name, model_type, dtype=dtype
        )
        print(f"dtype of model is {dtype}, so dequantize model")
        print("before dequantize======================================")
        print(model)
        model = model.dequantize()
        quant_attr_list = list(model.__dict__.keys())
        quant_llama_config_attr_list = list(model.config.__dict__.keys())
        for attr in quant_attr_list:
            if attr not in float_attr_list:
                delattr(model, attr)
        for attr in quant_llama_config_attr_list:
            if attr not in float_llama_config_attr_list:
                delattr(model.config, attr)
        model = model.bfloat16()
        model = model.to("cpu")
        print("finish dequnatize=======================================")
        print(model)
    model = PeftModel.from_pretrained(model, checkpoint)
    model = model.merge_and_unload()
    model.save_pretrained(os.path.join(checkpoint, merge_suffix))
    tokenizer.save_pretrained(os.path.join(checkpoint, merge_suffix))


if __name__ == "__main__":
    Fire(merge)
