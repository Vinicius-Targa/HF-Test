from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from API_Key import HUGGING_FACE_API_KEY

hf_model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

hf_req_files = [
    'config.json',
    'generation_config.json',
    'model.safetensors',
    'special_tokens_map.json',
    'tokenizer.json',
    'tokenizer.model',
    'tokenizer_config.json',
    'eval_results.json'
]

for filename in hf_req_files:
    dl_location = hf_hub_download(
        repo_id=hf_model,
        filename=filename,
        token=HUGGING_FACE_API_KEY
    )
    print(f'File downloaded to: {dl_location}')

model = AutoModelForCausalLM.from_pretrained(hf_model)
tokenizer = AutoTokenizer.from_pretrained(hf_model, legacy=False)

text_gen_pipeline = pipeline(
    'text-generation',
    model=model,
    device=-1,
    tokenizer=tokenizer,
    max_length=100
)

res = text_gen_pipeline("Why is the sky blue?")
print(res)