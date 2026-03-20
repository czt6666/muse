import json

from datasets import load_dataset
from huggingface_hub import hf_hub_download

data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Clothing_Shoes_and_Jewelry", split="full", trust_remote_code=True)
print(len(data))
print(data[0])

