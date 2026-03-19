import json

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from utils.invoke import invoke

data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Clothing_Shoes_and_Jewelry", split="full[:20]", trust_remote_code=True)
print(type(data)) # <class 'datasets.arrow_dataset.Dataset'>
print(len(data))
# 取前2000
# data_2000 = data[:2000]
print(data[0])


cate = {}
for metadata in data:
    categories = metadata['categories']
    for i in categories:
        if i in cate:
            cate[i] += 1
        else:
            cate[i] = 1

print(cate)

cate_li = []
for cate, num in tqdm(cate.items()):
    # if num < 5:
    #     continue
    content_system = (
        "You are an AI assistant specialized in identifying categories of clothing, footwear, and accessories. "
        "I will give you a label, and you need to determine if this label belongs to one of the following three categories: \n"
        "1. Types of clothing (e.g., jacket, down coat, T-shirt, etc. "
        "2. Types of footwear (e.g., high heels, sneakers, sandals, etc. "
        "3. Types of accessories/decorative items (e.g., necklace, handbag, hat, etc.) \n"
        "If the given label belongs to any of the above three categories, please answer 'Yes'. "
        "If it does not belong, please answer 'No'. "
        "Please only answer 'Yes' or 'No' without any additional explanation. "
        "For example: \n"
        "Input: jacket "
        "Output: Yes "
        "Input: high heels "
        "Output: Yes "
        "Input: necklace "
        "Output: Yes "
        "Input: clothing "
        "Output: No \n "
        "Output only 'Yes' or 'No'. "
    )

    response = invoke(content_system, f"Tag: {cate}")
    print(cate, response)

    if "Yes" in response:
        cate_li.append(cate)


with open("categories.json", "w", encoding="utf-8") as file:
    json.dump(cate_li, file, ensure_ascii=False, indent=2)
