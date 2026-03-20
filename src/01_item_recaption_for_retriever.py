import base64
import json
import os

import requests
from datasets import load_dataset
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from pydantic import BaseModel
from tqdm import tqdm

from utils.invoke import invoke


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_product_description(text, image_path):
    content_system = "You are a item feature extractor. " \
                     "Given the information: \n" \
                     "1. The Image of the item. " \
                     "2. The Text information of the item" \
                     "You need to follow the steps below: \n" \
                     "1. Identify the part of the product in the image. " \
                     "2. Focus on the product itself and ignore other objects in the image. " \
                     "3. Generate a detail description of the product including all basic features and visual features. \n" \
                     "Output the only the description of the item."
    content_user = []
    text_pre = f"Text information:{text},"
    content_user.append({"type": "text", "text": f"{text_pre}"}, )
    base64_image = encode_image(image_path)
    content_user.append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}})
    return invoke(content_system, content_user)

def _get_first_image_url(item):
    """从 item['images'] 中取第一张图 URL，优先级 hi_res > large > thumb"""
    image_dict = item.get("images", {})
    for key in ["hi_res", "large", "thumb"]:
        urls = image_dict.get(key, [])
        if urls and urls[0]:
            return urls[0]
    return None


def download_image(url, save_path):
    if os.path.exists(save_path):
        return save_path
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
    return save_path

data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Clothing_Shoes_and_Jewelry", split="full[:20]", trust_remote_code=True)

os.makedirs("images_main", exist_ok=True)
new_descriptions = {}
for id, item in tqdm(enumerate(data)):
    image_url = _get_first_image_url(item)
    print(f"Downloading image {id}: {image_url}")
    if not image_url:
        print(f"Item {id}: No image URL found, skipping.")
        continue

    image_path = f"images_main/{id}.jpg"
    download_image(image_url, image_path)
    text_information = f"Title:{item['title']}; Description: {item['description']}; " \
                       f"Feature: {item['features']}; Categories: {item['categories']}"

    new_description = generate_product_description(text_information, image_url)
    item_profile = dict(item)
    item_profile["new_description"] = new_description
    new_descriptions[id] = item_profile

# 写入 updated_item_profile.json
output_path = os.path.join(os.path.dirname(__file__), "updated_item_profile.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(list(new_descriptions.values()), f, ensure_ascii=False, indent=2)

