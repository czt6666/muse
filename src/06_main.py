# This is a sample Python script.
import os

from dotenv import load_dotenv

from src.chat.conv_manager import Cmanager
from src.chat.system_chat import Recsys
from src.chat.user_chat import User
import json
from tqdm import tqdm
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

load_dotenv()
api_base = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
db_path = os.getenv("DB_PATH")
data_path = os.getenv("DATA_PATH")
model_name = os.getenv("MODEL_NAME")

with open('user_profiles.json', 'r', encoding='utf-8') as file:
    users = json.load(file)

user = User(base_url=api_base, api_key=api_key)
recsys = Recsys(db_path=db_path, data_path=data_path, model_path=model_name, base_url=api_base, api_key=api_key)
cmanager = Cmanager(user=user, recsys=recsys, base_url=api_base, api_key=api_key)

for u in tqdm(users):
    try:
        cmanager.conv_process(u)
    except Exception as e:
        print("Error", e)
        continue
