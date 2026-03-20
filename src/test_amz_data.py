from datasets import load_dataset

# 用户评论
dataset1 = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
print(dataset1["full"][0]) 

# 商品数据
dataset2 = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True)
print(dataset2[0])

# 5core_rating_only_All_Beauty
# 0core → 不做k-core过滤
# rating_only → 只保留评分交互
# All_Beauty → 商品类别
dataset3 = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "0core_rating_only_All_Beauty", trust_remote_code=True)
print(dataset3['full'][0:5])

# 按时间节点切分的训练集和测试集
dataset4 = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "0core_timestamp_All_Beauty", trust_remote_code=True)
print(dataset4['train'][:5])
print(dataset4['test'][:5])

# 用户之前买/看/评过什么
# {
#  user_id: xxx
#  parent_asin: item5
#  rating: 5
#  history: item1 item2 item3 item4
# }
dataset5 = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "0core_timestamp_w_his_All_Beauty", trust_remote_code=True)
print(dataset5['train'][:5])
print(dataset5['valid'][:5])
print(dataset5['test'][:5])

# sequential recommendation 序列推荐 序列推荐

# C4: Complex Contexts Created by ChatGPT.
# Queries
dataset6 = load_dataset('McAuley-Lab/Amazon-C4')['test']
print(dataset6)
print(dataset6[288])

import json
from huggingface_hub import hf_hub_download

# Item Pool
filepath = hf_hub_download(
    repo_id='McAuley-Lab/Amazon-C4',
    filename='sampled_item_metadata_1M.jsonl',
    repo_type='dataset'
)

item_pool = []
with open(filepath, 'r') as file:
    for line in file:
        item_pool.append(json.loads(line.strip()))


print(len(item_pool))
print(item_pool[0])

# 有进行k次交互筛选
# 有用户购买/评价历史
# 有商品信息
# 有评价信息
# 有 benchmark
