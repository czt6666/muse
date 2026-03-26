# Muse 项目运行说明

## 一、项目是什么

这是论文 **《Muse: A Multimodal Conversational Recommendation Dataset with Scenario-Grounded User Profiles》** 的代码，用于**合成多模态对话推荐数据集**：

- **用户侧**：按“人设 + 购买场景 + 目标商品”生成虚拟用户。
- **系统侧**：模拟推荐系统，做需求理解、澄清、检索、推荐、寒暄。
- **对话**：用户与系统多轮对话（寒暄 / 拒绝推荐 / 接受），支持**图文多模态**（商品图 + 文本）。
- **输出**：每条数据包含 Persona、Scenario、Target_item、Mentioned_items、完整 Conversations 等，可用来训/评对话推荐模型。

---

## 二、从哪个文件开始跑（推荐顺序）

整体是**先准备数据 → 再生成用户画像 → 最后跑对话**。

| 步骤 | 运行文件 | 作用 | 前置条件 |
|------|----------|------|----------|
| 0 | 准备原始商品数据 | 需要一份商品 JSON（见下「数据准备」） | 无 |
| 1 | `extract_categories.py` | 从商品里抽品类 → 得到 `categories.json` | 已有 `updated_item_profile.json` |
| 2 | `get_categories2item.py` | 建品类→商品映射 → 得到 `category2items.json` | 已有 `categories.json` + `updated_item_profile.json` |
| 3 | （可选）`item_recaption_for_retriever.py` | 用 VLM 为商品生成 `new_description`，写入商品画像 | 有商品图 + 原始商品信息 |
| 4 | `generate_user_profiles.py` | 生成虚拟用户（人设+场景+目标商品）→ 得到 `user_profiles.json` | 见下「第一步前需要什么」 |
| 5 | `main.py` 或 `generate_convs.py` | **主入口**：按用户列表跑对话，写出 `convs/`、`detail_convs/` | 已有 `user_profiles.json`（或 `user_scenarios_7005.json`） |

**“想先跑起来”时**：若你**没有**现成的 `user_profiles.json` / `user_scenarios_7005.json`，就**从第 4 步 `generate_user_profiles.py` 开始**（前提是第 0～2 步的数据都准备好）。若**已有**用户列表，则直接运行 **`main.py`** 或 **`generate_convs.py`**。

---

## 三、db_path、data_path、model_name 怎么填

这三个参数在 **`main.py`**、**`generate_convs.py`**、**`generate_user_profiles.py`**、**`create_item_db.py`** 里都会用到，含义一致：

| 参数 | 含义 | 怎么填 | 示例 |
|------|------|--------|------|
| **db_path** | FAISS 向量库的**目录**（存商品向量索引，用于检索） | 填一个**本地文件夹路径**。若目录不存在或要重建，会用 `data_path` 建库并保存到这里；若已存在且不强制重建，会直接加载。 | `"./data/faiss_db"` 或 `"D:/Muse/faiss_index"` |
| **data_path** | **商品画像 JSON** 的路径（每条商品含 title、description、**new_description**、categories、features 等） | 填 `updated_item_profile.json` 的**完整路径**（相对或绝对均可）。 | `"./data/updated_item_profile.json"` 或 `"D:/Muse/updated_item_profile.json"` |
| **model_name** | 用于把商品文本打成向量的 **Embedding 模型**（如 BGE、sentence-transformers） | 填 HuggingFace 模型名或**本地模型目录路径**。需与 `sentence_transformers` / `langchain_community` 能加载的格式一致。 | `"BAAI/bge-m3"` 或 `"D:/models/bge-m3"` |

- **main.py / generate_convs.py** 里：把上面的 `db_path`、`data_path`、`model_name` 填到对应变量；同时填好 **api_base**、**api_key**（OpenAI 兼容接口），以及用户列表文件（见下）。
- **generate_user_profiles.py** 里：脚本中部有 `db_path`、`data_path`、`model_name` 和 `client = OpenAI(...)`，改成你的路径和 API；它还会读 **`category2items.json`**、**`updated_item_profile.json`**、**`reasons.json`**，这些文件要事先准备好。

---

## 四、第一步前需要准备的数据

要跑 **generate_user_profiles.py**，项目根目录（或脚本里写的路径）下需要有：

- **updated_item_profile.json**  
  格式大致为：`{ "item_id_1": { "title", "description", "new_description", "categories", "features", ... }, ... }`  
  其中 **new_description** 可由 `item_recaption_for_retriever.py` 生成，或自己写。
- **category2items.json**  
  由 `get_categories2item.py` 生成，格式：`{ "品类名": [ "item_id1", "item_id2", ... ], ... }`。
- **reasons.json**  
  购买动机，格式：`{ "Work": ["原因1", "原因2"], "Gifts": [...], ... }`，需自己准备或从论文/仓库示例里拿。
- **images_main/**（若做多模态）  
  商品图片，命名为 `{item_id}.jpg`，与 `conv_manager` / `system_chat` 里写的路径一致。

要跑 **main.py** 或 **generate_convs.py**，需要：

- 用户列表：**user_profiles.json** 或 **user_scenarios_7005.json**（由 `generate_user_profiles.py` 生成，或自己按同格式构造）。
- 同样需要 **db_path**、**data_path**、**model_name** 以及 **api_base**、**api_key**。

---

## 五、最后会得到什么效果

- **generate_user_profiles.py** 跑完后：得到 **user_profiles.json**（或你重命名/拷贝为 **user_scenarios_7005.json**），每行一个虚拟用户（profile + scenario + requirements + target_item）。
- **main.py / generate_convs.py** 跑完后：
  - **convs/conv_{i}.json**：第 i 条对话的纯消息列表（Assistant/User 轮次）。
  - **detail_convs/conv_{i}.json**：同一条对话的**完整标注**，包含 Persona、Scenario、Target_item、Mentioned_items、Conversations（带 Action、Mentioned_item 等）。

这些输出即论文中的 **Muse 多模态对话推荐数据集**，可用于训练或评估对话推荐、多轮澄清、多模态检索等模型。

---

## 六、配置示例（main.py）

```python
# 示例：按你本机路径和 API 修改
api_base = 'https://your-openai-compatible-api.com/v1'
api_key = 'your-api-key'

db_path = "./data/faiss_db"                    # FAISS 索引目录
data_path = "./data/updated_item_profile.json" # 商品画像
model_name = "BAAI/bge-m3"                     # 向量模型名或本地路径

# 用户列表：二选一
# with open('user_profiles.json', 'r') as f:
with open('user_scenarios_7005.json', 'r') as f:
    users = json.load(f)
```

把上述路径和 API 换成你自己的后，从 **generate_user_profiles.py**（没有用户列表时）或 **main.py**（已有用户列表时）开始跑即可。
