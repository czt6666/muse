from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from collections import defaultdict
import math


class AmazonBeautyMetaSearcher:
    def __init__(self):
        """
        默认加载固定数据集，并在初始化时：
        1. 构建 asin -> index 的高效索引
        2. 检查 asin 是否重复
        """
        self.dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_meta_All_Beauty",
            split="full",
            trust_remote_code=True
        )

        self.asin_to_index = {}
        self.duplicate_asins = defaultdict(list)

        self._build_index()

    def _build_index(self):
        """
        单次扫描构建索引。
        - 如果 asin 唯一：asin_to_index[asin] = row_idx
        - 如果 asin 重复：记录到 duplicate_asins 中
        """
        temp_positions = defaultdict(list)

        for idx, item in enumerate(self.dataset):
            asin = item.get("parent_asin")
            if asin is not None:
                temp_positions[asin].append(idx)

        for asin, indices in temp_positions.items():
            if len(indices) == 1:
                self.asin_to_index[asin] = indices[0]
            else:
                self.duplicate_asins[asin] = indices
                # 默认保留第一个，方便主查询函数仍可返回结果
                self.asin_to_index[asin] = indices[0]

        total_asins = len(temp_positions)
        duplicate_count = len(self.duplicate_asins)

        print(f"数据集加载完成，总样本数: {len(self.dataset)}")
        print(f"唯一 parent_asin 数量: {total_asins}")
        print(f"重复 parent_asin 数量: {duplicate_count}")

        if duplicate_count > 0:
            print("发现重复 ASIN，查询时默认返回该 ASIN 的第一条记录。")
        else:
            print("未发现重复 ASIN。")

    def has_duplicate_asin(self):
        """
        返回是否存在重复 asin
        """
        return len(self.duplicate_asins) > 0

    def get_duplicate_asins(self):
        """
        返回所有重复 asin 及其对应下标列表
        例如:
        {
            "B001XXX": [10, 25],
            "B002YYY": [33, 48, 100]
        }
        """
        return dict(self.duplicate_asins)

    def print_duplicate_summary(self, max_show=10):
        """
        打印重复 asin 摘要
        """
        if not self.duplicate_asins:
            print("没有重复的 parent_asin。")
            return

        print(f"共有 {len(self.duplicate_asins)} 个重复的 parent_asin，最多展示前 {max_show} 个：")
        for i, (asin, indices) in enumerate(self.duplicate_asins.items()):
            if i >= max_show:
                break
            print(f"ASIN: {asin}, 出现次数: {len(indices)}, 下标: {indices}")

    def query_by_asin(self, asin, show_images=True, max_images=5):
        """
        根据 parent_asin 查询商品详情
        - asin: 商品 parent_asin
        - show_images: 是否展示图片
        - max_images: 最多展示几张图片
        """
        if asin not in self.asin_to_index:
            print(f"未找到 ASIN = {asin} 的商品。")
            return None

        idx = self.asin_to_index[asin]
        item = self.dataset[idx]

        print("=" * 80)
        print(f"row index       : {idx}")
        print(f"parent_asin     : {item.get('parent_asin')}")
        print(f"title           : {item.get('title')}")
        print(f"main_category   : {item.get('main_category')}")
        print(f"average_rating  : {item.get('average_rating')}")
        print(f"rating_number   : {item.get('rating_number')}")
        print(f"price           : {item.get('price')}")
        print(f"store           : {item.get('store')}")
        print(f"subtitle        : {item.get('subtitle')}")
        print(f"author          : {item.get('author')}")
        print(f"categories      : {item.get('categories')}")
        print(f"features        : {item.get('features')}")
        print(f"description     : {item.get('description')}")
        print(f"details         : {item.get('details')}")
        print(f"bought_together : {item.get('bought_together')}")
        print("=" * 80)

        if asin in self.duplicate_asins:
            print(f"注意：该 ASIN 在数据集中重复出现，共 {len(self.duplicate_asins[asin])} 条，当前返回第一条。")
            print(f"所有对应下标: {self.duplicate_asins[asin]}")

        if show_images:
            self._show_item_images(item, max_images=max_images)

        return item

    def _collect_image_urls(self, item):
        """
        按优先级收集图片链接：
        hi_res > large > thumb
        并去重、过滤 None
        """
        image_dict = item.get("images", {})
        candidates = []

        for key in ["hi_res", "large", "thumb"]:
            urls = image_dict.get(key, [])
            if urls:
                candidates.extend(urls)

        # 去重并过滤空值
        seen = set()
        valid_urls = []
        for url in candidates:
            if url and url not in seen:
                seen.add(url)
                valid_urls.append(url)

        return valid_urls

    def _show_item_images(self, item, max_images=5):
        """
        展示商品图片
        """
        urls = self._collect_image_urls(item)

        if not urls:
            print("该商品没有可展示的图片。")
            return

        urls = urls[:max_images]
        n = len(urls)
        cols = min(3, n)
        rows = math.ceil(n / cols)

        plt.figure(figsize=(5 * cols, 5 * rows))

        shown = 0
        for i, url in enumerate(urls, start=1):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")

                plt.subplot(rows, cols, i)
                plt.imshow(img)
                plt.title(f"Image {i}")
                plt.axis("off")
                shown += 1
            except Exception as e:
                print(f"图片加载失败: {url}\n错误信息: {e}")

        if shown > 0:
            plt.tight_layout()
            plt.show()
        else:
            print("图片都加载失败了。")

if __name__ == '__main__':
    searcher = AmazonBeautyMetaSearcher()
    res = searcher.query_by_asin("B01CUPMQZE")
    print(res)