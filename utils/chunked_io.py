"""
大文件分段读写方案

适用场景：item 数量很大（如 10 万+），一次性 load 全量 JSON 会 OOM 或过慢。
思路：按 chunk 分批处理，每批写一个分片文件，最后合并；或流式读大 JSON。
"""

import json
import os
from pathlib import Path


def write_chunked(data_iter, out_dir: str, prefix: str = "chunk", chunk_size: int = 500):
    """
    分批写入：每 chunk_size 条写一个 {prefix}_{i}.json，最后写 {prefix}_final.json。

    data_iter: 迭代 (id, value) 或 (id, dict)
    out_dir: 输出目录
    prefix: 分片文件名前缀
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    chunk = {}
    t = 1
    for id_, val in data_iter:
        chunk[id_] = val
        if len(chunk) >= chunk_size:
            path = os.path.join(out_dir, f"{prefix}_{t}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
            t += 1
            chunk = {}
    if chunk:
        path = os.path.join(out_dir, f"{prefix}_final.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)


def merge_chunked(chunk_dir: str, prefix: str = "chunk", out_path: str = "merged.json"):
    """
    合并分片：读取 {prefix}_1.json .. {prefix}_N.json 和 {prefix}_final.json，合并为一个大 dict。
    """
    merged = {}
    i = 1
    while True:
        path = os.path.join(chunk_dir, f"{prefix}_{i}.json")
        if not os.path.exists(path):
            break
        with open(path, "r", encoding="utf-8") as f:
            merged.update(json.load(f))
        i += 1
    final_path = os.path.join(chunk_dir, f"{prefix}_final.json")
    if os.path.exists(final_path):
        with open(final_path, "r", encoding="utf-8") as f:
            merged.update(json.load(f))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


def iter_jsonl_chunked(filepath: str, chunk_size: int = 500):
    """
    流式读 JSONL 大文件，按 chunk 迭代，避免一次性 load 全量。

    Yields: list[dict]，每批最多 chunk_size 条
    """
    chunk = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk
