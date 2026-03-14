from pathlib import Path
from typing import List, Tuple
import os
import json

def get_label(label_path):
    with open(label_path, 'r') as f:
        data = json.load(f)
    label_map = {}
    for i, item in enumerate(list(data)):
        label_map[i+1] = item
    return label_map

def match_account_instructs(accounts_path: str, instructs_path: str, label_path: str) -> List[Tuple[str, str, int]]:
    """将 `accounts_path` 和 `instructs_path` 下的 JSON 文件按文件名（不含扩展名）一一配对。

    Args:
        accounts_path: 指向 accounts 目录或单个 account json 文件的路径。
        instructs_path: 指向 instructs 目录或单个 instruct json 文件的路径。
        label_path: 指向标签文件的路径。

    Returns:
        列表，元素为三元组 `(account_path, instruct_path, label)`。
        当文件名（不含扩展名）相同时，返回 `label` 为 1 的配对。
    """
    accounts = Path(accounts_path)
    instructs = Path(instructs_path)

    if accounts.is_file():
        account_files = [accounts]
    else:
        account_files = sorted([p for p in accounts.rglob("*.json") if p.is_file()], key=get_key)

    if instructs.is_file():
        instruct_files = [instructs]
    else:
        instruct_files = sorted([p for p in instructs.rglob("*.json") if p.is_file()], key=get_key)

    instruct_map = {get_key(p): p for p in instruct_files}

    label_map = get_label(label_path)

    pairs: List[Tuple[str, str, int]] = []
    for a in account_files:
        key = get_key(a)
        inst = instruct_map.get(key)
        label = label_map.get(key)  # 默认标签为 0，如果没有找到对应的标签
        if inst is not None and label is not None:
            pairs.append((str(a), str(inst), label))

    return pairs

def get_key(filePath: Path) -> int:
    return int(filePath.stem.split("_")[-1])

if __name__ == "__main__":
    # 简单示例：在仓库根目录运行时会尝试匹配 valid_permission 下的目录
    base = Path(os.getcwd()) / "valid_permission"
    accounts_dir = base / "accounts"
    instructs_dir = base / "instructs"
    label_path = base / "answer_valid_permission.json"

    data_pairs = match_account_instructs(str(accounts_dir), str(instructs_dir), str(label_path))
    for account_path, instruct_path, label in data_pairs[:20]:
        print(account_path, "<->", instruct_path, "label=", label)
