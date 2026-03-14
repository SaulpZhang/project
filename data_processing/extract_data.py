import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, List
from pathlib import Path


def extract_json_data(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data


def _element_to_dict(elem: ET.Element) -> Any:
    d: Dict[str, Any] = {}
    if elem.attrib:
        for k, v in elem.attrib.items():
            d[f"@{k}"] = v
    children = list(elem)
    if children:
        for child in children:
            child_dict = _element_to_dict(child)
            tag = child.tag
            if "}" in tag:
                tag = tag.split("}", 1)[1]
            if tag in d:
                if not isinstance(d[tag], list):
                    d[tag] = [d[tag]]
                d[tag].append(child_dict)
            else:
                d[tag] = child_dict
    text = (elem.text or "").strip()
    if text:
        if children or elem.attrib:
            d["#text"] = text
        else:
            return text
    return d


def _safe_json_load(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return s


def extract_account_id(account: Dict) -> Any:
    return account.get("account_id")


def extract_buckets(account: Dict) -> List[Dict]:
    raw = account.get("buckets")
    # Normalize buckets to a list of dicts
    if raw is None:
        buckets_json_data: List = []
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = raw
        if isinstance(parsed, dict):
            buckets_json_data = [parsed]
        elif isinstance(parsed, list):
            buckets_json_data = parsed
        else:
            buckets_json_data = []
    elif isinstance(raw, dict):
        buckets_json_data = [raw]
    elif isinstance(raw, list):
        buckets_json_data = raw
    else:
        buckets_json_data = []

    processed: List[Dict] = []
    for b in buckets_json_data:
        if not isinstance(b, dict):
            processed.append(b)
            continue
        pb = b.copy()
        bp = b.get("bucketpolicy") or b.get("bucket_policy")
        pb["bucket_policy"] = _safe_json_load(bp)

        ba = b.get("bucketacl") or b.get("bucket_acl")
        if isinstance(ba, str) and ba.strip():
            try:
                        elem = ET.fromstring(ba)
                        root_tag = elem.tag
                        if "}" in root_tag:
                            root_tag = root_tag.split("}", 1)[1]
                        pb["bucket_acl"] = {root_tag: _element_to_dict(elem)}
            except Exception:
                pb["bucket_acl"] = {}
        else:
            pb["bucket_acl"] = ba or {}

        processed.append(pb)
    return processed


def extract_agencies(account: Dict) -> List[Dict]:
    agencies = account.get("agencies") or []
    out: List[Dict] = []
    for a in agencies:
        if not isinstance(a, dict):
            out.append(a)
            continue
        aa = a.copy()
        tp = a.get("trust_policy")
        aa["trust_policy"] = _safe_json_load(tp)
        out.append(aa)
    return out


def extract_users(account: Dict) -> List[Dict]:
    return account.get("users") or []


def extract_groups(account: Dict) -> List[Dict]:
    return account.get("groups") or []


def extract_admin_group_user_ids(account: Dict) -> List[str]:
    return account.get("admin_group_user_ids") or []


def extract_identity_policies(account: Dict) -> List[Dict]:
    policies = account.get("identity_policies") or []
    out: List[Dict] = []
    for p in policies:
        if not isinstance(p, dict):
            out.append(p)
            continue
        pp = p.copy()
        policy_str = p.get("policy")
        pp["policy"] = _safe_json_load(policy_str)
        out.append(pp)
    return out


def extract_account_data(path: str) -> Dict:
    account_json_data = extract_json_data(path)
    data: Dict[str, Any] = {}
    data["account_id"] = extract_account_id(account_json_data)
    data["buckets"] = extract_buckets(account_json_data)
    data["agencies"] = extract_agencies(account_json_data)
    data["users"] = extract_users(account_json_data)
    data["groups"] = extract_groups(account_json_data)
    data["admin_group_user_ids"] = extract_admin_group_user_ids(account_json_data)
    data["identity_policies"] = extract_identity_policies(account_json_data)
    return data


def extract_instruct_data(path: str) -> Dict:
    instruct_json_data = extract_json_data(path)
    data = {}
    data["instruct"] = instruct_json_data["instruct"]
    return data


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    account_path = base / "valid_permission" / "accounts" / "account_1_1.json"
    account_data = extract_account_data(str(account_path))
    print(json.dumps(account_data, indent=2))

    instruct_path = base / "valid_permission" / "instructs" / "instruct_1_1.json"
    instruct_data = extract_instruct_data(str(instruct_path))