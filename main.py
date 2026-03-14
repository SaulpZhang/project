import yaml
import data_processing.match_pairs as match_pairs
import os
import argparse

if __name__ == "__main__":
    argsParser = argparse.ArgumentParser(description="Natural language to SMT-LIB V2 code")
    argsParser.add_argument("--account_dir", type=str, default='./valid_permission/accounts')
    argsParser.add_argument("--instruct_dir", type=str, default='./valid_permission/instructs')
    argsParser.add_argument("--label_file", type=str, default='./valid_permission/answer_valid_permission.json')
    args = argsParser.parse_args()

    with open("cfg/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_process_config = config.get("data_process", {})


    data_base_dir = data_process_config.get("data_base_dir")

    accounts_dir = args.account_dir if args.account_dir else os.path.join(
        data_base_dir, data_process_config.get("accounts_dir", "accounts")
    )
    instructs_dir = args.instruct_dir if args.instruct_dir else os.path.join(
        data_base_dir, data_process_config.get("instructs_dir", "instructs")
    )
    label_file = args.label_file if args.label_file else os.path.join(
        data_base_dir,
        data_process_config.get("label_file", "answer_valid_permission.json"),
    )

    data_pairs = match_pairs.match_account_instructs(
        accounts_dir, instructs_dir, label_file
    )
    print(f"Matched {len(data_pairs)} pairs of accounts and instructs.")
    for account_path, instruct_path, label in data_pairs:
        print(account_path, "<->", instruct_path, "label=", label)

    
