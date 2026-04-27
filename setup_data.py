import argparse
import json
import os
import subprocess
from huggingface_hub import snapshot_download


def download_civil_comments():
    from datasets import load_dataset

    os.makedirs("data/civil_comments", exist_ok=True)

    print("Downloading google/civil_comments train split...")
    train = load_dataset("google/civil_comments", split="train")
    print("Downloading google/civil_comments test split...")
    test = load_dataset("google/civil_comments", split="test")

    splits = {
        "toxic": train.filter(lambda x: x["toxicity"] >= 0.8),
        "non_toxic": train.filter(lambda x: x["toxicity"] < 0.2),
        "full": train,
        "toxic_test": test.filter(lambda x: x["toxicity"] >= 0.8),
    }

    for name, subset in splits.items():
        path = f"data/civil_comments/{name}.jsonl"
        print(f"Saving {len(subset)} records → {path}")
        with open(path, "w", encoding="utf-8") as f:
            for row in subset:
                f.write(json.dumps({"text": row["text"]}) + "\n")

    print("Civil Comments data saved to data/civil_comments/")
    print(f"  toxic.jsonl       — {len(splits['toxic'])} toxic comments (toxicity >= 0.8)")
    print(f"  non_toxic.jsonl   — {len(splits['non_toxic'])} non-toxic comments (toxicity < 0.2)")
    print(f"  full.jsonl        — {len(splits['full'])} all train comments")
    print(f"  toxic_test.jsonl  — {len(splits['toxic_test'])} toxic test comments (for eval prompts)")


def download_eval_data():
    snapshot_download(
        repo_id="open-unlearning/eval",
        allow_patterns="*.json",
        repo_type="dataset",
        local_dir="saves/eval",
    )


def download_idk_data():
    snapshot_download(
        repo_id="open-unlearning/idk",
        allow_patterns="*.jsonl",
        repo_type="dataset",
        local_dir="data",
    )


def download_wmdp():
    url = "https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-corpora.zip"
    dest_dir = "data/wmdp"
    zip_path = os.path.join(dest_dir, "wmdp-corpora.zip")

    os.makedirs(dest_dir, exist_ok=True)
    subprocess.run(["wget", url, "-O", zip_path], check=True)
    subprocess.run(["unzip", "-P", "wmdpcorpora", zip_path, "-d", dest_dir], check=True)


def main():
    parser = argparse.ArgumentParser(description="Download and setup evaluation data.")
    parser.add_argument(
        "--eval_logs",
        action="store_true",
        help="Downloads TOFU, MUSE  - retain and finetuned models eval logs and saves them in saves/eval",
    )
    parser.add_argument(
        "--idk",
        action="store_true",
        help="Download idk dataset from HF hub and stores it data/idk.jsonl",
    )
    parser.add_argument(
        "--wmdp",
        action="store_true",
        help="Download and unzip WMDP dataset into data/wmdp",
    )
    parser.add_argument(
        "--civil_comments",
        action="store_true",
        help="Download and filter Civil Comments dataset into data/civil_comments",
    )

    args = parser.parse_args()

    if args.eval_logs:
        download_eval_data()
    if args.idk:
        download_idk_data()
    if args.wmdp:
        download_wmdp()
    if args.civil_comments:
        download_civil_comments()


if __name__ == "__main__":
    main()
