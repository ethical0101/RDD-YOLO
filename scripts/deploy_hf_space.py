"""Upload deploy/hf_space/ to a Hugging Face Space (Docker SDK, free CPU hardware).

You need your own (free) Hugging Face account. Log in once yourself - the token is entered by you and
stored by huggingface_hub, this script never sees it:
    .venv\\Scripts\\hf auth login

Then:
    .venv\\Scripts\\python scripts\\deploy_hf_space.py --space <your-hf-username>/rdd-yolo
The Space builds the Docker image automatically (first build ~10 minutes) and is served at
    https://<your-hf-username>-rdd-yolo.hf.space
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "deploy" / "hf_space"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<username>/<space-name>")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    if not (BUNDLE / "Dockerfile").exists():
        sys.exit("Build the bundle first: python scripts/build_hf_space.py")

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        sys.exit("Not logged in. Run:  .venv\\Scripts\\hf auth login   (paste your own token there)")
    print(f"Logged in as {user}; uploading to Space {args.space} ...")
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(BUNDLE), repo_id=args.space, repo_type="space",
                      commit_message="Deploy RDD-YOLO")
    owner, name = args.space.split("/")
    print(f"Uploaded. Build logs: https://huggingface.co/spaces/{args.space}")
    print(f"App URL (after the build finishes): https://{owner}-{name}.hf.space".lower())


if __name__ == "__main__":
    main()
