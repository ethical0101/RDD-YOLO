"""Build the browser demo and publish it to GitHub Pages (branch gh-pages).

Requires: git push access to the repository (e.g. `gh auth login` done once by you).
Usage:   .venv\\Scripts\\python scripts\\deploy_github_pages.py [--repo ethical0101/RDD-YOLO]
Live at: https://<owner>.github.io/<repo>/
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "deploy" / "static_site"


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ethical0101/RDD-YOLO")
    args = ap.parse_args()
    name = args.repo.split("/")[1]
    run([str(ROOT / ".venv" / "Scripts" / "python.exe"), "scripts/export_static_site.py", "--base", f"/{name}/"], ROOT)
    if (SITE / ".git").exists():
        shutil.rmtree(SITE / ".git")
    run(["git", "init", "-q", "-b", "gh-pages"], SITE)
    run(["git", "add", "-A"], SITE)
    run(["git", "commit", "-q", "-m", "Deploy RDD-YOLO browser demo"], SITE)
    run(["git", "push", "-q", "--force", f"https://github.com/{args.repo}.git", "gh-pages:gh-pages"], SITE)
    owner = args.repo.split("/")[0]
    print(f"Published. Live in ~1 minute: https://{owner}.github.io/{name}/")


if __name__ == "__main__":
    main()
