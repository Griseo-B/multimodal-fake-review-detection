from pathlib import Path
import subprocess
import os
import sys

p = Path(r"C:\Users\BronyaGriseo\Desktop") / "\u5927\u521b" / "multimodal-fake-review-detection"
os.chdir(p)


def run(cmd, check=True):
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.stdout:
        print(r.stdout)
    if r.stderr:
        print(r.stderr)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r


# commit if needed
st = run(["git", "status", "--porcelain"], check=False)
if any(line.startswith("A ") or line.startswith("M ") or line.startswith("??") for line in (st.stdout or "").splitlines()):
    # already staged from before; if unstaged, add
    run(["git", "add", "-A"])
log = run(["git", "log", "-1", "--oneline"], check=False)
if log.returncode != 0:
    run(["git", "commit", "-m", "Initial commit: project scaffold and data templates"])
else:
    print("commit exists:", log.stdout.strip())

# remote?
rem = run(["git", "remote"], check=False)
if "origin" not in (rem.stdout or ""):
    r = run(
        [
            "gh",
            "repo",
            "create",
            "multimodal-fake-review-detection",
            "--private",
            "--source=.",
            "--remote=origin",
            "--push",
            "--description",
            "Qiushi Academic: multimodal fake review detection",
        ],
        check=False,
    )
    if r.returncode != 0:
        run(
            [
                "gh",
                "repo",
                "create",
                "ruc-qiushi-fake-review-2026",
                "--private",
                "--source=.",
                "--remote=origin",
                "--push",
                "--description",
                "Qiushi Academic: multimodal fake review detection",
            ]
        )
else:
    print("origin already set")
    run(["git", "push", "-u", "origin", "main"], check=False)

run(["git", "remote", "-v"])
run(["gh", "repo", "view", "--json", "url,name,visibility"])
run(["git", "status", "-sb"])
