#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT-Train full demo — runs all three sub-demos in sequence.

    python demo.py              # run all three
    python demo.py job          # just the job demo
    python demo.py session      # just the session demo
    python demo.py estimate     # just the explore/estimate demo

Server must be running:
    uvicorn server.main:app --port 8000
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = {
    "job":      HERE / "demo_job.py",
    "session":  HERE / "demo_session.py",
    "estimate": HERE / "demo_estimate.py",
}



def run(name: str) -> None:
    print(f"\n{'#'*60}")
    print(f"#  {name.upper()} DEMO")
    print(f"{'#'*60}\n")
    result = subprocess.run([sys.executable, str(SCRIPTS[name])])
    if result.returncode != 0:
        print(f"\n[demo_{name}.py exited with code {result.returncode}]")


def main() -> None:
    targets = sys.argv[1:] or list(SCRIPTS)
    for name in targets:
        if name not in SCRIPTS:
            print(f"Unknown demo: {name!r}. Choose from: {', '.join(SCRIPTS)}")
            sys.exit(1)
        run(name)


if __name__ == "__main__":
    main()
