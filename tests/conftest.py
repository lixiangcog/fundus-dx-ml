import os
import sys

# Run tests from the repo root: api/main.py loads best_model.pth via a
# root-relative path at import time.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
