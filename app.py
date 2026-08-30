"""
app.py
======
Root redirection launcher for Streamlit frontend.
Delegates to frontend/app.py
"""

import sys
from pathlib import Path

# Set up paths
_repo_root = Path(__file__).resolve().parent
_frontend_dir = _repo_root / "frontend"
for _p in [str(_repo_root), str(_frontend_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Execute frontend app
exec(open(_frontend_dir / "app.py", encoding="utf-8").read())
