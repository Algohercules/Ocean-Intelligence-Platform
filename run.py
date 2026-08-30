"""
run.py
======
Unified Launcher for Ocean Intelligence Platform.
Supports launching Backend API, Streamlit Frontend, or both concurrently.

Usage:
  python run.py             # Launches both FastAPI Backend and Streamlit Frontend
  python run.py --backend   # Launches FastAPI REST Backend only
  python run.py --frontend  # Launches Streamlit UI only
  python run.py --eval      # Runs ConvLSTM evaluation benchmarks
  python run.py --train     # Runs ConvLSTM training loop
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import API_HOST, API_PORT


def parse_args():
    parser = argparse.ArgumentParser(description="Ocean Intelligence Platform Launcher")
    parser.add_argument("--backend", action="store_true", help="Run FastAPI backend server only")
    parser.add_argument("--frontend", action="store_true", help="Run Streamlit frontend only")
    parser.add_argument("--eval", action="store_true", help="Run model evaluation")
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--port", type=int, default=API_PORT, help="Backend API port")
    parser.add_argument("--ui-port", type=int, default=8501, help="Streamlit UI port")
    return parser.parse_args()


def run_backend(port: int = API_PORT):
    import uvicorn
    print(f"[Launcher] Starting FastAPI Backend on http://{API_HOST}:{port} ...")
    uvicorn.run("backend.api.main:app", host=API_HOST, port=port, reload=True)


def run_frontend(ui_port: int = 8501):
    print(f"[Launcher] Starting Streamlit Frontend on http://localhost:{ui_port} ...")
    frontend_entry = BASE_DIR / "frontend" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(frontend_entry),
        "--server.port",
        str(ui_port),
        "--browser.gatherUsageStats",
        "false"
    ]
    subprocess.run(cmd)


def run_all(api_port: int = API_PORT, ui_port: int = 8501):
    print("=" * 65)
    print("🌊  INDIAN OCEAN INTELLIGENCE PLATFORM - LAUNCHING FULL STACK  🌊")
    print("=" * 65)
    print(f"[*] API Backend Docs : http://localhost:{api_port}/docs")
    print(f"[*] Streamlit UI     : http://localhost:{ui_port}")
    print("=" * 65 + "\n")

    # Start backend process
    backend_proc = subprocess.Popen([
        sys.executable,
        "-m",
        "uvicorn",
        "backend.api.main:app",
        "--host",
        API_HOST,
        "--port",
        str(api_port)
    ])

    time.sleep(1.5)

    try:
        run_frontend(ui_port=ui_port)
    except KeyboardInterrupt:
        print("\n[Launcher] Shutting down services...")
    finally:
        backend_proc.terminate()
        backend_proc.wait()
        print("[Launcher] Platform stopped gracefully.")


def main():
    args = parse_args()

    if args.eval:
        from scripts.evaluate_model import evaluate
        evaluate()
    elif args.train:
        from scripts.train_model import train
        train()
    elif args.backend:
        run_backend(port=args.port)
    elif args.frontend:
        run_frontend(ui_port=args.ui_port)
    else:
        # Default: run full stack
        run_all(api_port=args.port, ui_port=args.ui_port)


if __name__ == "__main__":
    main()
