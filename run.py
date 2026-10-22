"""
run.py
Unified application runner for AcoustiGuard AI.
Starts both the FastAPI backend and Vite frontend development server
with health polling, automatic browser launch, and graceful shutdown.

Usage:
    python run.py
    python run.py --no-browser
    python run.py --port 8000 --frontend-port 5173
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def check_prerequisites() -> bool:
    """Validate that Python packages and frontend node_modules are ready."""
    print("[*] Checking environment prerequisites...")

    # Check critical Python modules
    required_modules = ["fastapi", "uvicorn", "torch", "scipy", "numpy"]
    missing_py = []
    for mod in required_modules:
        try:
            __import__(mod)
        except ImportError:
            missing_py.append(mod)

    if missing_py:
        print(f"[!] Missing required Python modules: {', '.join(missing_py)}")
        print("    Run: pip install -r requirements.txt")
        return False

    # Check Node / npm
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    if not shutil.which(npm_cmd) and not shutil.which("npm"):
        print("[!] Node.js / npm not found in system PATH.")
        print("    Please install Node.js 18+ from https://nodejs.org/")
        return False

    # Check frontend/node_modules
    if not (FRONTEND_DIR / "node_modules").exists():
        print("[*] frontend/node_modules not found. Installing frontend dependencies...")
        res = subprocess.run([npm_cmd, "install"], cwd=str(FRONTEND_DIR), shell=(sys.platform == "win32"))
        if res.returncode != 0:
            print("[!] 'npm install' failed inside frontend/ directory.")
            return False

    return True


def wait_for_backend(url: str, timeout_sec: float = 20.0) -> bool:
    """Poll the backend health endpoint until it responds with HTTP 200."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AcoustiGuard-Runner/1.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Start AcoustiGuard AI Backend and Frontend servers.")
    parser.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    parser.add_argument("--frontend-port", type=int, default=5173, help="Frontend port (default: 5173)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web browser")
    args = parser.parse_args()

    if not check_prerequisites():
        sys.exit(1)

    npm_executable = "npm.cmd" if sys.platform == "win32" else "npm"
    processes = []

    def cleanup(signum=None, frame=None):
        print("\n[*] Stopping AcoustiGuard AI servers...")
        for proc in processes:
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True, check=False)
                else:
                    proc.terminate()
                    proc.wait(timeout=3)
            except Exception:
                pass
        print("[+] All processes stopped. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("\n" + "=" * 62)
    print("         AcoustiGuard AI — Application Launch")
    print("=" * 62)

    # 1. Start FastAPI Backend
    print(f"[*] Starting FastAPI backend on http://127.0.0.1:{args.port}...")
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    processes.append(backend_proc)

    # 2. Wait for Backend Health
    health_url = f"http://127.0.0.1:{args.port}/api/health"
    print(f"[*] Waiting for backend health check ({health_url})...")
    if not wait_for_backend(health_url, timeout_sec=20.0):
        print("[!] Backend failed to start within 20 seconds.")
        if backend_proc.stderr:
            err_output = backend_proc.stderr.read().decode("utf-8", errors="replace")
            print("Backend error log:\n", err_output)
        cleanup()
        sys.exit(1)

    print(f"[+] Backend is online and ready (PID: {backend_proc.pid})")

    # 3. Start Frontend Development Server
    print(f"[*] Starting React/Vite frontend on http://localhost:{args.frontend_port}...")
    frontend_cmd = [npm_executable, "run", "dev"]
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        shell=(sys.platform == "win32"),
    )
    processes.append(frontend_proc)

    # Give frontend a moment to bind port
    time.sleep(1.5)
    print(f"[+] Frontend dev server is active (PID: {frontend_proc.pid})")

    app_url = f"http://localhost:{args.frontend_port}"
    docs_url = f"http://127.0.0.1:{args.port}/docs"

    print("\n" + "-" * 62)
    print(f"  [>] Web Application:  {app_url}")
    print(f"  [>] Backend REST API: http://127.0.0.1:{args.port}")
    print(f"  [>] API Documentation: {docs_url}")
    print("-" * 62)
    print("  Application running. Press Ctrl+C in this terminal to stop.")
    print("-" * 62 + "\n")

    if not args.no_browser:
        try:
            webbrowser.open(app_url)
        except Exception:
            pass

    # Monitor processes until interrupted
    try:
        while True:
            if backend_proc.poll() is not None:
                print("[!] Backend process exited unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("[!] Frontend process exited unexpectedly.")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
