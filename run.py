"""
Multi-Modal RAG Platform - Launch Script
Launches the FastAPI backend and serves the high-performance React UI.
Usage:
    python run.py
"""

import os
import sys
import time
import webbrowser
import subprocess
import uvicorn

def main():
    print("=" * 65)
    print("🧠 Starting Multi-Modal RAG Platform (React + FastAPI)")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(base_dir, "frontend", "dist")

    # Check if frontend production bundle is built
    if not os.path.exists(dist_dir) or not os.path.exists(os.path.join(dist_dir, "index.html")):
        print("⚡ Building modern React frontend for first-time launch...")
        frontend_dir = os.path.join(base_dir, "frontend")
        try:
            subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True, shell=True)
            print("✅ React frontend build complete.")
        except Exception as e:
            print(f"⚠️ Warning: Could not automatically build frontend: {e}")
            print("You can build manually with: cd frontend && npm run build")

    url = "http://127.0.0.1:8000"
    print(f"\n🚀 Multi-Modal RAG Platform ready at: {url}")
    print("   • React Modern UI:         http://127.0.0.1:8000")
    print("   • Interactive OpenAPI Docs: http://127.0.0.1:8000/docs")
    print("=" * 65)

    # Open browser automatically after a short delay
    def open_browser():
        time.sleep(1.8)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Run Uvicorn server
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
