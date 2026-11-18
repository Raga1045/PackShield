"""Entry point to run DefektGuard Flask backend."""

import os
import sys
from pathlib import Path

# Ensure backend directory is in python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from __init__ import create_app

app = create_app()

if __name__ == "__main__":
    port = config.PORT
    debug = config.DEBUG
    print(f"🚀 DefektGuard Backend running on http://0.0.0.0:{port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
