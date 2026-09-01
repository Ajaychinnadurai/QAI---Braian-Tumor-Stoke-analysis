"""
One-Click Application Launcher
Starts the Healthcare Analytics API & Web Dashboard Server
"""

import os
import sys
import webbrowser
from backend.server import start_server

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    print("\n============================================================")
    print("QuantumHealth AI - Predictive Healthcare Analytics")
    print("Based on: Base1.pdf (Elsevier, 2026)")
    print("Real Datasets: 5,110 Stroke Records & 253 Brain MRI Scans")
    print(f"Dashboard URL: http://localhost:{port}")
    print("============================================================\n")

    # Launch server
    start_server(port=port)
