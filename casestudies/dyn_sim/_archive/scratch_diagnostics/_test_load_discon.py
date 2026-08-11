"""Diagnostic: try to load the ZMQ-enabled libDISCON.dll and report the exact
Windows error if a dependency is missing (so we know which DLL to deploy)."""

import ctypes
import os
import sys

PROJECT = r"c:\Users\augusth\OneDrive - SINTEF\Dokumenter\Sommerprosjekt\sommer2026\tops-openfast-sommer2026"
DISCON = os.path.join(PROJECT, "test1002", "ControlData", "libDISCON.dll")

dirs = [
    r"C:\Users\augusth\rosco_build\zmq_deploy",
]
# Optionally also expose the full mingw64 runtime (pass "mingw" as an argument).
if len(sys.argv) > 1 and sys.argv[1] == "mingw":
    dirs.append(r"C:\msys64\mingw64\bin")

for d in dirs:
    if os.path.isdir(d):
        os.add_dll_directory(d)
        print(f"added dll dir: {d}")

print(f"loading: {DISCON}")
try:
    h = ctypes.WinDLL(DISCON)
    print("RESULT: LOADED OK ->", h._handle)
except OSError as e:
    print("RESULT: LOAD FAILED ->", repr(e))
