r"""Windows performance snapshot for the remote CST machine.

Purpose (2026-08-06): the spoke-and-wheel extraction died with "Could not
compute preconditioner" while CST reported only 3.17 GB of 31.94 GB free
BEFORE the first solve -- something is holding ~28 GB.  Known suspects, in
order of likelihood on this machine:
  * orphaned CST solver processes (Solver_HF_Tet_FD_AMD64): killing a
    Python extraction script does NOT abort the CST solve it launched --
    live-verified twice on the local machine on 2026-08-06;
  * multiple CST Design Environment instances, each holding open projects
    with meshes/results in RAM;
  * stale Python extraction processes;
  * another user's jobs (lab machine).

This script takes one snapshot and writes it BOTH to stdout and to
`remote_performance_report.txt` next to this file -- the repo folder is
OneDrive-synced, so once sync completes the report can be read directly
from the other machine.  Uses only the Python standard library plus
PowerShell/CIM commands available on any Windows 10/11 box.  Every section
is fault-tolerant: a failed probe prints the error and moves on.

Run on the remote machine (any Python 3):
    python analyze_remote_performance.py
"""
import ctypes
import ctypes.wintypes
import datetime
import platform
import subprocess
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent / "remote_performance_report.txt"
_lines = []


def emit(s=""):
    print(s, flush=True)
    _lines.append(s)


def section(title):
    emit()
    emit("=" * 72)
    emit(title)
    emit("=" * 72)


def ps(cmd, timeout=60):
    """Run one PowerShell command, return stdout (or an error marker)."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "").rstrip()
        if r.returncode != 0 and not out:
            return f"<powershell failed rc={r.returncode}: " \
                   f"{(r.stderr or '').strip()[:300]}>"
        return out
    except Exception as e:                                # noqa: BLE001
        return f"<probe failed: {e}>"


def memory_status():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.wintypes.DWORD),
                    ("dwMemoryLoad", ctypes.wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st


def main():
    emit(f"remote performance snapshot -- "
         f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    emit(f"host: {platform.node()}  ({platform.platform()})")
    emit(f"python: {sys.version.split()[0]}  ({sys.executable})")

    section("A. MEMORY (the headline number)")
    try:
        st = memory_status()
        gb = 1024 ** 3
        emit(f"physical RAM : {st.ullTotalPhys / gb:6.2f} GB total, "
             f"{st.ullAvailPhys / gb:6.2f} GB available "
             f"({st.dwMemoryLoad}% in use)")
        emit(f"commit charge: {(st.ullTotalPageFile - st.ullAvailPageFile) / gb:6.2f} "
             f"GB used of {st.ullTotalPageFile / gb:6.2f} GB limit")
        if st.ullAvailPhys / gb < 8:
            emit(">>> LOW available RAM: a multi-million-cell FD solve needs "
                 "well over 8 GB free. Find the holder below. <<<")
    except Exception as e:                                # noqa: BLE001
        emit(f"<memory probe failed: {e}>")

    section("B. TOP 25 PROCESSES BY RESIDENT MEMORY (working set)")
    emit(ps(
        "Get-Process | Sort-Object WorkingSet64 -Descending | "
        "Select-Object -First 25 "
        "@{n='WS_MB';e={[int]($_.WorkingSet64/1MB)}}, "
        "@{n='PrivMB';e={[int]($_.PrivateMemorySize64/1MB)}}, "
        "Id, ProcessName, "
        "@{n='Started';e={try{$_.StartTime.ToString('MM-dd HH:mm')}catch{''}}} | "
        "Format-Table -AutoSize | Out-String -Width 160"))

    section("C. CST-RELATED PROCESSES (orphaned solvers show up here)")
    emit(ps(
        "$p = Get-Process | Where-Object {$_.ProcessName -match "
        "'cst|solver|amds|frontend'}; "
        "if ($p) { $p | Select-Object "
        "@{n='WS_MB';e={[int]($_.WorkingSet64/1MB)}}, Id, ProcessName, "
        "@{n='CPU_s';e={[int]$_.CPU}}, "
        "@{n='Started';e={try{$_.StartTime.ToString('MM-dd HH:mm')}catch{''}}} | "
        "Format-Table -AutoSize | Out-String -Width 160 } "
        "else { 'no CST-related processes running' }"))
    emit("command lines (which project each CST process belongs to):")
    emit(ps(
        "Get-CimInstance Win32_Process | Where-Object {$_.Name -match "
        "'cst|Solver|AMDS'} | Select-Object ProcessId, "
        "@{n='Cmd';e={($_.CommandLine -replace '\\s+',' ')}} | "
        "Format-List | Out-String -Width 300"))

    section("D. PYTHON PROCESSES (stale extraction scripts)")
    emit(ps(
        "$q = Get-CimInstance Win32_Process | Where-Object "
        "{$_.Name -match 'python'}; if ($q) { $q | Select-Object ProcessId, "
        "@{n='WS_MB';e={[int]($_.WorkingSetSize/1MB)}}, "
        "@{n='Cmd';e={($_.CommandLine -replace '\\s+',' ')}} | "
        "Format-List | Out-String -Width 300 } "
        "else { 'no python processes' }"))

    section("E. CPU")
    emit(ps(
        "Get-CimInstance Win32_Processor | Select-Object Name, "
        "NumberOfCores, NumberOfLogicalProcessors, LoadPercentage | "
        "Format-Table -AutoSize | Out-String -Width 160"))
    emit("3-second load samples (_Total %):")
    emit(ps(
        "(Get-Counter '\\Processor(_Total)\\% Processor Time' "
        "-SampleInterval 1 -MaxSamples 3).CounterSamples | "
        "ForEach-Object {[int]$_.CookedValue} | Out-String", timeout=30))

    section("F. TOP 10 PROCESSES BY TOTAL CPU TIME")
    emit(ps(
        "Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 "
        "@{n='CPU_min';e={[int]($_.CPU/60)}}, Id, ProcessName, "
        "@{n='WS_MB';e={[int]($_.WorkingSet64/1MB)}} | "
        "Format-Table -AutoSize | Out-String -Width 160"))

    section("G. DISKS (CST scratch needs headroom)")
    emit(ps(
        "Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | "
        "Select-Object DeviceID, "
        "@{n='Free_GB';e={[int]($_.FreeSpace/1GB)}}, "
        "@{n='Total_GB';e={[int]($_.Size/1GB)}} | "
        "Format-Table -AutoSize | Out-String -Width 120"))

    section("H. LOGGED-IN SESSIONS (other users' jobs?)")
    try:
        r = subprocess.run(["qwinsta"], capture_output=True, text=True,
                           timeout=20)
        emit(r.stdout.rstrip() or "<no output>")
    except Exception as e:                                # noqa: BLE001
        emit(f"<qwinsta failed: {e}>")

    section("I. GPU (informational; hardware acceleration currently off)")
    emit(ps("try { nvidia-smi --query-gpu=name,memory.total,memory.used,"
            "utilization.gpu --format=csv } catch { 'no nvidia-smi' }",
            timeout=30))

    section("J. UPTIME")
    emit(ps(
        "$b = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime; "
        "'last boot: ' + $b.ToString('yyyy-MM-dd HH:mm') + "
        "'   up: ' + [string][int]((Get-Date)-$b).TotalHours + ' h'"))

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    emit()
    emit(f"report written to: {REPORT}")
    emit("(this folder is OneDrive-synced -- once sync completes, the "
         "report is readable from the other machine)")


if __name__ == "__main__":
    main()
