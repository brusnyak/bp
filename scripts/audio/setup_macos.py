#!/usr/bin/env python3
"""macOS audio routing setup for BP — BlackHole 2ch + loopback.

Idempotent: safe to run multiple times. Verifies and reports the selected devices.

Usage:
    python scripts/audio/setup_macos.py           # install + verify
    python scripts/audio/setup_macos.py --check   # only verify
    python scripts/audio/setup_macos.py --json    # machine-readable output

Requires: Homebrew (for BlackHole install). Run with admin for install step.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class AudioDevice:
    name: str
    uid: str
    is_input: bool
    is_output: bool
    is_virtual: bool = False


@dataclass
class SetupResult:
    ok: bool
    blackhole_installed: bool
    loopback_available: bool
    virtual_mic_name: Optional[str] = None
    loopback_device: Optional[str] = None
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    error: Optional[str] = None
    commands_run: list[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["commands_run"] = self.commands_run or []
        return d


def run(cmd: list[str], check: bool = True, capture: bool = True) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=capture, text=True, check=check)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.CalledProcessError as e:
        if capture:
            return e.returncode, e.stdout or "", e.stderr or ""
        raise


def has_brew() -> bool:
    return shutil.which("brew") is not None


def blackhole_installed() -> bool:
    """Check if BlackHole 2ch is present in CoreAudio."""
    code, out, _ = run(["system_profiler", "SPAudioDataType"], check=False)
    return code == 0 and "BlackHole 2ch" in out


def install_blackhole() -> tuple[bool, str]:
    """Install BlackHole 2ch via Homebrew. Returns (success, message)."""
    if not has_brew():
        return False, "Homebrew not found. Install from https://brew.sh"
    code, out, err = run(["brew", "install", "blackhole-2ch"], check=False)
    if code == 0:
        return True, "BlackHole 2ch installed via Homebrew"
    return False, f"brew install failed: {err or out}"


def list_audio_devices() -> list[AudioDevice]:
    """Parse `system_profiler SPAudioDataType` for device list."""
    code, out, _ = run(["system_profiler", "SPAudioDataType"], check=False)
    if code != 0:
        return []
    
    devices = []
    current = {}
    for line in out.splitlines():
        line = line.strip()
        if line.endswith(":") and not line.startswith(" "):
            if current:
                devices.append(AudioDevice(**current))
            current = {"name": line[:-1], "uid": "", "is_input": False, "is_output": False}
        elif line.startswith("UID: "):
            current["uid"] = line[5:]
        elif line == "Input Channels: 2" or line == "Input Channels: 1":
            current["is_input"] = True
        elif line == "Output Channels: 2" or line == "Output Channels: 1":
            current["is_output"] = True
    if current:
        devices.append(AudioDevice(**current))
    
    # Mark virtual devices
    for d in devices:
        if "BlackHole" in d.name or "VB-Cable" in d.name or "null-sink" in d.name.lower():
            d.is_virtual = True
    return devices


def select_devices(devices: list[AudioDevice]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Select (virtual_mic, loopback_device, input_device) for BP.
    
    Returns names (not UIDs) as used by the browser's MediaDevices API.
    """
    virtual_mic = None
    loopback = None
    input_dev = None
    
    # 1. Virtual mic = BlackHole 2ch (output device that apps can use as mic input)
    for d in devices:
        if "BlackHole 2ch" in d.name and d.is_output:
            virtual_mic = d.name
            break
    
    # 2. Loopback = BlackHole 2ch (monitor of output) or built-in loopback
    # On macOS with BlackHole, the monitor is the same device used as input
    for d in devices:
        if "BlackHole 2ch" in d.name and d.is_input:
            loopback = d.name
            break
    
    # 3. Input = real microphone (not virtual)
    for d in devices:
        if d.is_input and not d.is_virtual and "Microphone" in d.name:
            input_dev = d.name
            break
    if not input_dev:
        for d in devices:
            if d.is_input and not d.is_virtual:
                input_dev = d.name
                break
    
    return virtual_mic, loopback, input_dev


def setup_macos(install: bool = True) -> SetupResult:
    """Main setup routine."""
    commands = []
    
    # Check BlackHole
    if not blackhole_installed():
        if install:
            ok, msg = install_blackhole()
            commands.append("brew install blackhole-2ch")
            if not ok:
                return SetupResult(
                    ok=False, blackhole_installed=False, loopback_available=False,
                    error=msg, commands_run=commands
                )
        else:
            return SetupResult(
                ok=False, blackhole_installed=False, loopback_available=False,
                error="BlackHole 2ch not installed. Run with --install or install manually: brew install blackhole-2ch",
                commands_run=commands
            )
    else:
        commands.append("BlackHole 2ch already installed")
    
    # List devices
    devices = list_audio_devices()
    if not devices:
        return SetupResult(
            ok=False, blackhole_installed=True, loopback_available=False,
            error="No audio devices found via system_profiler",
            commands_run=commands
        )
    
    virtual_mic, loopback, input_dev = select_devices(devices)
    
    result = SetupResult(
        ok=bool(virtual_mic),
        blackhole_installed=True,
        loopback_available=bool(loopback),
        virtual_mic_name=virtual_mic,
        loopback_device=loopback,
        input_device=input_dev,
        output_device=next((d.name for d in devices if "BlackHole" in d.name and d.is_output), None),
        commands_run=commands
    )
    
    if not virtual_mic:
        result.error = "BlackHole 2ch installed but not detected as audio device. Try: sudo launchctl kickstart -k system/com.apple.audio.coreaudiod"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="macOS audio routing setup for BP")
    parser.add_argument("--check", action="store_true", help="Only verify, don't install")
    parser.add_argument("--install", action="store_true", help="Force install (default if not --check)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()
    
    if platform.system() != "Darwin":
        print("This script is for macOS only.", file=sys.stderr)
        sys.exit(1)
    
    result = setup_macos(install=not args.check)
    
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"BlackHole installed: {result.blackhole_installed}")
        print(f"Loopback available: {result.loopback_available}")
        print(f"Virtual mic: {result.virtual_mic_name or 'NOT FOUND'}")
        print(f"Loopback device: {result.loopback_device or 'NOT FOUND'}")
        print(f"Input device (mic): {result.input_device or 'NOT FOUND'}")
        print(f"Output device: {result.output_device or 'NOT FOUND'}")
        if result.error:
            print(f"ERROR: {result.error}")
        if result.commands_run:
            print("\nCommands run:")
            for c in result.commands_run:
                print(f"  $ {c}")
    
    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()