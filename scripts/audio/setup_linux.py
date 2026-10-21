#!/usr/bin/env python3
"""Linux audio routing setup for BP — PipeWire null-sink + loopback.

Idempotent: safe to run multiple times. Verifies and reports the selected devices.

Usage:
    python scripts/audio/setup_linux.py           # create + verify
    python scripts/audio/setup_linux.py --check   # only verify
    python scripts/audio/setup_linux.py --json    # machine-readable output

Requires: PipeWire (pactl), pw-cli. Run as user (not root).
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
    description: str
    is_input: bool
    is_output: bool
    is_virtual: bool = False
    module: Optional[str] = None


@dataclass
class SetupResult:
    ok: bool
    pipewire_running: bool
    null_sink_created: bool
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


def has_pipewire() -> bool:
    """Check if PipeWire is running."""
    code, out, _ = run(["pactl", "info"], check=False)
    return code == 0 and "PipeWire" in out


def get_pactl_sources() -> list[AudioDevice]:
    """Parse `pactl list short sources`."""
    code, out, _ = run(["pactl", "list", "short", "sources"], check=False)
    if code != 0:
        return []
    devices = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            idx, name, driver, fmt, state = parts[:5]
            desc = ""
            # Get description from full list
            code2, out2, _ = run(["pactl", "list", "sources"], check=False)
            for block in out2.split("\n\n"):
                if f"Name: {name}" in block:
                    for l in block.splitlines():
                        if "Description:" in l:
                            desc = l.split("Description:", 1)[1].strip()
                            break
                    break
            devices.append(AudioDevice(
                name=name, description=desc, is_input=True, is_output=False,
                is_virtual="null-sink" in name or "loopback" in name.lower()
            ))
    return devices


def get_pactl_sinks() -> list[AudioDevice]:
    """Parse `pactl list short sinks`."""
    code, out, _ = run(["pactl", "list", "short", "sinks"], check=False)
    if code != 0:
        return []
    devices = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            idx, name, driver, fmt, state = parts[:5]
            desc = ""
            code2, out2, _ = run(["pactl", "list", "sinks"], check=False)
            for block in out2.split("\n\n"):
                if f"Name: {name}" in block:
                    for l in block.splitlines():
                        if "Description:" in l:
                            desc = l.split("Description:", 1)[1].strip()
                            break
                    break
            devices.append(AudioDevice(
                name=name, description=desc, is_input=False, is_output=True,
                is_virtual="null-sink" in name or "loopback" in name.lower()
            ))
    return devices


def null_sink_exists(name: str = "bp_virtual_sink") -> bool:
    """Check if our null-sink already exists."""
    sinks = get_pactl_sinks()
    return any(name in s.name for s in sinks)


def create_null_sink(name: str = "bp_virtual_sink") -> tuple[bool, str]:
    """Create a PipeWire null-sink for virtual mic + loopback."""
    # Create null-sink (virtual output that can be monitored)
    code, out, err = run([
        "pactl", "load-module", "module-null-sink",
        f"sink_name={name}",
        f"sink_properties=device.description='BP Virtual Sink'"
    ], check=False)
    if code != 0:
        return False, f"Failed to create null-sink: {err or out}"
    
    # Create remap-source so other apps see it as a proper input device
    remap_name = f"{name}_mic"
    code, out, err = run([
        "pactl", "load-module", "module-remap-source",
        f"source_name={remap_name}",
        f"master={name}.monitor",
        f"source_properties=device.description='BP Virtual Mic'"
    ], check=False)
    if code != 0:
        # Clean up null-sink on failure
        run(["pactl", "unload-module", out.strip()], check=False)
        return False, f"Failed to create remap-source: {err or out}"
    
    return True, f"Created {name} + {remap_name}"


def remove_null_sink(name: str = "bp_virtual_sink") -> None:
    """Remove our null-sink and remap-source."""
    sinks = get_pactl_sinks()
    for s in sinks:
        if name in s.name and s.module:
            run(["pactl", "unload-module", s.module], check=False)
    sources = get_pactl_sources()
    for s in sources:
        if name in s.name and s.module:
            run(["pactl", "unload-module", s.module], check=False)


def select_devices(sources: list[AudioDevice], sinks: list[AudioDevice]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Select (virtual_mic, loopback_device, input_device) for BP."""
    virtual_mic = None
    loopback = None
    input_dev = None
    
    # Virtual mic = remap-source of our null-sink (apps use this as mic input)
    for s in sources:
        if "bp_virtual_sink_mic" in s.name or "BP Virtual Mic" in s.description:
            virtual_mic = s.name
            break
    
    # Loopback = monitor of our null-sink (for capturing system audio)
    for s in sources:
        if "bp_virtual_sink.monitor" in s.name:
            loopback = s.name
            break
    
    # Input = real microphone (not virtual)
    for s in sources:
        if s.is_input and not s.is_virtual and ("mic" in s.name.lower() or "microphone" in s.description.lower()):
            input_dev = s.name
            break
    if not input_dev:
        for s in sources:
            if s.is_input and not s.is_virtual:
                input_dev = s.name
                break
    
    return virtual_mic, loopback, input_dev


def setup_linux(remove: bool = False) -> SetupResult:
    """Main setup routine."""
    commands = []
    
    if not has_pipewire():
        return SetupResult(
            ok=False, pipewire_running=False, null_sink_created=False,
            loopback_available=False, error="PipeWire not running (pactl info failed)",
            commands_run=commands
        )
    commands.append("PipeWire detected via pactl")
    
    if remove:
        remove_null_sink()
        commands.append("pactl unload-module (bp_virtual_sink*)")
        return SetupResult(
            ok=True, pipewire_running=True, null_sink_created=False,
            loopback_available=False, commands_run=commands
        )
    
    # Create or verify null-sink
    if not null_sink_exists():
        ok, msg = create_null_sink()
        commands.append(f"pactl load-module module-null-sink sink_name=bp_virtual_sink")
        commands.append(f"pactl load-module module-remap-source master=bp_virtual_sink.monitor source_name=bp_virtual_sink_mic")
        if not ok:
            return SetupResult(
                ok=False, pipewire_running=True, null_sink_created=False,
                loopback_available=False, error=msg, commands_run=commands
            )
        commands.append(f"Created: {msg}")
    else:
        commands.append("bp_virtual_sink already exists")
    
    # List devices
    sources = get_pactl_sources()
    sinks = get_pactl_sinks()
    
    virtual_mic, loopback, input_dev = select_devices(sources, sinks)
    
    result = SetupResult(
        ok=bool(virtual_mic and loopback),
        pipewire_running=True,
        null_sink_created=True,
        loopback_available=bool(loopback),
        virtual_mic_name=virtual_mic,
        loopback_device=loopback,
        input_device=input_dev,
        output_device=next((s.name for s in sinks if "bp_virtual_sink" in s.name), None),
        commands_run=commands
    )
    
    if not virtual_mic:
        result.error = "Virtual mic (bp_virtual_sink_mic) not found after setup"
    elif not loopback:
        result.error = "Loopback (bp_virtual_sink.monitor) not found after setup"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Linux audio routing setup for BP (PipeWire)")
    parser.add_argument("--check", action="store_true", help="Only verify, don't create")
    parser.add_argument("--remove", action="store_true", help="Remove BP null-sink")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()
    
    if platform.system() != "Linux":
        print("This script is for Linux only.", file=sys.stderr)
        sys.exit(1)
    
    if args.check:
        # Just verify
        result = SetupResult(
            ok=False, pipewire_running=has_pipewire(), null_sink_created=null_sink_exists(),
            loopback_available=False, commands_run=[]
        )
        if result.pipewire_running:
            sources = get_pactl_sources()
            virtual_mic, loopback, input_dev = select_devices(sources, get_pactl_sinks())
            result.virtual_mic_name = virtual_mic
            result.loopback_device = loopback
            result.input_device = input_dev
            result.loopback_available = bool(loopback)
            result.ok = bool(virtual_mic and loopback)
    else:
        result = setup_linux(remove=args.remove)
    
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"PipeWire running: {result.pipewire_running}")
        print(f"Null-sink created: {result.null_sink_created}")
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