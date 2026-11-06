"""Opt-in language-pack profiles; no runtime/model imports."""
from copy import deepcopy


def resolve_profile(pack, name="default"):
    resolved = deepcopy(pack)
    if name == "default":
        return resolved, name
    profiles = pack.get("profiles", {})
    if name not in profiles:
        raise ValueError(f"Unknown profile {name!r} for {pack.get('pair', 'unconfigured route')}")
    for stage in ("stt", "vad"):
        resolved.setdefault(stage, {}).update(profiles[name].get(stage, {}))
    return resolved, name
