"""Relocate historical provenance paths without changing the original records."""
import json
from pathlib import Path


def resolve_recorded_path(value, root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"Parent traversal in recorded path: {value}")
    if not path.is_absolute():
        return root / path
    mapping = root / "config/artifact_paths.json"
    if mapping.exists():
        config = json.loads(mapping.read_text())
        prefixes = config["original_roots"]
        for original in sorted(prefixes, key=len, reverse=True):
            try:
                relative = path.relative_to(original)
            except ValueError:
                continue
            destination = (root / prefixes[original]).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Artifact mapping escapes package root")
            return destination / relative
        if config.get("strict", True):
            raise ValueError(f"Unmapped historical path: {value}")
    # The private working checkout can still read its original external records.
    # A release always installs the strict mapping above, so this fallback is off.
    return path
