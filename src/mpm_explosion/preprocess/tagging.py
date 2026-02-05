from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import re


@dataclass
class ObjectTag:
    name: str
    material: str = "default"
    breakable: bool = False
    structural: bool = False


def build_object_tags(
    object_names: list[str],
    *,
    rules: Optional[Dict[str, Dict]] = None,
) -> Dict[str, ObjectTag]:
    """
    Build tags based on name matching.

    rules example:
      {
        "glass": {"material":"glass", "breakable":True},
        "wall": {"material":"concrete", "structural":True}
      }
    """
    rules = rules or {
        r"glass": {"material": "glass", "breakable": True, "structural": False},
        r"wall|pillar|beam": {"material": "concrete", "breakable": False, "structural": True},
    }

    out: Dict[str, ObjectTag] = {}
    for n in object_names:
        tag = ObjectTag(name=n)
        lower = n.lower()
        for pat, cfg in rules.items():
            if re.search(pat, lower):
                for k, v in cfg.items():
                    setattr(tag, k, v)
        out[n] = tag
    return out