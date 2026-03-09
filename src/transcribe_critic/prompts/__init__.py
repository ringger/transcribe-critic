"""Prompt resource loader for transcribe-critic LLM stages."""

from __future__ import annotations

import sys
from functools import lru_cache
from importlib import resources

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@lru_cache(maxsize=None)
def load_prompt(name: str) -> dict[str, str]:
    """Load a prompt TOML file by name, returning a dict of prompt strings.

    Each TOML file contains keys like ``primary``, ``retry``, ``system``
    whose values are the prompt templates.  Use :func:`str.format` or
    f-string interpolation on the returned strings to fill placeholders.
    """
    prompt_file = resources.files("transcribe_critic.prompts") / f"{name}.toml"
    text = prompt_file.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    return data
