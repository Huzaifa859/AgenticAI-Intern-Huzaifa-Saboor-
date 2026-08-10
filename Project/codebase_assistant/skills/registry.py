"""
registry.py
===========

Defines SkillRegistry, the central place agent skills are registered
and looked up by name.

This mirrors ToolRegistry but operates one level up: ToolRegistry holds
raw callables, SkillRegistry holds BaseSkill instances that carry their
own model-facing metadata.

TODO: Implement real registration, lookup, and invocation, and decide
how (or whether) skills are surfaced through ToolRegistry so agents
reach both through a single interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseSkill


class SkillRegistry:
    """
    Central registry for the skills available to agents.
    """

    def __init__(self) -> None:
        """Initialize an empty skill registry."""
        # Intended backing state: skill name -> skill instance.
        self._skills: Dict[str, BaseSkill] = {}

    def register_skill(self, skill: BaseSkill) -> None:
        """
        Register a skill under its declared `name`.

        Args:
            skill: The BaseSkill instance to register.

        TODO: Implement real registration, including duplicate-name and
        empty-name validation.
        """
        # TODO: implement real skill registration
        pass

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """
        Retrieve a registered skill by name.

        Args:
            name: Name of the skill to retrieve.

        Returns:
            The skill if found, otherwise None (placeholder always
            returns None).

        TODO: Implement real lookup.
        """
        # TODO: implement real skill lookup
        return None

    def list_skills(self) -> List[Dict[str, Any]]:
        """
        List metadata for all registered skills.

        Returns:
            A list of skill descriptors (placeholder empty list).

        TODO: Return each registered skill's `describe()` output.
        """
        # TODO: implement real skill listing
        return []
