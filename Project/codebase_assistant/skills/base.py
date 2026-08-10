"""
base.py
=======

Defines BaseSkill, the abstract interface every agent skill implements.

TODO: Implement the concrete skills named in the milestone plan —
`read_file`, `search_codebase`, and `explain_function` — as subclasses
of BaseSkill, and register them via SkillRegistry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseSkill(ABC):
    """
    Abstract base class for all agent skills.

    Subclasses declare a `name` and `description` (the metadata a model
    sees when choosing a skill) and implement `execute`.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """
        Run the skill.

        Args:
            **kwargs: Skill-specific arguments.

        Returns:
            The skill's result.

        TODO: Implement in each concrete subclass.
        """
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        """
        Describe this skill for model-facing tool selection.

        Returns:
            A dict carrying the skill's name and description.

        TODO: Include a JSON-Schema parameter definition once skills
        declare their arguments.
        """
        # TODO: add parameter schema alongside name/description
        return {"name": self.name, "description": self.description}
