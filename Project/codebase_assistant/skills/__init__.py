"""
skills
======

Agent skills — named, described capabilities an agent can invoke.

Skills sit one level above tools. A *tool* is a raw I/O primitive
(read a file, search text); a *skill* is an agent-facing capability
that may compose several tools plus model calls into one unit, and
carries the description/parameter metadata a model needs in order to
choose it.

Contains:
- BaseSkill: abstract interface every skill implements.
- SkillRegistry: registration and lookup of available skills.

NOTE: Placeholder only. No skill is implemented and the registry
performs no real registration yet.
"""

from .base import BaseSkill
from .registry import SkillRegistry

__all__ = ["BaseSkill", "SkillRegistry"]
