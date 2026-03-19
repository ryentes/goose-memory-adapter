from typing import Literal, get_args

MemoryType = Literal["episodic", "semantic", "entity", "task"]
VALID_MEMORY_TYPES = set(get_args(MemoryType))