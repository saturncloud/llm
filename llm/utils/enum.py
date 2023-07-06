from enum import Enum
from typing import List


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    @classmethod
    def values(cls) -> List[str]:
        return [str(value) for value in list(cls)]  # type: ignore
