from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, Dict, List


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    @classmethod
    def values(cls) -> List[str]:
        return [str(value) for value in list(cls)]  # type: ignore


@dataclass
class DataclassBase:
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        _data = {}
        for field in fields(cls):
            if field.name in data:
                if isinstance(field.type, type) and issubclass(field.type, DataclassBase):
                    value = field.type.from_dict(data[field.name])
                else:
                    value = data[field.name]
                _data[field.name] = value
        return cls(**_data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
