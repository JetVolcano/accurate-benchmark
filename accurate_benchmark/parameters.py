from collections.abc import Iterator
from typing import Any


class SingleParam:
    def __init__(self, value: Any) -> None:
        self.__value: Any = value

    @property
    def value(self) -> Any:
        return self.__value

    @value.setter
    def value(self, value: Any) -> None:
        if not isinstance(value, type(self.__value)):
            raise TypeError(
                f"Expected value of type {type(self.__value).__name__}, got {type(value).__name__}"
            )
        self.__value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__value!r})"

    def __iter__(self) -> Iterator:
        return iter(self.__value)

    def __hash__(self) -> int:
        return hash(self.__value)

    def __bool__(self) -> bool:
        return bool(self.value)

    def __sizeof__(self) -> int:
        return self.value.__sizeof__() if hasattr(self.value, "__sizeof__") else 0

    def __or__(self, other: Any) -> Any:
        return (
            self.value | other.value
            if isinstance(other, SingleParam)
            else self.value | other
        )

    def __ror__(self, value: Any) -> Any:
        return (
            value.value | self.value if isinstance(value, SingleParam) else value | self.value
        )

    def __and__(self, other: Any) -> Any:
        return (
            self.value & other.value
            if isinstance(other, SingleParam)
            else self.value & other
        )

    def __rand__(self, value: Any) -> Any:
        return (
            value.value & self.value if isinstance(value, SingleParam) else value & self.value
        )

    def __xor__(self, other: Any) -> Any:
        return (
            self.value ^ other.value
            if isinstance(other, SingleParam)
            else self.value ^ other
        )

    def __rxor__(self, value: Any) -> Any:
        return (
            value.value ^ self.value if isinstance(value, SingleParam) else value ^ self.value
        )

    def __invert__(self) -> Any:
        return ~self.value if isinstance(self.value, int) else NotImplemented

    def __eq__(self, other: Any) -> bool:
        return (
            self.value == other.value
            if isinstance(other, SingleParam)
            else self.value == other
        )

    def __ne__(self, other: Any) -> bool:
        return (
            self.value != other.value
            if isinstance(other, SingleParam)
            else self.value != other
        )

    def __lt__(self, other: Any) -> bool:
        return (
            self.value < other.value
            if isinstance(other, SingleParam)
            else self.value < other
        )

    def __le__(self, other: Any) -> bool:
        return (
            self.value <= other.value
            if isinstance(other, SingleParam)
            else self.value <= other
        )

    def __gt__(self, other: Any) -> bool:
        return (
            self.value > other.value
            if isinstance(other, SingleParam)
            else self.value > other
        )

    def __ge__(self, other: Any) -> bool:
        return (
            self.value >= other.value
            if isinstance(other, SingleParam)
            else self.value >= other
        )

    def __add__(self, other: Any) -> Any:
        return (
            self.value + other.value
            if isinstance(other, SingleParam)
            else self.value + other
        )

    def __sub__(self, other: Any) -> Any:
        return (
            self.value - other.value
            if isinstance(other, SingleParam)
            else self.value - other
        )

    def __mul__(self, other: Any) -> Any:
        return (
            self.value * other.value
            if isinstance(other, SingleParam)
            else self.value * other
        )

    def __div__(self, other: Any) -> Any:
        return (
            self.value / other.value
            if isinstance(other, SingleParam)
            else self.value / other
        )

    def __trudiv__(self, other: Any) -> Any:
        return (
            self.value // other.value
            if isinstance(other, SingleParam)
            else self.value // other
        )

    def __mod__(self, other: Any) -> Any:
        return (
            self.value % other.value
            if isinstance(other, SingleParam)
            else self.value % other
        )

    def __pow__(self, other: Any) -> Any:
        return (
            self.value**other.value
            if isinstance(other, SingleParam)
            else self.value**other
        )
