from collections.abc import Iterator
from typing import Any


class SingleParam[T_co]:
    def __init__(self, value: T_co) -> None:
        self.__value: T_co = value

    @property
    def value(self) -> T_co:
        return self.__value

    @value.setter
    def value(self, value: Any) -> None:
        if not isinstance(value, type(self.__value)):
            raise TypeError(
                f"Expected value of type {type(self.__value).__name__}, got {type(value).__name__}"
            )
        self.__value = value

    def __iter__(self) -> Iterator:
        yield self.__value

    def __repr__(self) -> str:
        return f"{self.__value!r}"

    def __hash__(self) -> int:
        return hash(self.__value)

    def __bool__(self) -> bool:
        return bool(self.value)
