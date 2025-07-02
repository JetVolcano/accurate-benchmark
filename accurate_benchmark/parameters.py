from collections.abc import Iterator
from typing import Any


class SingleParam:
    def __init__(self, value: Any) -> None:
        self.__value: Any = value

    @property
    def value(self) -> Any:
        """
        Returns the value of the parameter.

        :returntype Any:
        """
        return self.__value

    @value.setter
    def value(self, value: Any) -> None:
        """
        Sets the value of the parameter.

        :param value: The new value to set.
        :returntype None:
        """
        if not isinstance(value, type(self.__value)):
            raise TypeError(
                f"Expected value of type {type(self.__value).__name__}, got {type(value).__name__}"
            )
        self.__value = value

    def __repr__(self) -> str:
        """
        Returnns the string representation of the parameter.

        :returntype str:
        """
        return f"{self.__class__.__name__}({self.__value!r})"

    def __iter__(self) -> Iterator:
        return iter(self.__value)

    def __bool__(self) -> bool:
        """
        Returns the boolean value of the parameter's value.

        :returntype bool:
        """
        return bool(self.value)

    def __sizeof__(self) -> int:
        """
        Returns the size of the parameter's value in bytes.

        :returntype int:
        """
        return self.value.__sizeof__() if hasattr(self.value, "__sizeof__") else 0

    def __hash__(self) -> int:
        """
        Returns the hash of the parameter's value.

        :returntype int:
        """
        return hash(self.__value)

    def __or__(self, other: Any) -> Any:
        """
        Performs a bitwise OR operation with another parameter or value.

        :param other: The other parameter or value to perform the operation with.
        :returntype Any:
        """
        return (
            self.value | other.value
            if isinstance(other, SingleParam)
            else self.value | other
        )

    def __ror__(self, value: Any) -> Any:
        """
        Performs a bitwise OR operation with the parameter and another value.

        :param value: The value to perform the operation with.
        :returntype Any:
        """
        return (
            value | self.value if isinstance(value, SingleParam) else value | self.value
        )

    def __and__(self, other: Any) -> Any:
        """
        Performs a bitwise AND operation with another parameter or value.

        :param other: The other parameter or value to perform the operation with.
        :returntype Any:
        """
        return (
            self.value & other.value
            if isinstance(other, SingleParam)
            else self.value & other
        )

    def __rand__(self, value: Any) -> Any:
        """
        Performs a bitwise AND operation with the parameter and another value.

        :param value: The value to perform the operation with.
        :returntype Any:
        """
        return (
            value & self.value if isinstance(value, SingleParam) else value & self.value
        )

    def __xor__(self, other: Any) -> Any:
        """
        Performs a bitwise XOR operation with another parameter or value.

        :param other: The other parameter or value to perform the operation with.
        :returntype Any:
        """
        return (
            self.value ^ other.value
            if isinstance(other, SingleParam)
            else self.value ^ other
        )

    def __rxor__(self, value: Any) -> Any:
        """
        Performs a bitwise XOR operation with the parameter and another value.

        :param value: The value to perform the operation with.
        :returntype Any:
        """
        return (
            value ^ self.value if isinstance(value, SingleParam) else value ^ self.value
        )

    def __invert__(self) -> Any:
        """
        Performs a bitwise NOT operation on the parameter's value.

        :returntype Any:
        """
        return ~self.value if isinstance(self.value, int) else NotImplemented

    def __eq__(self, other: Any) -> bool:
        """
        Checks if the parameter is equal to another parameter or value.

        :param other: The other parameter or value to compare with.
        :returntype bool:
        """
        return (
            self.value == other.value
            if isinstance(other, SingleParam)
            else self.value == other
        )

    def __ne__(self, other: Any) -> bool:
        """
        Checks if the parameter is not equal to another parameter or value.

        :param other: The other parameter or value to compare wiht.
        :returntype bool:
        """
        return (
            self.value != other.value
            if isinstance(other, SingleParam)
            else self.value != other
        )

    def __lt__(self, other: Any) -> bool:
        """
        Checks if the parameter is less than another parameter or value.

        :param other: The other parameter or value to compare with.
        :returntype bool:
        """
        return (
            self.value < other.value
            if isinstance(other, SingleParam)
            else self.value < other
        )

    def __le__(self, other: Any) -> bool:
        """
        Checks if the parameter is less than or equal to another parameter or value.

        :param other: The other parameter or value to compare with.
        :returntype bool:
        """
        return (
            self.value <= other.value
            if isinstance(other, SingleParam)
            else self.value <= other
        )

    def __gt__(self, other: Any) -> bool:
        """
        Checks if the parameter is greater than another parameter or value.

        :param other: The other parameter or value to compare with.
        :returntype bool:
        """
        return (
            self.value > other.value
            if isinstance(other, SingleParam)
            else self.value > other
        )

    def __ge__(self, other: Any) -> bool:
        """
        Checks if the parameter is greater than or equal to another parameter or value.

        :param other: The other parameter or value to compare with.
        :returntype bool:
        """
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
