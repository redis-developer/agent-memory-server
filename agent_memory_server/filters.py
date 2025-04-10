from typing import Self

from pydantic import BaseModel
from pydantic.functional_validators import model_validator
from redisvl.query.filter import FilterExpression, Num, Tag


class TagFilter(BaseModel):
    field: str
    eq: str | None = None
    ne: str | None = None
    any: list[str] | None = None
    all: list[str] | None = None

    @model_validator(mode="after")
    def validate_filters(self) -> Self:
        if self.eq is not None and self.ne is not None:
            raise ValueError("eq and ne cannot both be set")
        if self.any is not None and self.all is not None:
            raise ValueError("any and all cannot both be set")
        if self.all is not None and len(self.all) == 0:
            raise ValueError("all cannot be an empty list")
        if self.any is not None and len(self.any) == 0:
            raise ValueError("any cannot be an empty list")
        return self

    def to_filter(self) -> FilterExpression:
        if self.eq is not None:
            return Tag(self.field) == self.eq
        if self.ne is not None:
            return Tag(self.field) != self.ne
        if self.any is not None:
            return Tag(self.field) == self.any
        if self.all is not None:
            return Tag(self.field) == self.all
        raise ValueError("No filter provided")


class NumFilter(BaseModel):
    field: str
    gt: int | None = None
    lt: int | None = None
    gte: int | None = None
    lte: int | None = None
    eq: int | None = None
    ne: int | None = None
    between: list[float] | None = None
    inclusive: str = "both"

    @model_validator(mode="after")
    def validate_filters(self) -> Self:
        if self.between is not None and len(self.between) != 2:
            raise ValueError("between must be a list of two numbers")
        if self.between is not None and self.eq is not None:
            raise ValueError("between and eq cannot both be set")
        if self.between is not None and self.ne is not None:
            raise ValueError("between and ne cannot both be set")
        if self.between is not None and self.gt is not None:
            raise ValueError("between and gt cannot both be set")
        if self.between is not None and self.lt is not None:
            raise ValueError("between and lt cannot both be set")
        if self.between is not None and self.gte is not None:
            raise ValueError("between and gte cannot both be set")
        return self

    def to_filter(self) -> FilterExpression:
        if self.between is not None:
            return Num(self.field).between(
                int(self.between[0]), int(self.between[1]), self.inclusive
            )
        if self.eq is not None:
            return Num(self.field) == self.eq
        if self.ne is not None:
            return Num(self.field) != self.ne
        if self.gt is not None:
            return Num(self.field) > self.gt
        if self.lt is not None:
            return Num(self.field) < self.lt
        if self.gte is not None:
            return Num(self.field) >= self.gte
        if self.lte is not None:
            return Num(self.field) <= self.lte
        raise ValueError("No filter provided")


class SessionId(TagFilter):
    field: str = "session_id"


class UserId(TagFilter):
    field: str = "user_id"


class Namespace(TagFilter):
    field: str = "namespace"


class CreatedAt(NumFilter):
    field: str = "created_at"


class LastAccessed(NumFilter):
    field: str = "last_accessed"


class Topics(TagFilter):
    field: str = "topics"


class Entities(TagFilter):
    field: str = "entities"
