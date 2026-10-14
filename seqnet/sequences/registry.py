"""Name -> factory registries that build components from JSON/YAML specs ``{"kind": <name>, **params}``.

Every pluggable family (sequences, bump profiles, time courses, participation patterns) owns one :class:`Registry`.
New members register themselves with ``@REGISTRY.register("name")``; nothing that consumes a family has to change
(open/closed).  Nested components are coerced by the owning dataclass (``REGISTRY.coerce`` in ``__post_init__``), so a
spec can mix instances and dicts at any depth.
"""
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")

__all__ = ["Registry"]


class Registry(Generic[T]):
    """Factories of one component family, keyed by ``kind``."""

    def __init__(self, family: str, base: type):
        self.family = family
        self.base = base
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, kind: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            if kind in self._factories:
                raise ValueError(f"{self.family} kind {kind!r} registered twice")
            self._factories[kind] = factory
            return factory

        return deco

    def kinds(self) -> list[str]:
        return sorted(self._factories)

    def build(self, spec: dict[str, Any]) -> T:
        """Build from ``{"kind": ..., **params}``."""
        if not isinstance(spec, dict) or "kind" not in spec:
            raise TypeError(f"{self.family} spec must be a dict with a 'kind' key, got {spec!r}")
        params = {k: v for k, v in spec.items() if k != "kind"}
        try:
            factory = self._factories[spec["kind"]]
        except KeyError:
            raise ValueError(f"unknown {self.family} kind {spec['kind']!r}; known: {self.kinds()}") from None
        return factory(**params)

    def coerce(self, value: Any) -> T:
        """Instances of the family pass through; dict specs are built."""
        if isinstance(value, self.base):
            return value
        return self.build(value)
