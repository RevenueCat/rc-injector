from typing import TypeVar
from unittest.mock import Mock

from .injector import Configuration, InjectorConfigurationError, TypeResolver

T = TypeVar("T")


class ErrorOnNotExplicitConfiguration(Configuration):
    def _get_default_resolver(self, cls: type[T]) -> TypeResolver[T]:
        raise InjectorConfigurationError(f"{cls} was not bound explicitly")


class MockOnNotExplicitConfiguration(Configuration):
    def _get_default_resolver(self, cls: type[T]) -> TypeResolver[T]:
        print(f"generating default for {cls}")
        resolver = TypeResolver[T](cls)
        mock = Mock(cls) if isinstance(cls, type) else Mock()
        resolver.to_instance(mock)
        return resolver
