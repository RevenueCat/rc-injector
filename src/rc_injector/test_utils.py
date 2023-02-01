from typing import Type, TypeVar
from unittest.mock import Mock

from .injector import Configuration, InjectorConfigurationError, TypeResolver

T = TypeVar("T")


class ErrorOnNotExplicitConfiguration(Configuration):
    def _get_default_resolver(self, cls: Type[T]) -> TypeResolver[T]:
        raise InjectorConfigurationError(f"{cls} was not binded explicity")


class MockOnNotExplicitConfiguration(Configuration):
    def _get_default_resolver(self, cls: Type[T]) -> TypeResolver[T]:
        print(f"generating default for {cls}")
        resolver = TypeResolver[T](cls)
        if isinstance(cls, type):
            mock = Mock(cls)
        else:
            mock = Mock()
        resolver.to_instance(mock)
        return resolver
