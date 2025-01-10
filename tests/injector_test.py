#!/usr/bin/env python3
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, NewType, Optional, Protocol, Type, TypeVar, Union, cast
from unittest.mock import Mock

import pytest

from rc_injector import (
    CircularDependencyError,
    Configuration,
    Injector,
    InjectorConfigurationError,
    InjectorInstantiationError,
)
from rc_injector.test_utils import (
    ErrorOnNotExplicitConfiguration,
    MockOnNotExplicitConfiguration,
)

T = TypeVar("T")


def test_no_bindinds_simple() -> None:
    class A:
        pass

    configuration = Configuration()
    injector = Injector(configuration)
    # Builds the right class
    assert isinstance(injector.get(A), A)
    # Resolution is cached, returns singleton
    assert id(injector.get(A)) == id(injector.get(A))


def test_no_bindings_chained() -> None:
    class A:
        pass

    class B:
        pass

    class C:
        def __init__(self, a: A, b: B) -> None:
            self.a = a
            self.b = b

    class D:
        def __init__(self, c: C, a: A) -> None:
            self.c = c
            self.a = a

    configuration = Configuration()
    injector = Injector(configuration)
    # Builds the right class
    assert isinstance(injector.get(D), D)
    # Resolution is cached, returns singleton
    assert id(injector.get(D)) == id(injector.get(D))
    assert id(injector.get(D).a) == id(injector.get(D).c.a)


def test_default_values() -> None:
    class A:
        pass

    default_a = A()

    class B:
        def __init__(self, a: A = default_a) -> None:
            self.a = a

    # unless specifically binded, it will use default
    configuration = Configuration()
    injector = Injector(configuration)
    # Builds the right class
    injected_a = injector.get(A)
    assert id(injector.get(B).a) != id(injected_a)
    assert id(injector.get(B).a) == id(default_a)

    # If binded, it is overriden
    configuration = Configuration()
    configuration.bind(A).globally()
    injector = Injector(configuration)
    # Builds the right class
    injected_a = injector.get(A)
    assert id(injector.get(B).a) == id(injected_a)
    assert id(injector.get(B).a) != id(default_a)


def test_bind_primitive_type_fails() -> None:
    class A:
        def __init__(self, a: str) -> None:
            self.a = a

    configuration = Configuration()
    with pytest.raises(InjectorConfigurationError):
        configuration.bind(str).globally().to_instance("foo")
    configuration.bind(A).globally().with_kwargs(a="foo")
    injector = Injector(configuration)
    assert injector.get(A).a == "foo"


def test_bind_to_instance() -> None:
    class A:
        pass

    class B:
        def __init__(self, a: A) -> None:
            self.a = a

    a_instance = A()

    configuration = Configuration()
    configuration.bind(A).globally().to_instance(a_instance)
    injector = Injector(configuration)
    assert id(injector.get(B).a) == id(a_instance)


def test_bind_abstract_to_class() -> None:
    class A(ABC):
        @abstractmethod
        def greet(self) -> str: ...

    class A1(A):
        def greet(self) -> str:
            return "A1"

    class A2(A):
        def greet(self) -> str:
            return "A2"

    class B:
        def __init__(self, a: A) -> None:
            self.a = a

    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(B)

    configuration = Configuration()
    configuration.bind(A).globally().to_class(A1)
    configuration.bind(A).for_parent(B).to_class(A2)
    injector = Injector(configuration)
    assert injector.get(A).greet() == "A1"
    assert injector.get(B).a.greet() == "A2"

    configuration = Configuration()
    configuration.bind(B).globally().with_arg_types(a=A1)
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(A)
    assert injector.get(B).a.greet() == "A1"

    configuration = Configuration()
    configuration.bind(B).globally().with_kwargs(a=A1())
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(A)
    assert injector.get(B).a.greet() == "A1"


def test_bind_protocol_to_class() -> None:
    class A(Protocol):
        def greet(self) -> str: ...

    class A1:
        def greet(self) -> str:
            return "A1"

    class A2:
        def greet(self) -> str:
            return "A2"

    class B:
        def __init__(self, a: A) -> None:
            self.a = a

    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(B)

    configuration = Configuration()
    configuration.bind(A).globally().to_class(A1)
    configuration.bind(A).for_parent(B).to_class(A2)
    injector = Injector(configuration)
    assert injector.get(A).greet() == "A1"
    assert injector.get(B).a.greet() == "A2"

    configuration = Configuration()
    configuration.bind(B).globally().with_arg_types(a=A1)
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(A)
    assert injector.get(B).a.greet() == "A1"

    configuration = Configuration()
    configuration.bind(B).globally().with_kwargs(a=A1())
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(A)
    assert injector.get(B).a.greet() == "A1"


def test_bind_to_constructor() -> None:
    class A:
        def __init__(self, foo: Optional[str] = None) -> None:
            self.foo = foo

    class B:
        def __init__(self, a: A) -> None:
            self.a = a

    def build_A() -> A:
        return A("custom")

    configuration = Configuration()
    configuration.bind(A).globally().to_constructor(build_A)
    injector = Injector(configuration)
    assert injector.get(B).a.foo == "custom"


def test_bind_chained_to_constructor() -> None:
    class A:
        def __init__(self, foo: Optional[str] = None) -> None:
            self.foo = foo

    class B:
        def __init__(self, a: A) -> None:
            self.a = a

    class C:
        def __init__(self, b: B) -> None:
            self.b = b

    def build_A() -> A:
        return A("custom")

    configuration = Configuration()
    configuration.bind(A).globally().to_constructor(build_A)
    injector = Injector(configuration)
    c = injector.get(C)
    assert c.b.a.foo == "custom"
    c_cached = injector.get(C)
    assert id(c) == id(c_cached)


def test_bind_to_constructor_with_dependencies_too() -> None:
    class OtherDep:
        def __init__(self) -> None:
            self.foo = "overriden"

    class A:
        def __init__(self, foo: Optional[str] = None) -> None:
            self.foo = foo

    class B:
        def __init__(self, a: A) -> None:
            self.a = a

    # Constructors can have dependencies injected too
    def build_A(other_dep: OtherDep) -> A:
        return A(other_dep.foo)

    configuration = Configuration()
    configuration.bind(A).globally().to_constructor(build_A)
    injector = Injector(configuration)
    assert injector.get(B).a.foo == "overriden"


# We can only resolve text-based annotations for
# classes defined in global scope
class CircularDep_A:
    def __init__(self, b: "CircularDep_B") -> None:
        self.b = b


class CircularDep_B:
    def __init__(self, c: "CircularDep_C") -> None:
        self.c = c


class CircularDep_C:
    def __init__(self, a: CircularDep_A) -> None:
        self.a = a


def test_circular_dependency() -> None:
    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(CircularDependencyError):
        injector.get(CircularDep_A)


def test_primitive_param() -> None:
    class A:
        def __init__(self, foo: str) -> None:
            self.foo = foo

    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(A)

    configuration = Configuration()
    configuration.bind(A).globally().with_kwargs(foo="bar")
    injector = Injector(configuration)
    assert isinstance(injector.get(A), A)
    assert injector.get(A).foo == "bar"


def test_primitive_param_with_default_value() -> None:
    class A:
        def __init__(self, foo: str = "default") -> None:
            self.foo = foo

    configuration = Configuration()
    injector = Injector(configuration)
    assert isinstance(injector.get(A), A)
    assert injector.get(A).foo == "default"

    configuration = Configuration()
    configuration.bind(A).globally().with_kwargs(foo="bar")
    injector = Injector(configuration)
    assert isinstance(injector.get(A), A)
    assert injector.get(A).foo == "bar"


def test_global_and_parent_binding() -> None:
    class A:
        def __init__(self, foo: str) -> None:
            self.foo = foo

    class B:
        def __init__(self, a: A) -> None:
            self.a = a

    class C:
        def __init__(self, a: A) -> None:
            self.a = a

    configuration = Configuration()
    configuration.bind(A).globally().with_kwargs(foo="global")
    injector = Injector(configuration)
    assert injector.get(B).a.foo == "global"
    assert injector.get(C).a.foo == "global"

    configuration = Configuration()
    configuration.bind(A).globally().with_kwargs(foo="global")
    configuration.bind(A).for_parent(B).with_kwargs(foo="for_B")
    configuration.bind(A).for_parent(C).with_kwargs(foo="for_C")
    injector = Injector(configuration)
    assert injector.get(A).foo == "global"
    assert injector.get(B).a.foo == "for_B"
    assert injector.get(C).a.foo == "for_C"


def test_optional_and_union_types() -> None:
    class A:
        pass

    class B:
        def __init__(self, a: Optional[A]) -> None:
            self.a = a

    class C:
        def __init__(self, a_or_b: Union[A, B]) -> None:
            self.a_or_b = a_or_b

    # If not binded, it will fail
    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(B)
    with pytest.raises(InjectorConfigurationError):
        injector.get(C)

    # Just binding the classes, will fail
    configuration = Configuration()
    configuration.bind(A).globally()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(B)
    with pytest.raises(InjectorConfigurationError):
        injector.get(C)

    # Explicitly binding the Optional and Unions works
    configuration = Configuration()
    # To bind Optional and Union you will need to cast
    # the complex type to make strict type-check happy
    configuration.bind(cast(Type[A], Optional[A])).globally().to_class(A)
    configuration.bind(cast(Type[A], Union[A, B])).globally().to_class(A)
    injector = Injector(configuration)
    assert isinstance(injector.get(B).a, A)
    assert isinstance(injector.get(C).a_or_b, A)

    configuration = Configuration()
    configuration.bind(B).globally().with_arg_types(a=A)
    configuration.bind(C).globally().with_arg_types(a_or_b=B)
    injector = Injector(configuration)
    assert isinstance(injector.get(B).a, A)
    assert isinstance(injector.get(C).a_or_b, B)

    configuration = Configuration()
    configuration.bind(B).globally().with_kwargs(a=None)
    configuration.bind(C).globally().with_kwargs(a_or_b=B(A()))
    injector = Injector(configuration)
    assert injector.get(B).a is None
    assert isinstance(injector.get(C).a_or_b, B)


def test_bind_new_type_and_type_alias() -> None:
    class A:
        pass

    class B:
        pass

    AOrB = NewType("AOrB", Union[A, B])  # type: ignore
    BOrA = Union[B, A]

    class UsesNewType:
        def __init__(self, a_or_b: AOrB) -> None:
            self.a_or_b = a_or_b

    class UsesAlias:
        def __init__(self, b_or_a: BOrA) -> None:
            self.b_or_a = b_or_a

    configuration = Configuration()
    configuration.bind(A).globally()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(UsesNewType)
    with pytest.raises(InjectorConfigurationError):
        injector.get(UsesAlias)

    configuration = Configuration()
    configuration.bind(AOrB).globally().to_class(A)
    configuration.bind(cast(Type[B], BOrA)).globally().to_class(B)
    injector = Injector(configuration)
    assert isinstance(injector.get(UsesNewType).a_or_b, A)
    assert isinstance(injector.get(UsesAlias).b_or_a, B)


def test_generics() -> None:
    class Container(Generic[T]):
        pass

    class NeedsAnyContainer:
        def __init__(
            self,
            a: Container,  # type: ignore
        ) -> None:
            self.a = a

    class NeedsIntContainer:
        def __init__(self, a: Container[int]) -> None:
            self.a = a

    class AlsoNeedsIntContainer:
        def __init__(self, a: Container[int]) -> None:
            self.a = a

    configuration = Configuration()
    injector = Injector(configuration)
    # Check containers are injected, but they are different
    # instances if the type parameter is not the same
    assert isinstance(injector.get(NeedsAnyContainer).a, Container)
    assert isinstance(injector.get(NeedsIntContainer).a, Container)
    assert isinstance(injector.get(AlsoNeedsIntContainer).a, Container)
    assert id(injector.get(NeedsAnyContainer).a) != id(
        injector.get(NeedsIntContainer).a
    )
    assert id(injector.get(NeedsIntContainer).a) == id(
        injector.get(AlsoNeedsIntContainer).a
    )


def test_generics_instances() -> None:
    class Container(Generic[T]):
        def __init__(self, value: T) -> None:
            self.value = value

    class NeedsAnyContainer:
        def __init__(
            self,
            a: Container,  # type: ignore
        ) -> None:
            self.a = a

    class NeedsIntContainer:
        def __init__(self, a: Container[int]) -> None:
            self.a = a

    class AlsoNeedsIntContainer:
        def __init__(self, a: Container[int]) -> None:
            self.a = a

    str_container = Container("foo")
    int_container = Container(1)

    configuration = Configuration()
    # We are binding to generic container, so strict type-check
    # will complaint about the instance provided being incompatible
    # Container[str]. Casting it will make type-check happy.
    configuration.bind(cast(Type[Container[str]], Container)).globally().to_instance(
        str_container
    )
    configuration.bind(Container[int]).for_parent(NeedsIntContainer).to_instance(
        int_container
    )
    injector = Injector(configuration)
    # The generic container was injected str_container:
    assert injector.get(NeedsAnyContainer).a.value == "foo"
    # The bind on Container[int] for parent NeedsIntContainer
    # was properly fed the int_container
    assert injector.get(NeedsIntContainer).a.value == 1
    # Since Container requires and argument and there
    # is no global binding for Container[int], this
    # will fail
    with pytest.raises(InjectorInstantiationError):
        injector.get(AlsoNeedsIntContainer)


def test_error_on_not_explicit_bind_configuration() -> None:
    class Dependency:
        pass

    class ClassToTest:
        def __init__(self, dep: Dependency):
            self.dep = dep

    configuration = ErrorOnNotExplicitConfiguration()
    configuration.bind(ClassToTest).globally()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(ClassToTest)

    configuration = ErrorOnNotExplicitConfiguration()
    configuration.bind(ClassToTest).globally()
    configuration.bind(Dependency).globally().to_instance(Mock())
    injector = Injector(configuration)
    assert isinstance(injector.get(ClassToTest).dep, Mock)


def test_mock_on_not_explitict_bind_configuration() -> None:
    class Dependency:
        def some_method(self) -> str:
            return "PRODUCTION_VALUE"

    class ClassToTest:
        def __init__(self, dep: Dependency):
            self.dep = dep

    configuration = MockOnNotExplicitConfiguration()
    configuration.bind(ClassToTest).globally()
    injector = Injector(configuration)
    assert isinstance(injector.get(ClassToTest).dep, Mock)

    # We can access the mock to configure it just
    # asking the injector for the dependency
    injector.get(Dependency).some_method.return_value = "TEST_VALUE"  # type: ignore
    assert injector.get(ClassToTest).dep.some_method() == "TEST_VALUE"

    with pytest.raises(AttributeError):
        # Mock is checking the type and funcion signatures
        injector.get(Dependency).method_does_not_exist()  # type: ignore


def test_concurrent_injection() -> None:
    class A:
        def __init__(self) -> None:
            time.sleep(0.01)

    configuration = Configuration()
    configuration.bind(A).globally()
    injector = Injector(configuration)

    def task() -> bool:
        return isinstance(injector.get(A), A)

    threads = 20
    with ThreadPoolExecutor() as executor:
        running_tasks = [executor.submit(task) for _ in range(threads)]
        ok = 0
        for running_task in running_tasks:
            if running_task.result():
                ok += 1
    assert ok == threads
