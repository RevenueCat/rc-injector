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


# Tests for resolve() method
def test_resolve_function_with_single_injectable_param() -> None:
    class Screen:
        def say(self, text: str) -> str:
            return f"Screen: {text}"

    def salute(medium: Screen, content: str) -> str:
        return medium.say(content)

    configuration = Configuration()
    screen_instance = Screen()
    configuration.bind(Screen).globally().to_instance(screen_instance)

    injector = Injector(configuration)
    new_salute = injector.resolve(salute)

    # Test that medium is injected, only content is required
    result = new_salute("hello")
    assert result == "Screen: hello"

    # Test with keyword argument
    result = new_salute(content="world")
    assert result == "Screen: world"


def test_resolve_function_with_multiple_params() -> None:
    class Logger:
        def log(self, msg: str) -> str:
            return f"LOG: {msg}"

    class Database:
        def save(self, data: str) -> str:
            return f"SAVED: {data}"

    def process(user_input: str, logger: Logger, db: Database, debug: bool) -> str:
        logger.log(user_input)
        db.save(user_input)
        return f"Processed: {user_input}, debug={debug}"

    configuration = Configuration()
    configuration.bind(Logger).globally().to_instance(Logger())
    configuration.bind(Database).globally().to_instance(Database())

    injector = Injector(configuration)
    wrapped_process = injector.resolve(process)

    # Only user_input and debug are required (logger and db are injected)
    result = wrapped_process("data", True)
    assert result == "Processed: data, debug=True"

    # Can use keyword arguments
    result = wrapped_process(user_input="test", debug=False)
    assert result == "Processed: test, debug=False"


def test_resolve_function_with_all_injectable_params() -> None:
    class A:
        pass

    class B:
        pass

    def func(a: A, b: B) -> str:
        return f"{type(a).__name__}, {type(b).__name__}"

    configuration = Configuration()
    configuration.bind(A).globally()
    configuration.bind(B).globally()

    injector = Injector(configuration)
    wrapped = injector.resolve(func)

    # No parameters required
    result = wrapped()
    assert result == "A, B"


def test_resolve_function_with_no_injectable_params() -> None:
    def add(x: int, y: int) -> int:
        return x + y

    configuration = Configuration()
    injector = Injector(configuration)
    wrapped = injector.resolve(add)

    # All parameters still required (primitives can't be injected)
    result = wrapped(5, 10)
    assert result == 15


def test_resolve_function_with_nested_dependencies() -> None:
    class Config:
        def get_value(self) -> str:
            return "config_value"

    class Service:
        def __init__(self, config: Config) -> None:
            self.config = config

    def handler(service: Service, user_id: str) -> str:
        return f"{service.config.get_value()}-{user_id}"

    configuration = Configuration()
    configuration.bind(Config).globally()
    configuration.bind(Service).globally()

    injector = Injector(configuration)
    wrapped = injector.resolve(handler)

    result = wrapped("user123")
    assert result == "config_value-user123"


def test_resolve_with_bound_instance() -> None:
    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def increment(self) -> int:
            self.count += 1
            return self.count

    def use_counter(counter: Counter) -> int:
        return counter.increment()

    configuration = Configuration()
    counter_instance = Counter()
    configuration.bind(Counter).globally().to_instance(counter_instance)

    injector = Injector(configuration)
    wrapped = injector.resolve(use_counter)

    # All calls should use the same instance
    assert wrapped() == 1
    assert wrapped() == 2
    assert wrapped() == 3


def test_resolve_with_constructor_binding() -> None:
    class Greeter:
        def __init__(self, prefix: str = "Hello") -> None:
            self.prefix = prefix

        def greet(self, name: str) -> str:
            return f"{self.prefix}, {name}"

    def use_greeter(greeter: Greeter, name: str) -> str:
        return greeter.greet(name)

    def custom_greeter_builder() -> Greeter:
        return Greeter("Greetings")

    configuration = Configuration()
    configuration.bind(Greeter).globally().to_constructor(custom_greeter_builder)

    injector = Injector(configuration)
    wrapped = injector.resolve(use_greeter)

    result = wrapped("Alice")
    assert result == "Greetings, Alice"


def test_resolve_shared_dependencies() -> None:
    class SharedResource:
        def __init__(self) -> None:
            self.id = id(self)

    def func1(resource: SharedResource) -> int:
        return resource.id

    def func2(resource: SharedResource) -> int:
        return resource.id

    configuration = Configuration()
    configuration.bind(SharedResource).globally()

    injector = Injector(configuration)
    wrapped1 = injector.resolve(func1)
    wrapped2 = injector.resolve(func2)

    # Both functions should get the same singleton instance
    assert wrapped1() == wrapped2()


def test_resolve_with_positional_args() -> None:
    class Service:
        pass

    def handler(service: Service, arg1: str, arg2: int) -> str:
        return f"{arg1}-{arg2}"

    configuration = Configuration()
    configuration.bind(Service).globally()

    injector = Injector(configuration)
    wrapped = injector.resolve(handler)

    result = wrapped("test", 42)
    assert result == "test-42"


def test_resolve_with_keyword_args() -> None:
    class Service:
        pass

    def handler(service: Service, arg1: str, arg2: int) -> str:
        return f"{arg1}-{arg2}"

    configuration = Configuration()
    configuration.bind(Service).globally()

    injector = Injector(configuration)
    wrapped = injector.resolve(handler)

    result = wrapped(arg1="test", arg2=42)
    assert result == "test-42"


def test_resolve_with_mixed_args() -> None:
    class Service:
        pass

    def handler(service: Service, arg1: str, arg2: int, arg3: bool) -> str:
        return f"{arg1}-{arg2}-{arg3}"

    configuration = Configuration()
    configuration.bind(Service).globally()

    injector = Injector(configuration)
    wrapped = injector.resolve(handler)

    # Mix positional and keyword
    result = wrapped("test", 42, arg3=True)
    assert result == "test-42-True"


def test_resolve_with_default_values() -> None:
    class Service:
        pass

    def handler(service: Service, arg1: str = "default") -> str:
        return arg1

    configuration = Configuration()
    configuration.bind(Service).globally()

    injector = Injector(configuration)
    wrapped = injector.resolve(handler)

    # Can omit arg with default
    result = wrapped()
    assert result == "default"

    # Can override default
    result = wrapped("custom")
    assert result == "custom"


def test_resolve_primitive_param_without_binding() -> None:
    class Service:
        pass

    def handler(service: Service, value: str) -> str:
        return value

    configuration = Configuration()
    configuration.bind(Service).globally()

    injector = Injector(configuration)
    wrapped = injector.resolve(handler)

    # Primitive parameter must be provided
    result = wrapped("test")
    assert result == "test"

    # Missing primitive parameter should raise TypeError
    with pytest.raises(TypeError):
        wrapped()


def test_resolve_untyped_param() -> None:
    class Service:
        pass

    def handler(service: Service, untyped) -> str:  # type: ignore
        return str(untyped)

    configuration = Configuration()
    configuration.bind(Service).globally()

    injector = Injector(configuration)
    wrapped = injector.resolve(handler)

    # Untyped parameter must be provided
    result = wrapped("value")
    assert result == "value"


def test_resolve_abstract_without_binding() -> None:
    class AbstractService(ABC):
        @abstractmethod
        def do_something(self) -> str: ...

    def handler(service: AbstractService, user_input: str) -> str:
        return service.do_something()

    configuration = Configuration()
    injector = Injector(configuration)

    # Abstract without binding should remain as required parameter
    wrapped = injector.resolve(handler)
    # Since AbstractService can't be injected, it should be a required param
    # This should raise TypeError for missing required argument
    with pytest.raises(TypeError):
        wrapped("test")


def test_resolve_function_signature_preserved() -> None:
    import inspect

    class Service:
        pass

    def handler(service: Service, arg1: str, arg2: int = 42) -> str:
        return f"{arg1}-{arg2}"

    configuration = Configuration()
    configuration.bind(Service).globally()

    injector = Injector(configuration)
    wrapped = injector.resolve(handler)

    # Check signature only has non-injectable params
    sig = inspect.signature(wrapped)
    param_names = list(sig.parameters.keys())
    assert param_names == ["arg1", "arg2"]
    assert sig.parameters["arg2"].default == 42
    assert sig.return_annotation is str


def test_resolve_function_metadata_preserved() -> None:
    class Service:
        pass

    def my_handler(service: Service, value: str) -> str:
        """This is a handler function."""
        return value

    configuration = Configuration()
    configuration.bind(Service).globally()

    injector = Injector(configuration)
    wrapped = injector.resolve(my_handler)

    # Check metadata is preserved
    assert wrapped.__name__ == "my_handler"
    assert wrapped.__doc__ == "This is a handler function."


def test_resolve_concurrent_calls() -> None:
    class Resource:
        def __init__(self) -> None:
            time.sleep(0.01)
            self.value = "resource"

    def handler(resource: Resource, arg: str) -> str:
        return f"{resource.value}-{arg}"

    configuration = Configuration()
    configuration.bind(Resource).globally()

    injector = Injector(configuration)
    wrapped = injector.resolve(handler)

    def task(i: int) -> str:
        return wrapped(f"arg{i}")  # type: ignore

    threads = 10
    with ThreadPoolExecutor() as executor:
        running_tasks = [executor.submit(task, i) for i in range(threads)]
        results = [t.result() for t in running_tasks]

    # All should succeed
    assert len(results) == threads
    for i, result in enumerate(results):
        assert result == f"resource-arg{i}"


def test_resolve_circular_dependency_in_function() -> None:
    # Use module-level classes for forward references to work
    def handler(a: CircularDep_A) -> str:
        return "test"

    configuration = Configuration()
    injector = Injector(configuration)
    wrapped = injector.resolve(handler)

    # Should detect circular dependency (wrapped in InjectorInstantiationError)
    with pytest.raises(InjectorInstantiationError) as exc_info:
        wrapped()
    # Verify the underlying cause is CircularDependencyError
    assert isinstance(exc_info.value.__cause__, CircularDependencyError)
