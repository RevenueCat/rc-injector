import collections
import threading
import time
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    NewType,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
)
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

    # unless specifically bound, it will use default
    configuration = Configuration()
    injector = Injector(configuration)
    # Builds the right class
    injected_a = injector.get(A)
    assert id(injector.get(B).a) != id(injected_a)
    assert id(injector.get(B).a) == id(default_a)

    # If bound, it is overridden
    configuration = Configuration()
    configuration.bind(A).globally()
    injector = Injector(configuration)
    # Builds the right class
    injected_a = injector.get(A)
    assert id(injector.get(B).a) == id(injected_a)
    assert id(injector.get(B).a) != id(default_a)

    class C:
        def __init__(self, a: A) -> None:
            self.a = a

    # A binding scoped to another parent is not a binding for this
    # one, so the default is kept
    configuration = Configuration()
    configuration.bind(A).for_parent(C)
    injector = Injector(configuration)
    assert id(injector.get(B).a) == id(default_a)
    assert id(injector.get(C).a) != id(default_a)


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


class AliasCycle_A:
    # Needs itself through a transparent alias
    def __init__(self, a: "Annotated[AliasCycle_A, 'meta']") -> None:
        self.a = a


def test_circular_dependency() -> None:
    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(CircularDependencyError):
        injector.get(CircularDep_A)

    # Needing itself through an alias is a cycle all the same, as the
    # alias is resolved as the class it refers to
    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(CircularDependencyError):
        injector.get(AliasCycle_A)

    # An already resolved instance needs nothing built, so a binding
    # to one breaks the cycle, even for the very class being built at
    # the root, that keeps its default instantiation
    configuration = Configuration()
    a_for_c = Mock()
    configuration.bind(CircularDep_A).for_parent(CircularDep_C).to_instance(a_for_c)
    injector = Injector(configuration)
    assert injector.get(CircularDep_A).b.c.a is a_for_c


def test_constructor_raising() -> None:
    class A:
        def __init__(self) -> None:
            raise ValueError("boom")

    configuration = Configuration()
    injector = Injector(configuration)
    # The error from the constructor is wrapped, and kept as the cause
    with pytest.raises(InjectorInstantiationError, match="Constructor raised") as e:
        injector.get(A)
    assert isinstance(e.value.__cause__, ValueError)


def test_primitive_param() -> None:
    class A:
        def __init__(self, foo: str) -> None:
            self.foo = foo

    configuration = Configuration()
    injector = Injector(configuration)
    # The error has to name the param that can't be injected
    with pytest.raises(InjectorConfigurationError, match="param `foo` is a primitive"):
        injector.get(A)

    configuration = Configuration()
    configuration.bind(A).globally().with_kwargs(foo="bar")
    injector = Injector(configuration)
    assert isinstance(injector.get(A), A)
    assert injector.get(A).foo == "bar"


def test_value_types_cannot_be_built() -> None:
    class Needs:
        def __init__(self, retries: Annotated[int, "meta"] = 5) -> None:
            self.retries = retries

    annotated_int = cast(type[int], Annotated[int, "meta"])

    # A primitive or a container has nothing to build, so asking for one
    # is an error rather than a default-constructed `0`, `''` or `[]`.
    # Directly, through an alias, or as a parameterized container
    configuration = Configuration()
    injector = Injector(configuration)
    for value_type in (int, str, list, annotated_int, cast(type[Any], list[int])):
        with pytest.raises(InjectorConfigurationError, match="primitive or container"):
            injector.get(value_type)

    # A bare binding of one is the same mistake, and it would otherwise
    # even override the default in a signature
    configuration = Configuration()
    configuration.bind(annotated_int).globally()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError, match="primitive or container"):
        injector.get(annotated_int)
    with pytest.raises(InjectorConfigurationError, match="primitive or container"):
        injector.get(Needs)

    # The value has to be provided: with no binding the default is used,
    # and a binding to an instance is what overrides it
    configuration = Configuration()
    injector = Injector(configuration)
    assert injector.get(Needs).retries == 5
    configuration = Configuration()
    configuration.bind(annotated_int).globally().to_instance(7)
    injector = Injector(configuration)
    assert injector.get(Needs).retries == 7
    assert injector.get(annotated_int) == 7


def test_untyped_param() -> None:
    class A:
        def __init__(self, foo) -> None:  # type: ignore[no-untyped-def]
            self.foo = foo

    configuration = Configuration()
    injector = Injector(configuration)
    # The error has to name the param, not dump the whole signature
    with pytest.raises(InjectorConfigurationError, match="param `foo` is not typed"):
        injector.get(A)


def test_parameterized_primitive_params() -> None:
    class A:
        pass

    class NeedsList:
        def __init__(self, x: list[str]) -> None:
            self.x = x

    class NeedsTypingList:
        # The pre-PEP 585 spelling has to be recognized too
        def __init__(self, x: typing.List[str]) -> None:  # noqa: UP006
            self.x = x

    class NeedsDict:
        def __init__(self, x: dict[str, A]) -> None:
            self.x = x

    class NeedsBool:
        def __init__(self, flag: bool) -> None:
            self.flag = flag

    class NeedsLiteral:
        def __init__(self, x: Literal["a", "b"]) -> None:
            self.x = x

    # A parameterized primitive is a value as much as a plain one, so it
    # can't be injected: it used to be built as an empty `list()`/`dict()`
    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError, match="param `x` is a primitive"):
        injector.get(NeedsList)
    with pytest.raises(InjectorConfigurationError, match="param `x` is a primitive"):
        injector.get(NeedsTypingList)
    with pytest.raises(InjectorConfigurationError, match="param `x` is a primitive"):
        injector.get(NeedsDict)
    # `bool` was missing from the primitive types, so it was injected False
    with pytest.raises(InjectorConfigurationError, match="param `flag` is a primitive"):
        injector.get(NeedsBool)
    with pytest.raises(InjectorConfigurationError, match="param `x` is a primitive"):
        injector.get(NeedsLiteral)

    # The value has to be provided, as for any other primitive
    configuration = Configuration()
    configuration.bind(NeedsList).globally().with_kwargs(x=["foo"])
    configuration.bind(NeedsBool).globally().with_kwargs(flag=True)
    injector = Injector(configuration)
    assert injector.get(NeedsList).x == ["foo"]
    assert injector.get(NeedsBool).flag is True

    # Unlike plain primitives, the parameterized ones can be bound, as
    # they are specific enough to identify what has to be injected
    configuration = Configuration()
    configuration.bind(cast(type[Any], list[str])).globally().to_instance(["bar"])
    injector = Injector(configuration)
    assert injector.get(NeedsList).x == ["bar"]


def test_container_params_are_not_injected() -> None:
    class Foo:
        pass

    class NeedsListOfFoo:
        def __init__(self, foos: list[Foo]) -> None:
            self.foos = foos

    class NeedsDeque:
        def __init__(self, x: collections.deque[Foo]) -> None:
            self.x = x

    class NeedsCounter:
        def __init__(self, x: collections.Counter[str]) -> None:
            self.x = x

    class NeedsSequence:
        def __init__(self, x: typing.Sequence[Foo]) -> None:
            self.x = x

    class NeedsIterable:
        def __init__(self, x: typing.Iterable[Foo]) -> None:
            self.x = x

    class NeedsCallable:
        def __init__(self, x: Callable[[int], str]) -> None:
            self.x = x

    configuration = Configuration()
    injector = Injector(configuration)
    # `Foo` on its own is injectable, but a container of it is a value:
    # the injector will not build `[injected_foo]` out of thin air
    assert isinstance(injector.get(Foo), Foo)
    with pytest.raises(InjectorConfigurationError, match="`foos` is a primitive or"):
        injector.get(NeedsListOfFoo)
    # The rest of the `collections` family too. These can be built with
    # no arguments, so they used to be injected silently empty
    with pytest.raises(InjectorConfigurationError, match="`x` is a primitive or"):
        injector.get(NeedsDeque)
    with pytest.raises(InjectorConfigurationError, match="`x` is a primitive or"):
        injector.get(NeedsCounter)
    # And its abstract interfaces, which have nothing to build either
    with pytest.raises(InjectorConfigurationError, match="`x` is a primitive or"):
        injector.get(NeedsSequence)
    with pytest.raises(InjectorConfigurationError, match="`x` is a primitive or"):
        injector.get(NeedsIterable)
    # Including the ones that are not containers at all
    with pytest.raises(InjectorConfigurationError, match="`x` is a primitive or"):
        injector.get(NeedsCallable)


def test_container_param_with_default_value() -> None:
    class Foo:
        pass

    def default_handler(value: int) -> str:
        return str(value)

    class A:
        def __init__(
            self,
            foos: tuple[Foo, ...] = (),
            handler: Callable[[int], str] = default_handler,
        ) -> None:
            self.foos = foos
            self.handler = handler

    # A default in the signature is used, as for any other param, so
    # only a mandatory param of a value type fails to resolve
    configuration = Configuration()
    injector = Injector(configuration)
    assert injector.get(A).foos == ()
    assert injector.get(A).handler(1) == "1"

    configuration = Configuration()
    configuration.bind(A).globally().with_kwargs(foos=(Foo(),))
    injector = Injector(configuration)
    assert len(injector.get(A).foos) == 1


def test_own_container_class_is_still_injected() -> None:
    class Foo:
        pass

    # Only the stdlib containers are values. A container of your own is
    # a dependency like any other class, even if it is iterable
    class Registry:
        def __init__(self) -> None:
            self.items: list[Foo] = []

        def __iter__(self) -> typing.Iterator[Foo]:
            return iter(self.items)

        def __len__(self) -> int:
            return len(self.items)

    class NeedsRegistry:
        def __init__(self, registry: Registry) -> None:
            self.registry = registry

    configuration = Configuration()
    injector = Injector(configuration)
    assert isinstance(injector.get(NeedsRegistry).registry, Registry)


def test_primitive_param_with_default_value() -> None:
    class A:
        def __init__(self, foo: str = "default") -> None:
            self.foo = foo

    class Falsy:
        # A falsy default is a default all the same
        def __init__(self, flag: bool = False, count: int = 0, name: str = "") -> None:
            self.flag = flag
            self.count = count
            self.name = name

    configuration = Configuration()
    injector = Injector(configuration)
    assert injector.get(Falsy).flag is False
    assert injector.get(Falsy).count == 0
    assert injector.get(Falsy).name == ""

    configuration = Configuration()
    configuration.bind(Falsy).globally().with_kwargs(flag=True)
    injector = Injector(configuration)
    assert injector.get(Falsy).flag is True

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


def test_parent_binding_only() -> None:
    class Foo:
        def __init__(self, tag: str = "default") -> None:
            self.tag = tag

    class Bar:
        def __init__(self, foo: Foo) -> None:
            self.foo = foo

    class Baz:
        def __init__(self, foo: Foo) -> None:
            self.foo = foo

    # A binding scoped to a parent only takes effect for it. Everyone
    # else keeps the default instantiation, with no global binding
    # needed to restore it
    configuration = Configuration()
    configuration.bind(Foo).for_parent(Bar).with_kwargs(tag="for_bar")
    injector = Injector(configuration)
    assert injector.get(Bar).foo.tag == "for_bar"
    assert injector.get(Baz).foo.tag == "default"
    assert injector.get(Foo).tag == "default"
    # The default is the same singleton wherever it is injected
    assert injector.get(Baz).foo is injector.get(Foo)
    assert injector.get(Bar).foo is not injector.get(Foo)

    # It is the default instantiation proper, the one the test
    # configurations hook: for them, a binding scoped to another
    # parent is not an explicit binding
    configuration = ErrorOnNotExplicitConfiguration()
    configuration.bind(Foo).for_parent(Bar).with_kwargs(tag="for_bar")
    configuration.bind(Bar).globally()
    configuration.bind(Baz).globally()
    injector = Injector(configuration)
    assert injector.get(Bar).foo.tag == "for_bar"
    with pytest.raises(InjectorConfigurationError, match="not bound explicitly"):
        injector.get(Baz)


def test_get_with_parent_cls() -> None:
    class A:
        def __init__(self, foo: str = "?") -> None:
            self.foo = foo

    class B:
        pass

    # The `parent_cls` param of `get()` selects the bindings scoped to
    # it, both when the instance has to be built and when it is cached
    configuration = Configuration()
    configuration.bind(A).globally().with_kwargs(foo="global")
    configuration.bind(A).for_parent(B).with_kwargs(foo="for_B")
    injector = Injector(configuration)
    assert injector.get(A, parent_cls=B).foo == "for_B"
    assert injector.get(A).foo == "global"
    assert injector.get(A, parent_cls=B).foo == "for_B"
    assert id(injector.get(A, parent_cls=B)) == id(injector.get(A, parent_cls=B))
    assert id(injector.get(A, parent_cls=B)) != id(injector.get(A))

    class C:
        def __init__(self, a: A) -> None:
            self.a = a

    # The parent only applies to what is asked for, it does not leak
    # into its dependencies: those are scoped to what is being built
    configuration = Configuration()
    configuration.bind(A).globally().with_kwargs(foo="global")
    configuration.bind(A).for_parent(B).with_kwargs(foo="for_B")
    injector = Injector(configuration)
    assert injector.get(C, parent_cls=B).a.foo == "global"

    # The parent is not being built, it only selects the bindings, so
    # asking for a class on behalf of itself is not a cycle: it gets
    # the bindings scoped to itself
    configuration = Configuration()
    configuration.bind(A).globally().with_kwargs(foo="global")
    configuration.bind(A).for_parent(A).with_kwargs(foo="for_A")
    injector = Injector(configuration)
    assert injector.get(A, parent_cls=A).foo == "for_A"


def test_get_with_parent_cls_needing_the_parent() -> None:
    class Bus:
        def __init__(self, injector: Injector) -> None:
            self.injector = injector

        def handler(self) -> "Handler":
            # Resolved on its own behalf, to get the bindings scoped to it
            return self.injector.get(Handler, parent_cls=Bus)

    class Handler:
        def __init__(self, bus: Bus, retries: int = 1) -> None:
            self.bus = bus
            self.retries = retries

    # A class resolving its dependencies on its own behalf is already
    # built when it asks, so a dependency needing it back is not a
    # cycle: it gets the instance the class is bound to...
    configuration = Configuration()
    configuration.bind(Handler).for_parent(Bus).with_kwargs(retries=5)
    injector = Injector(configuration)
    bus = Bus(injector)
    configuration.bind(Bus).globally().to_instance(bus)
    handler = bus.handler()
    assert handler.bus is bus
    assert handler.retries == 5

    # ...or the singleton the injector built for it
    configuration = Configuration()
    configuration.bind(Handler).for_parent(Bus).with_kwargs(retries=5)
    injector = Injector(configuration)
    configuration.bind(Injector).globally().to_instance(injector)
    bus = injector.get(Bus)
    handler = bus.handler()
    assert handler.bus is bus
    assert handler.retries == 5


def test_configuration_is_frozen_once_resolving() -> None:
    class A:
        pass

    class B:
        pass

    # Bindings can be made until the first lookup, the injector being
    # created earlier is fine...
    configuration = Configuration()
    injector = Injector(configuration)
    configuration.bind(A).globally().to_class(B)
    assert isinstance(injector.get(A), B)

    # ...but not afterwards: what a class resolves to is settled on
    # first use, so a later binding would not be seen. It fails loudly
    with pytest.raises(InjectorConfigurationError, match="started resolving"):
        configuration.bind(B)
    with pytest.raises(InjectorConfigurationError, match="started resolving"):
        configuration.bind(A)


def test_falsy_instances() -> None:
    class Flag:
        def __init__(self, name: str = "?", on: bool = True) -> None:
            self.name = name
            self.on = on

        def __bool__(self) -> bool:
            return self.on

    class B:
        pass

    # A falsy instance is cached and returned as any other
    configuration = Configuration()
    configuration.bind(Flag).globally().with_kwargs(on=False)
    injector = Injector(configuration)
    assert injector.get(Flag).on is False
    assert id(injector.get(Flag)) == id(injector.get(Flag))

    # And being falsy does not make the scoped binding fall back to
    # the global one
    configuration = Configuration()
    configuration.bind(Flag).globally().to_instance(Flag("global", on=True))
    configuration.bind(Flag).for_parent(B).to_instance(Flag("scoped", on=False))
    injector = Injector(configuration)
    assert injector.get(Flag, parent_cls=B).name == "scoped"
    assert injector.get(Flag).name == "global"


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
    configuration.bind(cast(type[A], Optional[A])).globally().to_class(A)
    configuration.bind(cast(type[A], Union[A, B])).globally().to_class(A)
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


def test_pep604_optional_and_union_types() -> None:
    class A:
        pass

    class B:
        def __init__(self, a: A | None) -> None:
            self.a = a

    class C:
        def __init__(self, a_or_b: A | B) -> None:
            self.a_or_b = a_or_b

    # If not bound, it will fail
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

    # Binding the union without saying what to build will fail too,
    # as there is nothing to instantiate for a union
    configuration = Configuration()
    configuration.bind(cast(type[A], A | None)).globally()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(B)

    # Explicitly binding the unions works. To bind them you will need
    # to cast the complex type to make strict type-check happy
    configuration = Configuration()
    configuration.bind(cast(type[A], A | None)).globally().to_class(A)
    configuration.bind(cast(type[A], A | B)).globally().to_class(A)
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


def test_pep604_and_typing_unions_are_interchangeable() -> None:
    class A:
        pass

    class UsesPep604:
        def __init__(self, a: A | None) -> None:
            self.a = a

    class UsesOptional:
        def __init__(self, a: Optional[A]) -> None:
            self.a = a

    # `A | None` and `Optional[A]` are the very same type, so a
    # binding made with either syntax resolves both constructors
    configuration = Configuration()
    configuration.bind(cast(type[A], Optional[A])).globally().to_class(A)
    injector = Injector(configuration)
    assert isinstance(injector.get(UsesPep604).a, A)
    assert isinstance(injector.get(UsesOptional).a, A)

    configuration = Configuration()
    configuration.bind(cast(type[A], A | None)).globally().to_class(A)
    injector = Injector(configuration)
    assert isinstance(injector.get(UsesPep604).a, A)
    assert isinstance(injector.get(UsesOptional).a, A)


def test_annotated_params() -> None:
    class A:
        pass

    class NeedsAnnotated:
        def __init__(self, a: Annotated[A, "meta"]) -> None:
            self.a = a

    class NeedsAnnotatedUnion:
        def __init__(self, a: Annotated[Optional[A], "meta"]) -> None:
            self.a = a

    class NeedsAnnotatedStr:
        def __init__(self, s: Annotated[str, "db_url"]) -> None:
            self.s = s

    # `Annotated` only decorates the type, so it is injected as the
    # type it decorates, sharing its instance
    configuration = Configuration()
    injector = Injector(configuration)
    assert isinstance(injector.get(NeedsAnnotated).a, A)
    assert id(injector.get(NeedsAnnotated).a) == id(injector.get(A))

    # And what it decorates is checked as usual
    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(NeedsAnnotatedUnion)
    with pytest.raises(InjectorConfigurationError):
        injector.get(NeedsAnnotatedStr)

    # An explicit binding wins over the decorated type, so `Annotated`
    # can be used to tell apart params that share the same type
    configuration = Configuration()
    configuration.bind(
        cast(type[str], Annotated[str, "db_url"])
    ).globally().to_instance("postgres://")
    injector = Injector(configuration)
    assert injector.get(NeedsAnnotatedStr).s == "postgres://"


def test_annotated_param_uses_the_bindings_of_its_type() -> None:
    class AbstractFoo(ABC):
        @abstractmethod
        def foo(self) -> None: ...

    class ConcreteFoo(AbstractFoo):
        def foo(self) -> None: ...

    class ProtocolFoo(Protocol):
        def foo(self) -> None: ...

    class ImplementsFoo:
        def foo(self) -> None: ...

    class Plain:
        pass

    default_plain = Plain()

    class NeedsAbstract:
        def __init__(self, a: Annotated[AbstractFoo, "meta"]) -> None:
            self.a = a

    class NeedsProtocol:
        def __init__(self, a: Annotated[ProtocolFoo, "meta"]) -> None:
            self.a = a

    class NeedsUnion:
        def __init__(self, a: Annotated[Optional[ConcreteFoo], "meta"]) -> None:
            self.a = a

    class NeedsPlainWithDefault:
        def __init__(self, a: Annotated[Plain, "meta"] = default_plain) -> None:
            self.a = a

    # An `Annotated` param with no binding of its own is the type it
    # refers to, bindings included: an abstract class, a Protocol or
    # a union that is bound is injected as configured...
    configuration = Configuration()
    configuration.bind(AbstractFoo).globally().to_class(ConcreteFoo)
    configuration.bind(ProtocolFoo).globally().to_class(ImplementsFoo)
    configuration.bind(
        cast(type[ConcreteFoo], Optional[ConcreteFoo])
    ).globally().to_class(ConcreteFoo)
    injector = Injector(configuration)
    assert isinstance(injector.get(NeedsAbstract).a, ConcreteFoo)
    assert isinstance(injector.get(NeedsProtocol).a, ImplementsFoo)
    assert isinstance(injector.get(NeedsUnion).a, ConcreteFoo)

    # ...and a binding overrides the default in the signature, as it
    # does for a plain param
    configuration = Configuration()
    injector = Injector(configuration)
    assert injector.get(NeedsPlainWithDefault).a is default_plain
    configuration = Configuration()
    configuration.bind(Plain).globally()
    injector = Injector(configuration)
    assert injector.get(NeedsPlainWithDefault).a is not default_plain


def test_transparent_alias_shares_binding_and_instance() -> None:
    class A:
        pass

    class B(A):
        pass

    class NeedsA:
        def __init__(self, a: A) -> None:
            self.a = a

    annotated_a = cast(type[A], Annotated[A, "meta"])

    # An alias with no binding of its own is what it refers to: asked
    # for directly, it is resolved through the binding of its target...
    configuration = Configuration()
    configuration.bind(A).globally().to_class(B)
    injector = Injector(configuration)
    assert isinstance(injector.get(annotated_a), B)

    # ...and shares its instance, so there is a single singleton
    configuration = Configuration()
    injector = Injector(configuration)
    assert injector.get(annotated_a) is injector.get(A)

    # The same when the alias is the type to inject for a param, or
    # the class to build, by configuration
    configuration = Configuration()
    configuration.bind(A).globally().to_class(B)
    configuration.bind(NeedsA).globally().with_arg_types(a=annotated_a)
    injector = Injector(configuration)
    assert isinstance(injector.get(NeedsA).a, B)

    class Wanted:
        pass

    configuration = Configuration()
    configuration.bind(A).globally().to_class(B)
    configuration.bind(Wanted).globally().to_class(cast(type[Wanted], annotated_a))
    injector = Injector(configuration)
    assert isinstance(injector.get(Wanted), B)

    # And a binding scoped to the parent it is resolved for applies
    configuration = Configuration()
    configuration.bind(A).for_parent(NeedsA).to_class(B)
    injector = Injector(configuration)
    assert isinstance(injector.get(annotated_a, parent_cls=NeedsA), B)
    assert not isinstance(injector.get(annotated_a), B)


def test_class_asked_through_an_alias_is_built_as_itself() -> None:
    class Dep:
        pass

    class DepForReal(Dep):
        pass

    class Real:
        def __init__(self, dep: Dep) -> None:
            self.dep = dep

    annotated_real = cast(type[Real], Annotated[Real, "meta"])

    # Asked for through an alias, a class is still the parent of its
    # own dependencies, so the bindings scoped to it apply...
    configuration = Configuration()
    configuration.bind(Dep).for_parent(Real).to_class(DepForReal)
    injector = Injector(configuration)
    assert isinstance(injector.get(annotated_real).dep, DepForReal)
    # ...and the instance built is the very one the class itself gets
    assert injector.get(Real) is injector.get(annotated_real)
    assert isinstance(injector.get(Real).dep, DepForReal)


@pytest.mark.skipif(
    not hasattr(typing, "TypeAliasType"),
    reason="PEP 695 `type Foo = ...` aliases need Python 3.12+",
)
def test_nested_alias_resolves_layer_by_layer() -> None:
    class A:
        pass

    class B(A):
        pass

    class C(A):
        pass

    # Same as `type Inner = A` and `type Outer = Inner`
    type_alias_type: Any = getattr(typing, "TypeAliasType", None)
    inner = type_alias_type("Inner", A)
    outer = type_alias_type("Outer", inner)
    annotated_inner = cast(type[A], Annotated[inner, "meta"])

    class NeedsOuter:
        def __init__(self, a: outer) -> None:  # type: ignore[valid-type]
            self.a = a

    # A binding on an intermediate alias is where the resolution stops,
    # so the layers above share its binding and its instance, asked for
    # directly or as a dependency, while the innermost type keeps its own
    configuration = Configuration()
    configuration.bind(cast(type[A], inner)).globally().to_class(B)
    injector = Injector(configuration)
    assert isinstance(injector.get(outer), B)
    assert isinstance(injector.get(annotated_inner), B)
    assert isinstance(injector.get(NeedsOuter).a, B)
    assert injector.get(outer) is injector.get(inner)
    assert type(injector.get(A)) is A

    # A binding on an outer layer wins over the inner ones
    configuration = Configuration()
    configuration.bind(cast(type[A], inner)).globally().to_class(B)
    configuration.bind(cast(type[A], outer)).globally().to_class(C)
    injector = Injector(configuration)
    assert isinstance(injector.get(outer), C)
    assert isinstance(injector.get(inner), B)


@pytest.mark.skipif(
    not hasattr(typing, "TypeAliasType"),
    reason="PEP 695 `type Foo = ...` aliases need Python 3.12+",
)
def test_pep695_type_alias() -> None:
    class A:
        pass

    class B:
        pass

    # Same as `type AliasOfA = A` and `type AliasOfUnion = A | None`,
    # built dynamically so this module still parses on Python < 3.12
    type_alias_type: Any = getattr(typing, "TypeAliasType", None)
    alias_of_a = type_alias_type("AliasOfA", A)
    alias_of_union = type_alias_type("AliasOfUnion", Optional[A])

    class NeedsAliasOfA:
        def __init__(self, a: alias_of_a) -> None:  # type: ignore[valid-type]
            self.a = a

    class NeedsAliasOfUnion:
        def __init__(self, a: alias_of_union) -> None:  # type: ignore[valid-type]
            self.a = a

    # A `type` alias is transparent, so it is injected as the type it
    # refers to, sharing its instance
    configuration = Configuration()
    injector = Injector(configuration)
    assert isinstance(injector.get(NeedsAliasOfA).a, A)
    assert id(injector.get(NeedsAliasOfA).a) == id(injector.get(A))
    # Asked for directly as well, and through the binding of the type
    # it refers to
    assert injector.get(alias_of_a) is injector.get(A)
    configuration = Configuration()
    configuration.bind(A).globally().to_class(B)
    injector = Injector(configuration)
    assert isinstance(injector.get(alias_of_a), B)

    # An alias of a union is still a union
    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(NeedsAliasOfUnion)

    # And binding the alias itself wins over what it refers to
    configuration = Configuration()
    configuration.bind(cast(type[A], alias_of_a)).globally().to_class(B)
    configuration.bind(cast(type[A], alias_of_union)).globally().to_class(A)
    injector = Injector(configuration)
    assert isinstance(injector.get(NeedsAliasOfA).a, B)
    assert isinstance(injector.get(NeedsAliasOfUnion).a, A)


@pytest.mark.skipif(
    not hasattr(typing, "TypeAliasType"),
    reason="PEP 695 `type Foo = ...` aliases need Python 3.12+",
)
def test_pep695_type_alias_is_checked_as_its_target() -> None:
    class AbstractFoo(ABC):
        @abstractmethod
        def foo(self) -> None: ...

    class ProtocolFoo(Protocol):
        def foo(self) -> None: ...

    UserId = NewType("UserId", int)

    # An alias is checked as the type it refers to, so the error names
    # what is wrong with it instead of failing in the constructor
    type_alias_type: Any = getattr(typing, "TypeAliasType", None)
    for alias, kind in (
        (type_alias_type("AliasOfAbstract", AbstractFoo), "abstract"),
        (type_alias_type("AliasOfProtocol", ProtocolFoo), "a Protocol"),
        (type_alias_type("AliasOfNewType", UserId), "a NewType"),
    ):
        configuration = Configuration()
        injector = Injector(configuration)
        with pytest.raises(InjectorConfigurationError, match=rf"is {kind}"):
            injector.get(alias)


@pytest.mark.skipif(
    not hasattr(typing, "TypeAliasType"),
    reason="PEP 695 `type Foo = ...` aliases need Python 3.12+",
)
def test_pep695_type_alias_loop() -> None:
    # Only the `type` statement can build an alias that refers back to
    # itself, as it evaluates its value lazily. Run as text so this
    # module still parses on Python < 3.12
    aliases: dict[str, Any] = {"Annotated": Annotated}
    exec(  # noqa: S102
        "type Loop = Loop\n"
        "type Ping = Pong\n"
        "type Pong = Ping\n"
        "type Wrapped = Annotated[Wrapped, 'meta']\n"
        "type Json = dict[str, Json] | list[Json] | str | None\n",
        aliases,
    )

    loop = aliases["Loop"]

    class NeedsLoop:
        def __init__(self, a: loop) -> None:  # type: ignore[valid-type]
            self.a = a

    # An alias loop has no type to build, so it is an error rather
    # than an endless loop, asked for directly or as a dependency
    configuration = Configuration()
    injector = Injector(configuration)
    for name in ("Loop", "Ping", "Wrapped"):
        with pytest.raises(InjectorConfigurationError, match="refers back to itself"):
            injector.get(aliases[name])
    with pytest.raises(InjectorConfigurationError, match="refers back to itself"):
        injector.get(NeedsLoop)

    # A recursive alias is not a loop: only its top level is resolved,
    # and here that is a union
    with pytest.raises(InjectorConfigurationError, match="is a union"):
        injector.get(aliases["Json"])


def test_class_with_value_attribute_is_not_an_alias() -> None:
    class A:
        __value__ = "some value"

    class NeedsA:
        def __init__(self, a: A) -> None:
            self.a = a

    # Only a `type` alias resolves to what it refers to. A class with
    # a `__value__` attribute of its own is just a class
    configuration = Configuration()
    injector = Injector(configuration)
    assert isinstance(injector.get(A), A)
    assert isinstance(injector.get(NeedsA).a, A)


# Text-based annotations, either quoted forward references or the ones
# produced by `from __future__ import annotations`, can only be resolved
# for types reachable from the globals of the constructor's module
class TextAnnotation_Dep:
    pass


class TextAnnotation_OtherDep:
    pass


class TextAnnotation_UsesClass:
    def __init__(self, dep: "TextAnnotation_Dep") -> None:
        self.dep = dep


class TextAnnotation_UsesPep604:
    def __init__(self, dep: "TextAnnotation_Dep | None") -> None:
        self.dep = dep


class TextAnnotation_UsesOptional:
    def __init__(self, dep: "Optional[TextAnnotation_Dep]") -> None:
        self.dep = dep


class TextAnnotation_UsesUnion:
    def __init__(
        self, dep: "Union[TextAnnotation_Dep, TextAnnotation_OtherDep]"
    ) -> None:
        self.dep = dep


def test_text_based_annotations() -> None:
    # Plain classes are injected as usual
    configuration = Configuration()
    injector = Injector(configuration)
    assert isinstance(injector.get(TextAnnotation_UsesClass).dep, TextAnnotation_Dep)

    # Unions are recognized, whichever their spelling, so they will
    # fail unless bound
    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(InjectorConfigurationError):
        injector.get(TextAnnotation_UsesPep604)
    with pytest.raises(InjectorConfigurationError):
        injector.get(TextAnnotation_UsesOptional)
    with pytest.raises(InjectorConfigurationError):
        injector.get(TextAnnotation_UsesUnion)

    # And binding them works. A single `Optional` binding covers the
    # PEP 604 spelling of the same union as well
    configuration = Configuration()
    configuration.bind(
        cast(type[TextAnnotation_Dep], Optional[TextAnnotation_Dep])
    ).globally().to_class(TextAnnotation_Dep)
    configuration.bind(
        cast(
            type[TextAnnotation_Dep],
            Union[TextAnnotation_Dep, TextAnnotation_OtherDep],
        )
    ).globally().to_class(TextAnnotation_OtherDep)
    injector = Injector(configuration)
    assert isinstance(injector.get(TextAnnotation_UsesPep604).dep, TextAnnotation_Dep)
    assert isinstance(injector.get(TextAnnotation_UsesOptional).dep, TextAnnotation_Dep)
    assert isinstance(
        injector.get(TextAnnotation_UsesUnion).dep, TextAnnotation_OtherDep
    )


def test_text_based_annotation_out_of_global_scope_fails() -> None:
    class LocalDep:
        pass

    class UsesLocalDep:
        def __init__(self, dep: "LocalDep | None") -> None:
            self.dep = dep

    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(InjectorInstantiationError):
        injector.get(UsesLocalDep)


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
    with pytest.raises(
        InjectorConfigurationError, match="`a_or_b` because it is a NewType"
    ):
        injector.get(UsesNewType)
    with pytest.raises(InjectorConfigurationError):
        injector.get(UsesAlias)

    configuration = Configuration()
    configuration.bind(AOrB).globally().to_class(A)
    configuration.bind(cast(type[B], BOrA)).globally().to_class(B)
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
    configuration.bind(cast(type[Container[str]], Container)).globally().to_instance(
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
    # With no binding for it, Container[int] gets the default
    # instantiation, that has nothing standing in for the TypeVar
    # of its `value: T` param
    with pytest.raises(
        InjectorConfigurationError, match="param `value` because it is a TypeVar"
    ):
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


def test_thread_safe_injector_builds_a_single_instance() -> None:
    threads = 20

    def distinct_instances(injector: Injector, cls: type[Any]) -> int:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            return len(
                {
                    id(i)
                    for i in executor.map(lambda _: injector.get(cls), range(threads))
                }
            )

    # Without a lock, every thread asking while the first build is under
    # way builds its own. Forced with a barrier that only opens once all
    # of them are inside the constructor, so it is not left to timing
    barrier = threading.Barrier(threads, timeout=5)

    class Contended:
        def __init__(self) -> None:
            barrier.wait()

    configuration = Configuration()
    injector = Injector(configuration, thread_safe=False)
    assert distinct_instances(injector, Contended) == threads

    # With one, the default, the first build is the only one, and
    # everybody shares it
    built: list[object] = []

    class Slow:
        def __init__(self) -> None:
            # Long enough for every other thread to ask meanwhile
            time.sleep(0.01)
            built.append(self)

    configuration = Configuration()
    injector = Injector(configuration)
    assert distinct_instances(injector, Slow) == 1
    assert len(built) == 1


def test_thread_safe_injector_is_reentrant() -> None:
    class Inner:
        pass

    class Outer:
        def __init__(self, injector: Injector) -> None:
            # Asks the injector for more while being built by it
            self.inner = injector.get(Inner)

    configuration = Configuration()
    injector = Injector(configuration)
    configuration.bind(Injector).globally().to_instance(injector)
    assert isinstance(injector.get(Outer).inner, Inner)
