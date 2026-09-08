import collections
import time
import typing
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Callable,
    Generic,
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


def test_circular_dependency() -> None:
    configuration = Configuration()
    injector = Injector(configuration)
    with pytest.raises(CircularDependencyError):
        injector.get(CircularDep_A)

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
