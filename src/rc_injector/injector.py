import collections.abc
import inspect
import types
import typing
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
    cast,
)

T = TypeVar("T")

PRIMITIVE_TYPES = {
    bool,
    bytearray,
    bytes,
    complex,
    dict,
    float,
    frozenset,
    int,
    list,
    set,
    str,
    tuple,
}

# Modules holding the stdlib container types. Matching by module covers
# the whole `collections` family without listing every type, and only
# for the stdlib: a container class of your own stays injectable, as it
# is a dependency like any other class.
CONTAINER_MODULES = {"builtins", "collections"}
# Its members are all abstract interfaces, so none of them has an
# implementation to build
ABSTRACT_INTERFACES_MODULE = "collections.abc"

# The PEP 604 syntax `Foo | None` builds a `types.UnionType`, while
# `Optional[Foo]` and `Union[Foo, Bar]` build a `typing.Union`, and
# `typing.get_origin()` reports a different origin for each of them.
# Both flavors compare and hash equal, so they are interchangeable as
# binding keys, but any check on the origin has to accept both.
UNION_ORIGINS: set[Any] = {typing.Union, typing.Optional, types.UnionType}


def _eval_annotation(annotation: str, globalns: dict[str, Any]) -> Any:
    """
    Evaluate a text annotation on the scope it was defined in.

    Annotations are text when they are quoted forward references or
    when the module uses `from __future__ import annotations`. They
    can be plain class names ("Foo"), but also expressions such as
    "Foo | None", "Optional[Foo]" or "Union[Foo, Bar]".
    """
    # Evaluated on a copy of the globals, as `eval()` would otherwise
    # inject `__builtins__` into the module that defined the annotation
    return eval(annotation, dict(globalns))  # noqa: S307


# `type Foo = ...` aliases (PEP 695) build a `typing.TypeAliasType`,
# that only exists on Python 3.12+. With no such type, `isinstance()`
# against an empty tuple is simply never true
_TYPE_ALIAS_TYPES: tuple[type[Any], ...] = (
    (typing.TypeAliasType,) if hasattr(typing, "TypeAliasType") else ()
)


def _is_transparent_type(cls: Any) -> bool:
    """
    Whether the annotation just decorates another type.

    `Annotated[Foo, ...]` (PEP 593) and the `type Foo = ...` aliases
    (PEP 695) are transparent for type checkers, they are only a
    decorated way to refer to another type.

    Note that `NewType`s are **not** transparent: unlike the above,
    they are a distinct type and not a mere alias of their supertype.
    """
    return typing.get_origin(cls) is Annotated or isinstance(cls, _TYPE_ALIAS_TYPES)


def _unwrap_transparent_type(cls: Any) -> Any:
    """
    The type one layer of transparent annotation refers to.

    Checked apart from the unwrapping, with `_is_transparent_type()`,
    as an alias can refer to itself and then unwraps to the very same
    object.
    """
    if typing.get_origin(cls) is Annotated:
        return typing.get_args(cls)[0]
    # PEP 695 aliases keep the type they refer to in `__value__`
    return cls.__value__


def _check_alias_does_not_loop(seen: list[Any], step: Any) -> None:
    # An alias can refer back to itself (`type Foo = Foo`), directly or
    # through other aliases, and then there is nothing it stands for
    if step in seen:
        chain = " -> ".join(str(item) for item in [*seen, step])
        raise InjectorConfigurationError(
            f"{seen[0]} is an alias that refers back to itself "
            f"({chain}), so there is no type to build for it"
        )


def _unwrap_transparent_types(cls: Any) -> Any:
    """
    Resolve every layer of transparent annotation, down to the type the
    aliases ultimately refer to. See `_unwrap_transparent_type()`.
    """
    seen: list[Any] = [cls]
    while _is_transparent_type(cls):
        unwrapped = _unwrap_transparent_type(cls)
        _check_alias_does_not_loop(seen, unwrapped)
        seen.append(unwrapped)
        cls = unwrapped
    return cls


def _is_value_type(cls: Any) -> bool:
    """
    Whether the annotation is a value instead of a dependency.

    Values can't be injected, as there is nothing to build for them.
    A container is a value even when its items are injectable:
    `list[Foo]` is never built as `[injected_foo]`, the value has to
    be provided. Note that the containers that can be built with no
    arguments would otherwise be injected silently empty.

    Only mandatory params are checked against this: one with a default
    value in the signature uses it, whatever its type.
    """
    # A parameterized container is a value as much as a bare one
    cls = typing.get_origin(cls) or cls
    if cls in PRIMITIVE_TYPES or cls is Literal:
        return True
    if not isinstance(cls, type):
        return False
    module = getattr(cls, "__module__", None)
    if module == ABSTRACT_INTERFACES_MODULE:
        # `Sequence[Foo]` and `Callable[[int], str]` alike
        return True
    # Iterable, rather than Collection, so that the containers that only
    # define `__iter__` are covered as well
    return module in CONTAINER_MODULES and issubclass(cls, collections.abc.Iterable)


class InjectorError(Exception):
    pass


class InjectorConfigurationError(InjectorError):
    pass


class InjectorInstantiationError(InjectorError):
    pass


class CircularDependencyError(InjectorError):
    pass


class Abstract(Generic[T]):
    """
    Typing marker for the abstract classes passed to the injector.

    mypy rejects an abstract class where `type[T]` is expected, as it
    could not be instantiated. The parameters that take a class are
    typed `type[T] | Abstract[T]` so that an abstract class is
    accepted too, and it stays fully type safe: `injector.get(Foo)` is
    a `Foo` for an abstract `Foo` as much as for a concrete one.

    It is only a type. Pass the class itself, `bind(Foo)` and
    `injector.get(Foo)`, never an instance of this marker: the code
    asserts against one.
    """


class Param(NamedTuple):
    name: str
    type: type[Any]
    default: Any


class TypeResolver(Generic[T]):
    """
    Knows how to resolve a type into a concrete instance.
    """

    cls: type[T]
    _to_instance: T | None
    _to_class: type[Any] | None
    _to_constructor: Callable[..., T] | None
    _kwargs: dict[str, Any]
    _arg_types: dict[str, type[Any]]

    def __init__(self, cls: type[T]) -> None:
        self.cls = cls
        self._to_instance = None
        self._to_class = None
        self._to_constructor = None
        self._kwargs = {}
        self._arg_types = {}

    def to_instance(self, instance: T) -> None:
        """
        Bind the class to the specific given instance
        """
        if (
            self._to_class is not None
            or self._to_constructor is not None
            or self._kwargs
            or self._arg_types
        ):
            raise InjectorConfigurationError(
                f"Unable to bind {self.cls} to instance. Already bound: {self}"
            )
        self._to_instance = instance

    def to_class(self, cls: type[Any]) -> None:
        """
        Bind it to the provided class instead of original

        This is useful for abstract classes that need to
        be injected with a concrete implementation or for
        interfaces.
        """
        if (
            self._to_instance is not None
            or self._to_constructor is not None
            or self._kwargs
            or self._arg_types
        ):
            raise InjectorConfigurationError(
                f"Unable to bind {self.cls} to class. Already bound: {self}"
            )
        self._to_class = cls

    def to_constructor(self, constructor: Callable[..., T]) -> "TypeResolver[T]":
        """
        Use provided function to build the class
        """
        if self._to_instance is not None or self._to_class is not None:
            raise InjectorConfigurationError(
                f"Unable to bind {self.cls} to class. Already bound: {self}"
            )
        self._to_constructor = constructor
        return self

    def with_kwargs(self, **kwargs: Any) -> "TypeResolver[T]":
        """
        Define arguments for the class or the constructor

        Useful for primitive values that can't be injected or
        customizing specific instances.
        """
        if self._to_instance is not None:
            raise InjectorConfigurationError(
                f"Unable to define kwargs for {self.cls}: "
                "It is bound to a specific instance"
            )
        if self._to_class is not None:
            raise InjectorConfigurationError(
                f"Unable to define kwargs for {self.cls}: "
                f"It is bound to class {self._to_class}. "
                "Configure kwargs for that class directly"
            )
        for k, v in kwargs.items():
            self._kwargs[k] = v
        return self

    def with_arg_types(self, **kwargs: Any) -> "TypeResolver[T]":
        """
        Define the type to use for the class or the constructor

        Useful when a constructor has an Optional or Union and it
        is not clear what value to inject.
        """
        if self._to_instance is not None:
            raise InjectorConfigurationError(
                f"Unable to define arg types for {self.cls}: "
                "It is bound to a specific instance"
            )
        if self._to_class is not None:
            raise InjectorConfigurationError(
                f"Unable to define arg types for {self.cls}: "
                f"It is bound to class {self._to_class}. "
                "Configure them on that class directly"
            )
        for k, v in kwargs.items():
            self._arg_types[k] = v
        return self

    def _resolve_text_annotation(
        self, constructor: Callable[..., Any], param: inspect.Parameter
    ) -> type[Any]:
        """
        Convert a text-based annotation into the real type it refers to.

        It is evaluated on the scope of the constructor's module, so
        only the types reachable from there can be resolved.
        """
        constructor_context = getattr(constructor, "__globals__", None)
        if constructor_context is None:
            raise InjectorInstantiationError(
                f"Unable to parse constructor signature for class {self.cls}: "
                f"Param {param.name} has a text signature {param.annotation} "
                "and we are unable to find the constructor globals."
            )
        try:
            return cast(
                type[Any], _eval_annotation(param.annotation, constructor_context)
            )
        except Exception as e:
            raise InjectorInstantiationError(
                f"Unable to parse constructor signature for class {self.cls}: "
                f"Param {param.name} has a text signature {param.annotation} "
                f"that we are unable to evaluate: {e}. Only the types "
                "reachable from the globals of the constructor's module "
                "can be resolved."
            ) from e

    def _get_params(
        self, constructor: Callable[..., Any], is_init: bool = False
    ) -> list[Param]:
        params: list[Param] = []
        for param in inspect.signature(constructor).parameters.values():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            param_type: type[Any]
            if isinstance(param.annotation, str):
                param_type = self._resolve_text_annotation(constructor, param)
            else:
                param_type = param.annotation

            params.append(
                Param(name=param.name, type=param_type, default=param.default)
            )
        return params[1:] if is_init else params

    def _check_cls_is_not_value_type(self, cls: Any) -> None:
        if _is_value_type(cls):
            # A value has nothing to build: a default-constructed one,
            # `0`, `''` or `[]`, is never what was meant
            raise InjectorConfigurationError(
                f"{self._describe(cls)} is a primitive or container type "
                "that can't be built. Provide the value with `with_kwargs()` "
                "on the class that needs it, or bind it to a specific value "
                "with `to_instance()`"
            )

    def _check_cls_is_not_union(self, cls: Any) -> None:
        if typing.get_origin(cls) in UNION_ORIGINS:
            # Unions have multiple alternatives, so there is nothing to build
            raise InjectorConfigurationError(
                f"{self._describe(cls)} is a union and it can't be "
                "instantiated directly. "
                "You need to bind it to one of its alternatives with "
                "bind(cast(type[Foo], Optional[Foo])).globally().to_class(Foo)"
            )

    def _describe(self, cls: Any) -> str:
        # The class as it was asked for and, when that is an alias,
        # what it refers to, which is what the checks look at
        if cls is self.cls:
            return str(self.cls)
        return f"{self.cls} (alias of {cls})"

    def _check_cls_is_not_abstract(self, cls: Any) -> None:
        if inspect.isabstract(cls):
            # Abstract classes can't be instantiated directly
            raise InjectorConfigurationError(
                f"{self._describe(cls)} is abstract and it can't be injected. "
                "You need to bind a specific concrete implementation "
                "with bind(Foo).globally().to_class(ConcreteFoo)"
            )

    def _check_cls_is_not_protocol(self, cls: Any) -> None:
        if getattr(cls, "_is_protocol", False):
            # Protocol classes can't be instantiated directly
            raise InjectorConfigurationError(
                f"{self._describe(cls)} is a Protocol and it can't be injected. "
                "You need to bind a specific implementation of the Protocol"
                "with bind(MyProtocol).globally().to_class(ImplementsMyProtocol)"
            )

    def _check_cls_is_not_newtype(self, cls: Any) -> None:
        if callable(cls) and getattr(cls, "__supertype__", False):
            # NewType classes can't be instantiated directly
            raise InjectorConfigurationError(
                f"{self._describe(cls)} is a NewType and it can't be injected. "
                "You need to bind a specific implementation of the Protocol"
                "with bind(MyNewType).globally().to_class(Foo)"
            )

    def _check_param_can_be_built(
        self, constructor: Callable[..., Any], param: Param, is_bound: bool
    ) -> None:
        def constructor_error_context() -> str:
            if constructor == self.cls:
                constructor_name = "__init__"
            elif fn_name := getattr(constructor, "__name__", None):
                constructor_name = fn_name
            else:
                constructor_name = str(constructor)
            cls_name = str(self.cls)
            return (
                f"Error in constructor `{constructor_name}()` for class `{cls_name}`: "
            )

        def primitive_param_error() -> InjectorConfigurationError:
            return InjectorConfigurationError(
                constructor_error_context()
                + f"Constructor param `{param.name}` is a primitive or "
                f"container type: `{param.type}` that can't be injected. "
                "Provide the value to use with `with_kwargs()` in the "
                "injector configuration."
            )

        if not param.type or param.type == inspect.Parameter.empty:
            raise InjectorConfigurationError(
                constructor_error_context()
                + f"Constructor param `{param.name}` is not typed"
            )
        if param.type in PRIMITIVE_TYPES:
            raise primitive_param_error()
        if not is_bound:
            # Checks only for non-bound param.type
            origin_cls = typing.get_origin(param.type)
            if _is_value_type(param.type):
                raise primitive_param_error()
            if isinstance(param.type, TypeVar):
                raise InjectorConfigurationError(
                    constructor_error_context()
                    + f"Unable to determine how to inject param `{param.name}` "
                    f"because it is a TypeVar: {param.type}. The generic is "
                    "built with nothing standing in for it, so provide the "
                    "value with `with_kwargs()`, or bind the class to an "
                    "instance with `to_instance()`"
                )
            if origin_cls in UNION_ORIGINS:
                raise InjectorConfigurationError(
                    constructor_error_context()
                    + f"Unable to determine how to inject param `{param.name}` "
                    f"because it has multiple alternatives: {param.type}. "
                    "You must provide specific value with `with_kwargs()` or "
                    "specific type to use with `with_arg_types()` in the "
                    "injector configuration."
                )
            if inspect.isabstract(param.type):
                raise InjectorConfigurationError(
                    constructor_error_context()
                    + f"Unable to determine how to inject param `{param.name}` "
                    f"because it is abstract: {param.type}. "
                    "You must provide a concrete implementation: "
                    "a) Creating a binding for the class "
                    "bind(Foo).globally().to_class(ConcreteFoo), "
                    "b) overriding the value on the parent class "
                    "with `with_kwargs()`, or "
                    "c) overriding the type on the parent class "
                    "with `with_arg_types()`"
                )
            if getattr(param.type, "_is_protocol", False):
                raise InjectorConfigurationError(
                    constructor_error_context()
                    + f"Unable to determine how to inject param `{param.name}` "
                    f"because it is a Protocol: {param.type}. "
                    "You must provide a class that implements the protocol: "
                    "a) Creating a binding for the class "
                    "bind(Foo).globally().to_class(ImplementsFoo), "
                    "b) overriding the value on the parent class "
                    "with `with_kwargs()`, or "
                    "c) overriding the type on the parent class "
                    "with `with_arg_types()`"
                )
            if callable(param.type) and getattr(param.type, "__supertype__", False):
                raise InjectorConfigurationError(
                    constructor_error_context()
                    + f"Unable to determine how to inject param `{param.name}` "
                    f"because it is a NewType: {param.type}. "
                    "If the NewType is a union, you can bind it "
                    "to a concrete class. If it is a primitive type, you "
                    "should likely use `with_kwargs()` to define its value "
                    "on the class that needs the param"
                )

    def get_cached_instance(self) -> T | None:
        return self._to_instance

    def resolve_type(self, injector_context: "InjectorContext") -> T:
        if self._to_instance is not None:
            return self._to_instance
        if self._to_class is not None:
            # Cached as well, so that once the class it is bound to is
            # built, this one is handed out with no further resolution
            instance = cast(T, injector_context.get(self._to_class))
            self._to_instance = instance
            return instance
        if self._to_constructor is not None:
            constructor = self._to_constructor
            params = self._get_params(constructor)
        else:
            # No overrides, we will build the class itself: the one an
            # alias refers to, when it is one, and the checks look at it
            cls = _unwrap_transparent_types(self.cls)
            self._check_cls_is_not_value_type(cls)
            self._check_cls_is_not_union(cls)
            self._check_cls_is_not_abstract(cls)
            self._check_cls_is_not_protocol(cls)
            self._check_cls_is_not_newtype(cls)
            if origin_cls := typing.get_origin(cls):
                # For generic classes, we need to instantiate the
                # origin class
                cls = origin_cls

            constructor = cls
            params = self._get_params(cls.__init__, is_init=True)

        kwargs = dict(self._kwargs)
        for param in params:
            if param.name in kwargs:
                # Already provided in the configuration
                continue
            if overriden_param_type := self._arg_types.get(param.name):
                kwargs[param.name] = injector_context.get(overriden_param_type)
                continue
            # An alias with no binding of its own is the type it refers
            # to, bindings included. And bound for this class, as its
            # params are resolved with it as parent: a binding scoped to
            # another parent does not apply, so it neither overrides a
            # default nor stands in for the checks below
            configuration = injector_context.configuration
            param = param._replace(
                type=configuration.resolve_alias(param.type, parent_cls=self.cls)
            )
            is_bound = configuration.has_configured_bindings(
                param.type, parent_cls=self.cls
            )
            has_default_value = param.default != inspect.Parameter.empty
            if has_default_value and not is_bound:
                # There is a default value for the param, so
                # we will use it unless there is an specific
                # binding for the type.
                kwargs[param.name] = param.default
                continue
            self._check_param_can_be_built(
                constructor=constructor,
                param=param,
                is_bound=is_bound,
            )
            kwargs[param.name] = injector_context.get(param.type)

        try:
            instance = constructor(**kwargs)
        except Exception as e:
            raise InjectorInstantiationError(
                f"Unable to instantiate {self.cls}. Constructor raised Exception: {e}"
            ) from e
        # Cache instantiation for next usage
        self._to_instance = instance
        return instance


class Binding(Generic[T]):
    def __init__(
        self,
        cls: type[T],
    ):
        self.cls = cls
        self.global_resolver: TypeResolver[T] | None = None
        self.scoped_resolvers: dict[type[Any], TypeResolver[T]] = {}

    def globally(self) -> TypeResolver[T]:
        if self.global_resolver is None:
            self.global_resolver = TypeResolver[T](self.cls)
        return self.global_resolver

    def for_parent(self, parent_cls: type[Any]) -> TypeResolver[T]:
        if parent_cls not in self.scoped_resolvers:
            self.scoped_resolvers[parent_cls] = TypeResolver[T](self.cls)
        return self.scoped_resolvers[parent_cls]

    def get_type_resolver(self, parent_cls: type[T] | None) -> TypeResolver[T] | None:
        if parent_cls and parent_cls in self.scoped_resolvers:
            return self.scoped_resolvers[parent_cls]
        return self.global_resolver


class Configuration:
    bindings: dict[type[Any], Binding[Any]]
    _default_type_resolvers: dict[type[Any], TypeResolver[Any]]
    _settled_type_resolvers: dict[tuple[Any, Any], TypeResolver[Any]]

    def __init__(self) -> None:
        self.bindings = {}
        self._default_type_resolvers = {}
        # What each (class, parent) asked for resolves to, once settled,
        # so that handing out an instance is one lookup whatever the
        # bindings and aliases involved. It only depends on the bindings,
        # which is why those can't change once resolving has started
        self._settled_type_resolvers = {}

    def bind(self, cls: type[T] | Abstract[T]) -> Binding[T]:
        # Abstract is just a trick to make mypy like
        # abstract types passed into our injector
        assert not isinstance(cls, Abstract)  # noqa: S101
        if cls in PRIMITIVE_TYPES:
            raise InjectorConfigurationError(
                "Primitive types can't be bound. If you need to inject "
                "a specific value, use `with_kwargs()` on the parent class"
            )
        if self._settled_type_resolvers:
            raise InjectorConfigurationError(
                f"Unable to bind {cls}: the configuration can't change once the "
                "injector has started resolving, as what each class resolves "
                "to is settled on first use. Complete the bindings first."
            )
        if cls not in self.bindings:
            self.bindings[cls] = Binding[T](cls)
        return self.bindings[cls]

    def _get_default_resolver(self, cls: type[T]) -> TypeResolver[T]:
        return TypeResolver[T](cls)

    def get_type_resolver(
        self, cls: type[T], parent_cls: type[Any] | None
    ) -> TypeResolver[T]:
        """
        The resolver to build the class with, for the given parent.

        A binding scoped to the parent wins over a global one. With no
        binding that applies, the class gets the default instantiation,
        as any class with no bindings at all: a binding scoped to
        another parent is not a binding for this one.

        A transparent alias with no binding of its own is resolved as
        the type it refers to, so it shares its binding and its
        instance. See `resolve_alias()`.
        """
        key = (cls, parent_cls)
        type_resolver = self._settled_type_resolvers.get(key)
        if type_resolver is None:
            type_resolver = self._resolve_type_resolver(cls, parent_cls)
            self._settled_type_resolvers[key] = type_resolver
        return type_resolver

    def _resolve_type_resolver(
        self, cls: type[T], parent_cls: type[Any] | None
    ) -> TypeResolver[T]:
        cls = self.resolve_alias(cls, parent_cls=parent_cls)
        if cls in self.bindings:
            type_resolver = self.bindings[cls].get_type_resolver(parent_cls=parent_cls)
            if type_resolver is not None:
                return type_resolver
        if cls not in self._default_type_resolvers:
            self._default_type_resolvers[cls] = self._get_default_resolver(cls)
        return self._default_type_resolvers[cls]

    def resolve_alias(self, cls: type[T], parent_cls: type[Any] | None) -> type[Any]:
        """
        The type a transparent alias stands for, for the given parent.

        `Annotated[Foo, ...]` and `type Foo = ...` aliases are only a
        decorated way to refer to another type, so with no binding of
        their own they are the type they refer to, bindings included.
        The layers are peeled one at a time, as a binding on any of
        them is where the resolution stops. Anything that is not an
        alias is returned as is.
        """
        seen: list[Any] = [cls]
        while _is_transparent_type(cls) and not self.has_configured_bindings(
            cls, parent_cls=parent_cls
        ):
            unwrapped = _unwrap_transparent_type(cls)
            _check_alias_does_not_loop(seen, unwrapped)
            seen.append(unwrapped)
            cls = unwrapped
        return cls

    def has_configured_bindings(
        self, cls: type[T] | Abstract[T], parent_cls: type[Any] | None = None
    ) -> bool:
        """
        Whether a binding applies to the class when the parent asks for
        it, or at the root with no parent. Same rules as
        `get_type_resolver()`, so a binding scoped to another parent
        does not count.
        """
        if cls not in self.bindings:
            return False
        # Abstract is just a trick to make mypy like
        # abstract types passed into our injector
        assert not isinstance(cls, Abstract)  # noqa: S101
        return self.bindings[cls].get_type_resolver(parent_cls=parent_cls) is not None


class Injector:
    _configuration: Configuration

    def __init__(self, configuration: Configuration) -> None:
        self._configuration = configuration

    def get(
        self,
        cls: type[T] | Abstract[T],
        parent_cls: type[Any] | None = None,
    ) -> T:
        # Abstract is just a trick to make mypy like
        # abstract types passed into our injector
        assert not isinstance(cls, Abstract)  # noqa: S101
        # Fast path for already cached instances, that avoids having
        # to create a new injection context. Compared against None, so
        # that a falsy instance takes it as well.
        type_resolver = self._configuration.get_type_resolver(cls, parent_cls)
        instance = type_resolver.get_cached_instance()
        if instance is not None:
            return instance
        # Create a new injection context and get the instance from it.
        # The parent is handed over to it, as it is what selects the
        # bindings scoped to it.
        return InjectorContext(
            configuration=self._configuration, parent_cls=parent_cls
        ).get(cls)


class InjectorContext:
    configuration: Configuration
    stack: list[TypeResolver[Any]]

    def __init__(
        self, configuration: Configuration, parent_cls: type[Any] | None = None
    ) -> None:
        self.configuration: Configuration = configuration
        # What this context is building, innermost last. The resolvers
        # rather than the classes asked for: an alias is resolved as
        # the type it refers to, and the same class under another
        # binding is another thing to build. It only ever holds what
        # is under construction, so a resolver needing itself is a cycle
        self.stack: list[TypeResolver[Any]] = []
        # On whose behalf the root class is resolved. It selects the
        # scoped bindings for it, but it is not being built, so it does
        # not belong in the stack
        self.parent_cls = parent_cls

    def get(self, cls: type[T]) -> T:
        # Whatever is being built is the parent of its dependencies. The
        # root one has the parent it is resolved on behalf of, if any
        parent_cls = self.stack[-1].cls if self.stack else self.parent_cls
        type_resolver = self.configuration.get_type_resolver(cls, parent_cls=parent_cls)
        if type_resolver in self.stack:
            classes = [resolver.cls for resolver in self.stack]
            raise CircularDependencyError(
                f"Unable to instantiate {classes[0]} because {cls} "
                "causes a circular dependency. To build it, "
                f"ultimately {classes[-1]} is needed that needs {cls} "
                f"again. Dependency stack is: {classes}"
            )
        self.stack.append(type_resolver)
        try:
            return type_resolver.resolve_type(injector_context=self)
        finally:
            self.stack.pop()
