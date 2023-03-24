import inspect
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

T = TypeVar("T")

PRIMITIVE_TYPES = {
    bytes,
    dict,
    float,
    int,
    list,
    set,
    str,
    tuple,
}


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
    Abstract classes can't be passed to functions
    expecting Type[T] as they expect a concrete
    class that can be instantiated.
    This is a hack to be able to pass a Abstract
    Type, without mypy complaining, but still fully type safe
    To configure an abstract class "Foo", instead of
    passign just "Foo", pass "Abstract[Foo]()" to
    the injector and for the configuration.
    )
    """

    pass


class Param(NamedTuple):
    name: str
    type: Type[Any]
    default: Any


class TypeResolver(Generic[T]):
    """
    Knows how to resolve a type into a concrete instance.
    """

    cls: Type[T]
    _to_instance: Optional[T]
    _to_class: Optional[Type[Any]]
    _to_constructor: Optional[Callable[..., T]]
    _kwargs: Dict[str, Any]
    _arg_types: Dict[str, Type[Any]]

    def __init__(self, cls: Type[T]) -> None:
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
                f"Unable to bind {self.cls} to instance. Already binded: {self}"
            )
        self._to_instance = instance

    def to_class(self, cls: Type[Any]) -> None:
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
                f"Unable to bind {self.cls} to class. Already binded: {self}"
            )
        self._to_class = cls

    def to_constructor(self, constructor: Callable[..., T]) -> "TypeResolver[T]":
        """
        Use provided function to build the class
        """
        if self._to_instance is not None or self._to_class is not None:
            raise InjectorConfigurationError(
                f"Unable to bind {self.cls} to class. Already binded: {self}"
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
                "It is binded to an specific instance"
            )
        if self._to_class is not None:
            raise InjectorConfigurationError(
                f"Unable to define kwargs for {self.cls}: "
                f"It is binded to class {self._to_class}. "
                "Configure kwargs for that class directly"
            )
        for k, v in kwargs.items():
            self._kwargs[k] = v
        return self

    def with_arg_types(self, **kwargs: Any) -> "TypeResolver[T]":
        """
        Define the type to used for the class or the constructor

        Useful when a constructor has an Optional or Union and it
        is not clear what value to inject.
        """
        if self._to_instance is not None:
            raise InjectorConfigurationError(
                f"Unable to define arg types for {self.cls}: "
                "It is binded to an specific instance"
            )
        if self._to_class is not None:
            raise InjectorConfigurationError(
                f"Unable to define arg types for {self.cls}: "
                f"It is binded to class {self._to_class}. "
                "Configure them on that class directly"
            )
        for k, v in kwargs.items():
            self._arg_types[k] = v
        return self

    def _get_params(
        self, constructor: Callable[..., Any], is_init: bool = False
    ) -> List[Param]:
        params: List[Param] = []
        for param in inspect.signature(constructor).parameters.values():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            param_type: Type[Any]
            if isinstance(param.annotation, str):
                # Text-based annotation. Convert to real class
                candidate_param_type = constructor.__globals__.get(param.annotation)
                if candidate_param_type is None or not isinstance(
                    candidate_param_type, type
                ):
                    raise InjectorInstantiationError(
                        f"Unable to parse constructor signature for class {self.cls}: "
                        f"Param {param.name} has a text signature {param.annotation} "
                        "and we are unable to find that class in globals or not a "
                        "class."
                    )
                param_type = candidate_param_type
            else:
                param_type = param.annotation

            params.append(
                Param(name=param.name, type=param_type, default=param.default)
            )
        return params[1:] if is_init else params

    def _check_cls_is_not_abstract(self) -> None:
        if inspect.isabstract(self.cls):
            # Abstract classes can't be instantiated directly
            raise InjectorConfigurationError(
                f"{self.cls} is abstract and it can't be injected. "
                "You need to bind a specific concrete implementation "
                "with bind(Abstract(Foo)).globally().to_class(ConcreteFoo)"
            )

    def _check_cls_is_not_protocol(self) -> None:
        if getattr(self.cls, "_is_protocol", False):
            # Protocol classes can't be instantiated directly
            raise InjectorConfigurationError(
                f"{self.cls} is a Protocol and it can't be injected. "
                "You need to bind a specific implementation of the Protocol"
                "with bind(MyProtocol).globally().to_class(ImplementsMyProtocol)"
            )

    def _check_cls_is_not_newtype(self) -> None:
        if callable(self.cls) and getattr(self.cls, "__supertype__", False):
            # NewType classes can't be instantiated directly
            raise InjectorConfigurationError(
                f"{self.cls} is a NewType and it can't be injected. "
                "You need to bind a specific implementation of the Protocol"
                "with bind(MyNewType).globally().to_class(Foo)"
            )

    def _check_param_can_be_built(
        self, constructor: Callable[..., Any], param: Param, is_binded: bool
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

        if not param.type or param.type == inspect.Parameter.empty:
            raise InjectorConfigurationError(
                constructor_error_context()
                + f"Constructor param `{param}` is not typed"
            )
        if param.type in PRIMITIVE_TYPES:
            raise InjectorConfigurationError(
                constructor_error_context()
                + "Constructor param `{param.name}` is a primitive type: "
                f"`{param.type}` that can't be injected. Provide the value "
                "to use with `with_kwargs()` in the injector configuration."
            )
        if not is_binded:
            # Checks only for non-binded param.type
            origin_cls = typing.get_origin(param.type)
            if origin_cls in (typing.Union, typing.Optional):
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
            if callable(self.cls) and getattr(self.cls, "__supertype__", False):
                raise InjectorConfigurationError(
                    constructor_error_context()
                    + f"Unable to determine how to inject param `{param.name}` "
                    f"because it is a NewType: {param.type}. "
                    "If the NewType is a union, you can bind it "
                    "to a concrete class. If it is a primitive type, you "
                    "should likely use `with_kwargs()` to define its value "
                    "on the class that needs the param"
                )

    def get_cached_instance(self) -> Optional[T]:
        return self._to_instance

    def resolve_type(self, injector: "InjectorContext") -> T:
        if self._to_instance is not None:
            return self._to_instance
        if self._to_class is not None:
            return cast(T, injector.get(self._to_class))
        if self._to_constructor is not None:
            constructor = self._to_constructor
            params = self._get_params(constructor)
        else:
            # No overrides, we will build the class itself
            self._check_cls_is_not_abstract()
            self._check_cls_is_not_protocol()
            self._check_cls_is_not_newtype()

            cls = self.cls
            if origin_cls := typing.get_origin(self.cls):
                # For generic classes, we need to instantiate the
                # origin class
                cls = origin_cls

            constructor = cls
            params = self._get_params(cls.__init__, is_init=True)

        kwargs = dict(self._kwargs)
        for param in params:
            if param.name in kwargs:
                # Already provider in the configuration
                continue
            if overriden_param_type := self._arg_types.get(param.name):
                kwargs[param.name] = injector.get(overriden_param_type)
                continue
            is_binded = injector.configuration.has_configured_bindings(param.type)
            has_default_value = param.default != inspect.Parameter.empty
            if has_default_value and not is_binded:
                # There is a default value for the param, so
                # we will use it unless there is an specific
                # binding for the type.
                kwargs[param.name] = param.default
                continue
            self._check_param_can_be_built(
                constructor=constructor,
                param=param,
                is_binded=is_binded,
            )
            kwargs[param.name] = injector.get(param.type)

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
        cls: Type[T],
    ):
        self.cls = cls
        self.global_resolver: Optional[TypeResolver[T]] = None
        self.scoped_resolvers: Dict[Type[Any], TypeResolver[T]] = {}

    def globally(self) -> TypeResolver[T]:
        if self.global_resolver is None:
            self.global_resolver = TypeResolver[T](self.cls)
        return self.global_resolver

    def for_parent(self, parent_cls: Type[Any]) -> TypeResolver[T]:
        if parent_cls not in self.scoped_resolvers:
            self.scoped_resolvers[parent_cls] = TypeResolver[T](self.cls)
        return self.scoped_resolvers[parent_cls]

    def get_type_resolver(
        self, parent_cls: Optional[Type[T]]
    ) -> Optional[TypeResolver[T]]:
        if parent_cls and parent_cls in self.scoped_resolvers:
            return self.scoped_resolvers[parent_cls]
        return self.global_resolver


class Configuration:
    bindings: Dict[Type[Any], Binding[Any]]
    _default_type_resolvers: Dict[Type[Any], TypeResolver[Any]]

    def __init__(self) -> None:
        self.bindings = {}
        self._default_type_resolvers = {}

    def bind(self, cls: Union[Type[T], Abstract[T]]) -> Binding[T]:
        # Abstract is just a trick to make mypy like
        # abstract types passed into our injector
        assert not isinstance(cls, Abstract)  # noqa: S101
        if cls in PRIMITIVE_TYPES:
            raise InjectorConfigurationError(
                "Primitive types can't be binded. If you need to inject "
                "a specific value, use `with_kwargs()` on the partent class"
            )
        if cls not in self.bindings:
            self.bindings[cls] = Binding[T](cls)
        return self.bindings[cls]

    def _get_default_resolver(self, cls: Type[T]) -> TypeResolver[T]:
        return TypeResolver[T](cls)

    def get_type_resolver(
        self, cls: Type[T], parent_cls: Optional[Type[Any]]
    ) -> Optional[TypeResolver[T]]:
        type_resolver: Optional[TypeResolver[T]] = None
        if cls in self.bindings:
            type_resolver = self.bindings[cls].get_type_resolver(parent_cls=parent_cls)
        if not type_resolver:
            if cls not in self._default_type_resolvers:
                self._default_type_resolvers[cls] = self._get_default_resolver(cls)
            return self._default_type_resolvers[cls]
        return type_resolver

    def has_configured_bindings(self, cls: Union[Type[T], Abstract[T]]) -> bool:
        return cls in self.bindings


class Injector:
    _configuration: Configuration

    def __init__(self, configuration: Configuration) -> None:
        self._configuration = configuration

    def get(
        self,
        cls: Union[Type[T], Abstract[T]],
        parent_cls: Optional[Type[Any]] = None,
    ) -> T:
        # Abstract is just a trick to make mypy like
        # abstract types passed into our injector
        assert not isinstance(cls, Abstract)  # noqa: S101
        if type_resolver := self._configuration.get_type_resolver(cls, parent_cls):
            # Fast path for already cached instances, and avoid having to
            # create a new injection context
            if instance := type_resolver.get_cached_instance():
                return instance
        # Create a new injection context and get the instance from it
        return InjectorContext(configuration=self._configuration).get(cls)


class InjectorContext:
    configuration: Configuration
    stack: List[Type[Any]]

    def __init__(self, configuration: Configuration) -> None:
        self.configuration: Configuration = configuration
        self.stack: List[Type[Any]] = []

    def get(self, cls: Type[T]) -> T:
        if cls in self.stack:
            raise CircularDependencyError(
                f"Unable to instantiate {self.stack[0]} because {cls} "
                "causes a circular dependency. To build it, "
                f"ultimately {self.stack[-1]} is needed that needs {cls} "
                f"again. Dependency stack is: {self.stack}"
            )
        parent_cls = self.stack[-1] if self.stack else None
        type_resolver = self.configuration.get_type_resolver(cls, parent_cls=parent_cls)
        if type_resolver is None:
            raise InjectorInstantiationError(f"Unable to find a TypeResolver for {cls}")
        self.stack.append(cls)
        try:
            instance = type_resolver.resolve_type(injector=self)
        finally:
            self.stack.pop()
        return instance
