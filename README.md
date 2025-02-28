# rc-injector
Python dependency injector.

# Usage:

## Install:
```
pip install rc-injector
```
## Example usage:
Suppose you have your app with blueprints, and the blueprints can use a bunch of helpers like ConfigurationProvider, CacheClient, DBClient, Events, Jobs...

If you want to use dependency injection, you will have:

```python:
class App:
    def __init__(self, foo: FooBluePrint, bar: BarBluePrint) -> None:
        self.foo = foo
        self.bar = bar

    def do_foo_action(self) -> ...:
        return self.foo.action()

class FooBluePrint(BluePrint):
    def __init__(self, db: DBClient) -> None:
        self.db = db

    def action(self) -> ...:
        self.db.query...
        return ...
```

This architecture is great, but is cumbersome to use, because building the whole dependency tree by hand is tedious.

With rc-injector, this can be as simple as:

```python:
from rc_injector import Configuration, Injector

configuration = Configuration()

injector = Injector(configuration)
injector.get(App)
```

The injector can figure out how to build the dependency tree from the type hints.

Of course, this only works with classes that do not require configuration. It will more likely configure a few of the low level dependencies that need to be configured. For example:

```python:
from rc_injector import Configuration, Injector

prod_configuration_manager = ConfiguratioManager(...)

def build_prod_db_client() -> DBClient:
    ...

class CacheClient:
    def __init__(self, cfg: ConfigurationProvider, pool: str) -> None:
        ...

configuration = Configuration()
configuration.bind(ConfigurationProvider).globally().to_instance(prod_configuration_manager)
configuration.bind(DBClient).globally().to_constructor(build_prod_db_client)
configuration.bind(CacheClient).globally().with_kwargs(pool=CachePools.DEFAULT)
configuration.bind(Events).globally().to_class(KafkaEvents)
configuration.bind(KafkaEvents).globally().with_kwargs(queue="default")

injector = Injector(configuration)
injector.get(App)
```

A few observations:
1. We use `to_instance` to bind `ConfigurationProvider` to an specific instance that will act as singleton.
2. `to_constructor` helps us use a function helper to build the instance. Note that the instance will still behave as singleton, the functions will not be called for each usage.
3. `with_kwargs` allows us to define the value of some of the parameters of the class. This `CacheClient` might have a signature `__init__(self, cfg: ConfigurationProvider, pool: str)`. The `cfg` variable can be injected, but pool is a scalar so needs to be set to a particular value.
4. We use `to_class` to bind `Events` that is an abstract class with the interface to `KafkaEvents` that implements it using Kafka. We also define the queue name to use using `with_kwargs` to override the `queue` param.

Now imagine that `FooBluePrint` from the example now needs `CacheClient` due to some new features. You would just modify the signature with the new dependency:

```patch:
class FooBluePrint(BluePrint):
-    def __init__(self, db: DBClient) -> None:
+    def __init__(self, db: DBClient, cache: CacheClient) -> None:
```

Since the dependency is already configured, no changes to dependency injection are needed. The cache client will be ready to use!

Furthermore, tests will also use an injector. Integration tests might bind the real CacheClient to a local instance. Unit tests might mock it or provided a local implementation. So in this blueprint you won't need to worry about mocking cache client, worried about the test using production, etc.

Now imagine it's time for a refactor, we are going to split `FooBluePrint` into a few components and also use `Events`. Again, as long as there are no new low-level classes that require configuration, no changes to the injection are needed!

```patch:
+class FooDataAccess:
+    def __init__(self, db: DBClient, cache: CacheClient) -> None:
+        ...
+
+class FooEventSender:
+    def __init__(self, events: Events) -> None:
+        ...

class FooBluePrint(BluePrint):
-    def __init__(self, db: DBClient, cache: CacheClient) -> None:
+    def __init__(self, data: FooDataAccess, events: FooEventSender) -> None:
```

The cache pool usage is growing. `FooDataAccess` caches a lot of data and items are being evicted, causing a drop in hit rate. We want to move `FooDataAccess` cache to the best-effort pool. This is a configuration change, should not require complex changes to our application, and yeah, injector can help:

```patch:
configuration.bind(CacheClient).globally().with_kwargs(pool=CachePools.DEFAULT)
+ configuration.bind(CacheClient).for_parent(FooDataAccess).with_kwargs(pool=CachePools.BEST_EFFORT)
```

## API
The bindings and behavior of the injector are controlled with the `Configuration` class.

### Initialize
To initialize the injector, a config is needed:
```python:
configuration = Configuration()
injector = Injector(configuration)
```

### Global bindings:
Bind the given class to the configured type resolver.

```python:
configuration.bind(Foo).globally()
```

This returns a `TypeResolver[Foo]`, that can be further configured. See `TypeResolver` api below.

### Scoped bindings:
Bind the given class to the configured type resolver only for the given parent class.

```python:

class Bar:
    def __init__(self, foo: Foo) -> None:
        ...

class Baz:
    def __init__(self, foo: Foo) -> None:
        ...

configuration.bind(Foo).for_parent(Bar)
```
This binding will only take effect for `Bar`. `Baz` will continue to see the default instantiation for `Foo`.

This returns a `TypeResolver[Foo]`, that can be further configured. See `TypeResolver` api below.

### Type resolver
Once created the bind and set the scope (`bind(Foo).globally()` or `bind(Foo).for_parent(Bar)`) you will get a `TypeResolver` that allows to configure how the bound value will be resolved.

* `to_instance(instance)`: Binds to an specific instance. Useful for wiring globals into DI or when building the object is complicated and you prefer to control that.
* `to_class(Bar)`: Binds to a class. Useful to inject a comparible subclass, the concrete implementation of an abstract class or a class that implements a Protocol.
* `to_constructor(constructor_fn)`: The function will build the object. Note that the function will be also injected, so the function might use a configuration class and the injector will provide it. Useful for objects complicated to build.
* No `to_*` invoked: Binds to the class itself (it will use its `__init__()` as constructor). This makes sense, for example to control the behavior of singletons (See cache and singletons section), to revert a global bind to the original for a parent class, or for the test-specific configurations that expect explicit bindings.

Additionally, for the default and `to_constructor` resolutions, this extra configuration can be set:
* `with_kwargs(foo=bar)`: Overrides the value of given param in the constructor.
* `with_arg_types(foo=Foo)`: Overrides the type that will be used for the param. Similar to `for_parent(...).to_class(...)` that can also override the class, but it can work when you have two args with the same type (imagine `Processor(in: Queue, out: Queue)`) and will also work for constructor functions.

### Cache and singletons
The injector will cache **ALL** types, both specifically bound and those injected using the default. This means that **ALL classes will be singletons**.

Note that when binding the same class globally / for specific parents, obviously each one will get a different singleton.

While this is generally the preferred choice, there can be situations where this is not desired.

You can avoid this by:
a) Binding for each parent class:
```python:
configuration.bind(Container).for_parent(Foo)
configuration.bind(Container).for_parent(Bar)
```
With this, `Foo` and `Bar` will use different containers. Note that still all `Foo` instantiated with the injector will be the same instance, and will obviously also have the same `Container`.

b) Make your code build the instances by default:
```python: 
class Foo:
    def __init__(self, container: Optional[Foo] = None) -> None:
        self.container = container or Container()
```
The code is still testable, `Container` can be injected for tests (the test injector can even bind `Optional[Container]` to a mock), but it is clear that each class will use a different `Container` instance by default.

## Default values
The injector recognizes default values and will use them unless there is an specific binding for the class.

For example:
```python:
class A:
    def __init__(self, foo: str="foo") -> None:
        ...
```

Will just work as expected, and the default value will be used. If you would want to override this value with the injector, you will need to use:

```python:
configuration.bind(A).globally().with_kwargs(foo="override")
```

While having static instances as default values is not recommended, this will also work:

```python:
default_foo = Foo("static")
class A:
    def __init__(self, foo: Foo = default_foo) -> None:
        ...
```

By default, `A` will receive `default_foo` as parameter. To override, you will do:

```python:
configuration.bind(Foo).globally().to_instance(override_foo)
# Or just for A:
configuration.bind(Foo).for_parent(A).to_instance(override_foo)
```

## Optional and Unions
The injector will refuse to build `Optional` and `Union` types by default, as it doesn't know what of the multiple choices to injects.

For `Optional[Foo]` and `Union[Foo, Bar]` types binding just `Foo` will not work. You can `bind(Optional[Foo])` and `bind[Foo, Bar]` and map them normally to a instance, concrete class or constructor.

## Best practices
* Keep configuration settings out of your application-level classes' constructors, so more of them can be built automatically. You can use a `ConfigurationProvider` dependency to provide configuration settings to your app.
* Avoid Union for dependencies when possible, use Protocol or Abstract as they should have compatible apis.
* If is ok to have low-level dependencies (data access, ...) with configuration or as abstract / Protocol classes that force injecting a concrete instance and/or configuration.
* Build a production entry point separated from test and local envs, that is the only one that configures the injector for production.
* Prepare a shared test-specific injector. Specially for integration tests so the plumbing of configuring dependencies for test environment is only done once.
  
## Testing-specific configurations
This injector will try to build all classes not bound, and as long as no scalar or primitive values are needed, it will traverse the dependency tree and build all objects.

For testing it might be interesting to mock by default or fail if a dependency is needed and not specifically bound in the test, so two configuration sub-classes are provided:

### ErrorOnNotExplicitConfiguration
Will throw `ErrorOnNotExplicitConfiguration` for any class not bound.

```python:
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
```

### MockOnNotExplicitConfiguration
Will mock any classes not specifically bound.

```python:
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
```

# Develop

Install `uv` (https://docs.astral.sh/uv/) and run:

```bash:
uv venv
uv run --with nox nox
```
