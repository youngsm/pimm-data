"""Registry utility for string-based class lookups.

Vendored (self-contained) version of the mmcv/pimm Registry so pimm-data
does not depend on pimm. Compatible interface for configs that contain
``dict(type="ClassName", ...)`` and build via ``registry.build(cfg)``.
"""

import inspect
from collections import abc


def _is_seq_of(seq, expected_type, seq_type=None):
    exp_seq_type = abc.Sequence if seq_type is None else seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    return all(isinstance(item, expected_type) for item in seq)


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from configs dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f"but got {cfg}\n{default_args}"
            )
    if not isinstance(registry, Registry):
        raise TypeError(f"registry must be a Registry object, got {type(registry)}")
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(f"default_args must be a dict or None, got {type(default_args)}")

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")
    try:
        return obj_cls(**args)
    except Exception as e:
        raise type(e)(f"{obj_cls.__name__}: {e}")


class Registry:
    """A registry to map strings to classes.

    Registered objects can be built from configs::

        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        ...     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))
    """

    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope() if scope is None else scope

        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"(name={self._name}, items={self._module_dict})")

    @staticmethod
    def infer_scope():
        module = inspect.getmodule(inspect.stack()[2][0])
        if module is None:
            return "pimm_data"
        return module.__name__.split(".")[0]

    @staticmethod
    def split_scope_key(key):
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1:]
        return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def get(self, key):
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            if scope in self._children:
                return self._children[scope].get(real_key)
            parent = self.parent
            while parent is not None and parent.parent is not None:
                parent = parent.parent
            if parent is not None:
                return parent.get(key)
        return None

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert registry.scope not in self.children, (
            f"scope {registry.scope} exists in {self.name} registry")
        self.children[registry.scope] = registry

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError(f"module must be a class, got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered in {self.name}")
            self._module_dict[name] = module_class

    def register_module(self, name=None, force=False, module=None):
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, got {type(force)}")
        if isinstance(name, type):
            # mmcv's pre-1.0 form, `register_module(SomeClass)`. It was vendored
            # in with a DeprecationWarning and never used here — no call site in
            # this repo passes a class positionally. Rejected outright rather
            # than silently accepted, so the one supported spelling is the one
            # in every config.
            raise TypeError(
                f"register_module() got the class {name.__name__} as its first "
                f"positional argument. The signature is "
                f"register_module(name=None, force=False, module=None); "
                f"decorate with @{self.name.upper()}.register_module() instead.")

        if not (name is None or isinstance(name, str) or _is_seq_of(name, str)):
            raise TypeError(
                "name must be None, str, or sequence of str, "
                f"got {type(name)}")

        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register
