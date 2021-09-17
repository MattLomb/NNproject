class AttributeDict(dict):
    """
    Dict where values can be accessed using attribute syntax.
    Same as "EasyDict" in the NVIDIA stylegan git repository.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __getstate__(self):
        return dict(**self)

    def __setstate__(self, state):
        self.update(**state)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join('{}={}'.format(key, value) for key, value in self.items())
        )

    @classmethod
    def convert_dict_recursive(cls, obj):
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                obj[key] = cls.convert_dict_recursive(obj[key])
            if not isinstance(obj, cls):
                return cls(**obj)
        return obj