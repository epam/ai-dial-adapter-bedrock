def not_implemented(func):
    setattr(func, "_not_implemented", True)
    return func


def is_implemented(method):
    return not getattr(method, "_not_implemented", False)
