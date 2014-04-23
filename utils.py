def memoize(f):
    """
    Memoization decorator for functions taking one or more arguments.
    From http://code.activestate.com/recipes/578231-probably-the-
    fastest-memoization-decorator-in-the-/
    """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret
    return memodict(f)