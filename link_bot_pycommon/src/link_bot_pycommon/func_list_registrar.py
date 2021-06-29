class FuncListRegistrar:
    def __init__(self):
        self.funcs = []

    def __call__(self, f):
        self.funcs.append(f)
        return f

    def __iter__(self):
        for f in self.funcs:
            yield f
