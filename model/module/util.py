class Pack(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            return super(Pack, self).__getattr__(name)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack