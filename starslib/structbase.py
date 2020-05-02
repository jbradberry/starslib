import six

from .exceptions import ValidationError


class StructBase(type):
    def __new__(cls, name, bases, attrs):
        super_new = super(StructBase, cls).__new__
        parents = [b for b in bases if isinstance(b, StructBase)]
        if not parents:
            return super_new(cls, name, bases, attrs)

        new_attrs = {}
        new_attrs['__module__'] = attrs.pop('__module__')
        if '__classcell__' in attrs:
            new_attrs['__classcell__'] = attrs['__classcell__']
        new_class = super_new(cls, name, bases, new_attrs)

        new_class.add_to_class('fields', [])
        for obj_name, obj in attrs.items():
            new_class.add_to_class(obj_name, obj)

        by_name = dict((field.name, field) for field in new_class.fields)
        for field in new_class.fields:
            for ref in field.references:
                new_field = by_name[ref[0]]
                new_field.dynamic = (field, ref[1])
                ref[0] = new_field
                setattr(field, '_'+ref[1], new_field)

        if 'type' in attrs:
            new_class._registry[attrs['type']] = new_class

        return new_class

    def add_to_class(cls, name, value):
        if hasattr(value, 'contribute_to_class'):
            value.contribute_to_class(cls, name)
        else:
            setattr(cls, name, value)


class Vars(object):
    pass


@six.python_2_unicode_compatible
class Struct(six.with_metaclass(StructBase, object)):
    _registry = {}

    encrypted = True

    def __init__(self, sfile):
        self.file = sfile
        self._vars = Vars()
        self._vars._seq = sfile.counts.get(self.type, 0)

    def __str__(self):
        return "{%s}" % (', '.join("%s: %r" % (f.name,
                                               getattr(self, f.name, None))
                                   for f in self.fields
                                   if getattr(self, f.name) is not None),)

    @property
    def bytes(self):
        seq, self.prev, self.bit = [], 0, 0
        for field in self.fields:
            if not field.skip(self):
                seq.extend(field.deparse(self))
        if self.bit != 0 or self.prev != 0:
            raise ValidationError
        return tuple(seq)

    @bytes.setter
    def bytes(self, seq):
        self.byte, self.bit = 0, 0
        for field in self.fields:
            result = None if field.skip(self) else field.parse(self, seq)
            setattr(self, field.name, result)
        if self.byte != len(seq) or self.bit != 0:
            raise ValidationError("%s %s (%s %s) %s" % (self.__class__.__name__,
                                                        len(seq), self.byte,
                                                        self.bit, seq))

    def adjust(self):
        return
