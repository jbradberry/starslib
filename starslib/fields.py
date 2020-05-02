from bisect import bisect

from .exceptions import ValidationError, ParseError

import six


BITWIDTH_CHOICES = ((0, 0), (1, 8), (2, 16), (3, 32))
ftypes = ('xy', 'x', 'hst', 'm', 'h', 'r')

def filetypes(*args):
    def ftype_check(s):
        return s.file.type in args
    return ftype_check


class Value(object):
    """An accessor that is attached to a Struct class as a proxy for a Field.

    This is a descriptor (getter/setter) that gets an instance
    attached as a class attribute to a Struct subclass at
    class-construction time, that has access to the Field that it
    represents. It calls its Field's methods to do the cleaning,
    validation, and updating of related fields.
    """

    def __init__(self, field):
        self.field = field

    def __get__(self, obj, type=None):
        if obj is None:
            raise AttributeError
        # A field is dynamic if another field has a reference to it.
        if self.field.dynamic:
            field, t = self.field.dynamic
            # Fields that are involved in a relationship provide a method
            # beginning with 'value_' to report the current value of the
            # desired attribute.
            update = getattr(field, 'value_'+t)(obj)
            self.field.set_value(obj, update)
        # A field has references if some attribute of it, such as bitwidth,
        # is stored in another field.
        for ref, t in self.field.references:
            update = getattr(self.field, 'value_'+t)(obj)
            ref.set_value(obj, update, True)
        value = obj.__dict__[self.field.name]
        self.field.validate(obj, value)
        return value

    def __set__(self, obj, value):
        value = self.field.clean(value)
        obj.__dict__[self.field.name] = value
        for ref, t in self.field.references:
            ref.set_value(obj, getattr(self.field, 'value_'+t)(obj), True)
        self.field.validate(obj, value)


def make_contrib(new_cls, func=None):
    def contribute_to_class(self, cls, name):
        if func:
            func(self, cls, name)
        else:
            super(new_cls, self).contribute_to_class(cls, name)
        setattr(cls, self.name, Value(self))

    return contribute_to_class


class FieldBase(type):
    def __new__(cls, names, bases, attrs):
        new_cls = super(FieldBase, cls).__new__(cls, names, bases, attrs)
        new_cls.contribute_to_class = make_contrib(
            new_cls, attrs.get('contribute_to_class'))
        return new_cls


class Field(six.with_metaclass(FieldBase, object)):
    """A data member on a Struct.

    bitwidth: Specifies the number of bits to be consumed to populate
    this field.

    value: if non-None, the value that this field must always contain

    max: the maximum integer value this field may contain

    choices: specify the enumerated values this field may contain

    option: this field is optional, and is only present when option
    evaluates as True
    """

    counter = 0

    def __init__(self, bitwidth=16, value=None, max=None,
                 choices=None, option=None, **kwargs):
        """
        """

        self._counter = Field.counter
        Field.counter += 1

        self.references = []
        self.dynamic = None

        self._bitwidth = bitwidth
        if callable(bitwidth):
            self.bitwidth = self._callable_bitwidth
        elif isinstance(bitwidth, six.string_types):
            self.references.append([bitwidth, 'bitwidth'])
            self.bitwidth = self._ref_bitwidth
        else:
            self.bitwidth = self._const_bitwidth

        self.value = value
        self.max = max
        self.choices = choices
        self.option = option

    def _callable_bitwidth(self, obj):
        return self._bitwidth(obj)

    def _ref_bitwidth(self, obj):
        return self._bitwidth.get_value(obj)

    def _const_bitwidth(self, obj):
        return self._bitwidth

    def __lt__(self, other):
        return self._counter < other._counter

    def __le__(self, other):
        return self._counter <= other._counter

    def __eq__(self, other):
        return self._counter == other._counter

    def __ne__(self, other):
        return self._counter != other._counter

    def __gt__(self, other):
        return self._counter > other._counter

    def __ge__(self, other):
        return self._counter >= other._counter

    def contribute_to_class(self, cls, name):
        self.name = name
        self.struct = cls
        cls.fields.insert(bisect(cls.fields, self), self)

    def _parse_vars(self, obj, seq, vars):
        vars.bitwidth = self.bitwidth(obj)

    def _pre_parse(self, obj, seq, vars):
        if vars.bitwidth and obj.byte >= len(seq):
            raise ParseError("%s.%s: %s" % (self.struct.__name__,
                                            self.name, seq))

    def _parse(self, obj, seq, vars):
        pass

    def _post_parse(self, obj, seq, vars):
        return vars.result

    def parse(self, obj, seq):
        vars = obj._vars
        self._parse_vars(obj, seq, vars)
        self._pre_parse(obj, seq, vars)
        self._parse(obj, seq, vars)
        return self._post_parse(obj, seq, vars)

    def _deparse_vars(self, obj, vars):
        vars.bitwidth = self.bitwidth(obj)
        vars.value = getattr(obj, self.name)

    def _pre_deparse(self, obj, vars):
        pass

    def _deparse(self, obj, vars):
        pass

    def _post_deparse(self, obj, vars):
        return vars.result

    def deparse(self, obj):
        vars = obj._vars
        self._deparse_vars(obj, vars)
        self._pre_deparse(obj, vars)
        self._deparse(obj, vars)
        return self._post_deparse(obj, vars)

    def clean(self, value):
        return value

    def skip(self, obj):
        if self.option and not self.option(obj):
            return True
        if self.bitwidth(obj) is None:
            return True
        return False

    def validate(self, obj, value):
        bitwidth = self.bitwidth(obj)
        if value is None:
            if self.option:
                if self.option(obj):
                    raise ValidationError
                return True
            if bitwidth is not None and bitwidth > 0:
                raise ValidationError("bitwidth: %s" % bitwidth)
            return True
        if bitwidth is None and value is not None:
            raise ValidationError
        if self.value is not None and value != self.value:
            raise ValidationError("%s: %s != %s" % (self.name, value,
                                                    self.value))

    def get_value(self, obj):
        value = obj.__dict__[self.name]
        if value is None:
            return None
        if self.choices is not None:
            value = dict(self.choices)[value]
        return value

    def set_value(self, obj, value, side_effects=False):
        if value is not None and self.choices is not None:
            value = min((v, k) for k, v in self.choices if value <= v)[1]

        if side_effects:
            setattr(obj, self.name, value)
        else:
            obj.__dict__[self.name] = self.clean(value)


class Int(Field):
    def _parse(self, obj, seq, vars):
        vars.result = 0
        try:
            acc_bit = 0
            while vars.bitwidth > 0:
                if vars.bitwidth >= 8-obj.bit:
                    vars.result += (seq[obj.byte] >> obj.bit) << acc_bit
                    obj.byte += 1
                    acc_bit += 8-obj.bit
                    vars.bitwidth -= 8-obj.bit
                    obj.bit = 0
                else:
                    vars.result += ((seq[obj.byte] >> obj.bit) &
                                    (2**vars.bitwidth - 1)) << acc_bit
                    obj.bit += vars.bitwidth
                    vars.bitwidth = 0
        except IndexError:
            raise ParseError("%s %s: %s > %s" % (self.struct.__name__,
                                                 seq, obj.byte, len(seq)))

    def _deparse(self, obj, vars):
        result = []
        value = vars.value << obj.bit | obj.prev
        vars.bitwidth += obj.bit
        while vars.bitwidth >= 8:
            value, tmp = value >> 8, value & 0xff
            vars.bitwidth -= 8
            result.append(tmp)
        obj.prev, obj.bit = value, vars.bitwidth
        vars.result = result

    def clean(self, value):
        if value is None:
            return None
        return int(value)

    def validate(self, obj, value):
        if super(Int, self).validate(obj, value):
            return True
        if self.max is not None and value > self.max:
            raise ValidationError("%s: %s > %s" % (self.name, value, self.max))
        if not 0 <= value < 2**self.bitwidth(obj):
            raise ValidationError

    def value_bitwidth(self, obj):
        value = obj.__dict__[self.name]
        if value is None:
            return None
        return 0 if value == 0 else len(bin(value)) - 2


class Bool(Int):
    def __init__(self, **kwargs):
        kwargs.update(bitwidth=1, choices=None)
        super(Bool, self).__init__(**kwargs)

    def _post_parse(self, obj, seq, vars):
        return bool(vars.result)

    def clean(self, value):
        if value is None:
            return None
        return bool(value)


class Sequence(Field):
    """A field that stores some dynamic sequence of values.

    head: denotes some number of bits at the beginning of the
    bitstream that stores the length information. Must be a multiple
    of 8.

    length: an externally specified number of items for the
    sequence. May be a fixed number, a callable, or a string which
    encodes a reference to another field which stores the length. Only
    specified if head is None. If length is then also None, this
    generally means to consume all remaining bytes in the sequence.

    bitwidth: the number of bits each element of the sequence consumes
    """

    def __init__(self, head=None, length=None, bitwidth=8, **kwargs):
        kwargs.update(bitwidth=bitwidth)
        super(Sequence, self).__init__(**kwargs)
        self.head = head
        self._length = length
        if length is None:
            if head is not None:
                self.length = self._head_length
            else:
                self.length = self._remainder_length
        elif callable(length):
            self.length = self._callable_length
        elif isinstance(length, six.string_types):
            self.references.append([length, 'length'])
            self.length = self._ref_length
        else:
            self.length = self._const_length

    def _head_length(self, obj, seq=None):
        if seq is None:
            return None
        return sum(x << (8 * n) for n, x in
                   enumerate(seq[obj.byte:obj.byte+self.head//8]))

    def _remainder_length(self, obj, seq=None):
        if seq is None:
            return None
        return (len(seq) - obj.byte) // (self.bitwidth(obj) // 8)

    def _callable_length(self, obj, seq=None):
        return self._length(obj)

    def _ref_length(self, obj, seq=None):
        return self._length.get_value(obj)

    def _const_length(self, obj, seq=None):
        return self._length

    def _parse_vars(self, obj, seq, vars):
        super(Sequence, self)._parse_vars(obj, seq, vars)
        vars.length = self.length(obj, seq)
        if self._length is None and self.head is not None:
            obj.byte += self.head // 8

    def _pre_parse(self, obj, seq, vars):
        if vars.bitwidth % 8 != 0:
            raise ParseError
        if obj.bit != 0:
            raise ParseError

        if vars.length * vars.bitwidth//8 > len(seq) - obj.byte:
            raise ParseError("byte: %s, seq: %s" % (obj.byte, seq))

    def _parse(self, obj, seq, vars):
        result = seq[obj.byte:obj.byte + vars.length * vars.bitwidth//8]
        result = list(zip(*(iter(result),) * (vars.bitwidth//8)))
        result = [sum(x << (8 * n) for n, x in enumerate(b)) for b in result]
        obj.byte += vars.length * vars.bitwidth//8
        vars.result = result

    def _pre_deparse(self, obj, vars):
        if obj.prev != 0 or obj.bit != 0:
            raise ParseError
        if vars.bitwidth % 8 != 0:
            raise ParseError

    def _deparse(self, obj, vars):
        vars.result = [x >> (8 * n) & 0xff for x in vars.value
                       for n in range(vars.bitwidth//8)]

    def _post_deparse(self, obj, vars):
        L = len(vars.value)
        vars.L = L
        if self._length is None and self.head is not None:
            head = [L >> (8 * n) & 0xff for n in range(self.head//8)]
            vars.result = head + vars.result
        return vars.result

    def validate(self, obj, value):
        if super(Sequence, self).validate(obj, value):
            return True
        length = self.length(obj)
        if self._length is None:
            if self.head is not None:
                if not 0 <= len(value) < 2**self.head:
                    raise ValidationError
            else:
                if not 0 <= len(value) < 1024:
                    raise ValidationError
        # don't worry about the basestring case; the chained setattr
        # will get it.
        elif not isinstance(self._length, six.string_types):
            if len(value) != length:
                raise ValidationError

    def value_length(self, obj):
        value = obj.__dict__[self.name]
        return len(value) if value is not None else None


class Str(Sequence):
    def __init__(self, length=None, head=None, **kwargs):
        kwargs.update(bitwidth=8, length=length, head=head)
        super(Str, self).__init__(**kwargs)

    def _post_parse(self, obj, seq, vars):
        return ''.join(map(chr, vars.result))

    def _pre_deparse(self, obj, vars):
        vars.value = list(map(ord, vars.value))

    def validate(self, obj, value):
        if super(Str, self).validate(obj, value):
            return True
        if not isinstance(value, six.string_types):
            raise ValidationError


class CStr(Str):
    top = " aehilnorstbcdfgjkmpquvwxyz+-,!.?:;'*%$"

    def __init__(self, head=8, **kwargs):
        kwargs.update(head=head)
        super(CStr, self).__init__(**kwargs)

    def _pre_parse(self, obj, seq, vars):
        super(CStr, self)._pre_parse(obj, seq, vars)
        if vars.length == 0:
            obj.byte += 1

    def _post_parse(self, obj, seq, vars):
        return self.decompress(vars.result)

    def _pre_deparse(self, obj, vars):
        vars.value = self.compress(vars.value)

    def _post_deparse(self, obj, vars):
        vars.result = super(CStr, self)._post_deparse(obj, vars)
        if self._length is None:
            if vars.L == 0:
                vars.result.append(0)
        return vars.result

    def value_length(self, obj):
        value = self.compress(obj.__dict__[self.name])
        return len(value) if value is not None else None

    def decompress(self, lst):
        # break lst up into a sequence of 4-bit nibbles
        tmp = ((x >> i) & 0xf for x in lst for i in (4, 0))
        result = []
        for x in tmp:
            if 0x0 <= x <= 0xA:
                C = self.top[x]
            elif 0xB <= x <= 0xE:
                x = ((x - 0xB) << 4) + next(tmp)
                if x < 0x1A:
                    C = chr(x + 0x41)
                elif x < 0x24:
                    C = chr(x + 0x16)
                else:
                    C = self.top[x - 0x19]
            elif x == 0xF:
                try:
                    C = chr(next(tmp) + (next(tmp) << 4))
                except StopIteration:
                    break
            result.append(C)
        return ''.join(result)

    def compress(self, S):
        result = []
        for c in S:
            if c in self.top[:11]:
                result.append(self.top.index(c))
            elif c in self.top[11:]:
                tmp = self.top.index(c) + 0x19
                result.extend(((tmp >> 4) + 0xB, tmp & 0xF))
            else:
                tmp = ord(c)
                if 0x41 <= tmp < 0x5B:
                    tmp -= 0x41
                    result.extend(((tmp >> 4) + 0xB, tmp & 0xF))
                elif 0x30 <= tmp < 0x3A:
                    tmp -= 0x16
                    result.extend(((tmp >> 4) + 0xB, tmp & 0xF))
                else:
                    result.extend((0xF, tmp & 0xF, tmp >> 4))
        if len(result) % 2 != 0:
            result.append(0xF)
        return [(result[i] << 4)+result[i+1] for i in range(0, len(result), 2)]


class Array(Sequence):
    def __init__(self, head=8, length=None, **kwargs):
        kwargs.update(head=head, length=length)
        super(Array, self).__init__(**kwargs)

    def validate(self, obj, value):
        if super(Array, self).validate(obj, value):
            return True
        bitwidth = self.bitwidth(obj)
        if not all(0 <= x < 2**bitwidth for x in value):
            raise ValidationError

    def value_bitwidth(self, obj):
        value = obj.__dict__[self.name]
        if value is None:
            return None
        if not value:
            return 0
        return max(0 if x == 0 else len(bin(x)) - 2 for x in value)


class ObjArray(Array):
    def __init__(self, **kwargs):
        super(ObjArray, self).__init__(**kwargs)
        self.bitwidths = self.bitwidth
        self.bitwidth = self._inner_bitwidths

    def _inner_bitwidths(self, obj):
        return sum(bw for name, bw in self.bitwidths(obj))

    def _parse_vars(self, obj, seq, vars):
        super(ObjArray, self)._parse_vars(obj, seq, vars)
        vars.bitwidths = self.bitwidths(obj)

    def _post_parse(self, obj, seq, vars):
        bitwidths = vars.bitwidths
        bw = [(b[0], b[1], sum([0] + [v for _, v in bitwidths][:i]))
              for i, b in enumerate(bitwidths)]
        return [dict((k, (x >> o) & (2**b - 1)) for k, b, o in bw)
                for x in vars.result]

    def _deparse_vars(self, obj, vars):
        super(ObjArray, self)._deparse_vars(obj, vars)
        vars.bitwidths = self.bitwidths(obj)

    def _pre_deparse(self, obj, vars):
        Sequence._pre_deparse(self, obj, vars)
        bitwidths = vars.bitwidths
        bw = [(b[0], b[1], sum([0] + [v for _, v in bitwidths][:i]))
              for i, b in enumerate(bitwidths)]
        vars.value = [sum(x[k] << o for k, b, o in bw) for x in vars.value]

    def validate(self, obj, value):
        if Sequence.validate(self, obj, value):
            return True
        bitwidths = self.bitwidths(obj)
        if not all(all(0 <= x[k] < 2**v for k, v in bitwidths) for x in value):
            raise ValidationError
        if any(set(x.keys()) - set(b[0] for b in bitwidths) for x in value):
            raise ValidationError
