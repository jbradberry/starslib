from bisect import bisect
import numbers


class ValidationError(Exception):
    pass


class Value(object):
    def __init__(self, field):
        self.field = field

    def __get__(self, obj, type=None):
        if obj is None:
            raise AttributeError
        if self.field.option is not None and not self.field.option():
            return None
        return obj.__dict__[self.field.name]

    def __set__(self, obj, value):
        self.field.validate(value)
        value = self.field.clean(value)
        obj.__dict__[self.field.name] = value


def make_contrib(func=None):
    def contribute_to_class(self, cls, name):
        if func:
            func(self, cls, name)
        else:
            super(self.__class__, self).contribute_to_class(cls, name)
        setattr(cls, self.name, Value(self))

    return contribute_to_class


class FieldBase(type):
    def __new__(cls, names, bases, attrs):
        new_cls = super(FieldBase, cls).__new__(cls, names, bases, attrs)
        new_cls.contribute_to_class = make_contrib(
            attrs.get('contribute_to_class'))
        return new_cls


class Field(object):
    __metaclass__ = FieldBase

    counter = 0
    def __init__(self, size=16, **kwargs):
        self._counter = Field.counter
        Field.counter += 1
        self.size = size

    def __cmp__(self, other):
        return cmp(self._counter, other._counter)

    def contribute_to_class(self, cls, name):
        self.name = name
        self.struct = cls
        cls.fields.insert(bisect(cls.fields, self), self)

    def parse(self, seq, byte, bit):
        if byte >= len(seq) and not self.append:
            raise ValueError
        size = self.size

    def clean(self, value):
        return value

    def validate(self, value):
        if value is None and self.option is None:
            raise ValidationError
        if self.value is not None and value != self.value:
            raise ValidationError


class Int(Field):
    def validate(self, value):
        super(Int, self).validate(value)
        if not isinstance(value, number.Integral):
            raise ValidationError
        if hasattr(self, 'max') and value > self.max:
            raise ValidationError
        if not 0 <= value < 2**self.size:
            raise ValidationError


class Bool(Int):
    def __init__(self, **kwargs):
        kwargs.update(size=1)
        super(Bool, self).__init__(**kwargs)

    def clean(self, value):
        return bool(value)


class Str(Field):
    def validate(self, value):
        super(Str, self).validate(value)
        if not isinstance(value, basestring):
            raise ValidationError
        if len(value) != self.size:
            raise ValidationError


class CStr(Field):
    def validate(self, value):
        super(CStr, self).validate(value)
        if not isinstance(value, basestring):
            raise ValidationError


class StructBase(type):
    def __new__(self, cls, name, bases, attrs):
        super_new = super(StructBase, cls).__new__
        parents = [b for b in bases if isinstance(b, StructBase)]
        if not parents:
            return super_new(cls, name, bases, attrs)

        module = attrs.pop('__module__')
        new_class = super_new(cls, name, bases, {'__module__': module})

        for obj_name, obj in attrs.iteritems():
            new_class.add_to_class(obj_name, obj)

    def add_to_class(cls, name, value):
        if hasattr(value, 'contribute_to_class'):
            value.contribute_to_class(cls, name)
        else:
            setattr(cls, name, value)


class Struct(object):
    __metaclass__ = StructBase

    @property
    def bytes(self):
        return ()

    @bytes.setter
    def bytes(self, seq):
        byte, bit = 2, 0 # ignore the 16-bits of type/size info
        for field in fields:
            byte, bit = field.parse(seq, byte, bit)
        if byte != len(seq) or bit != 0:
            raise ValueError


class StarsFile(object):
    prime = (3,   5,   7,   11,  13,  17,  19,  23,
             29,  31,  37,  41,  43,  47,  53,  59,
             61,  67,  71,  73,  79,  83,  89,  97,
             101, 103, 107, 109, 113, 127, 131, 137,
             139, 149, 151, 157, 163, 167, 173, 179,
             181, 191, 193, 197, 199, 211, 223, 227,
             229, 233, 239, 241, 251, 257, 263, 279,
             271, 277, 281, 283, 293, 307, 311, 313,
             317, 331, 337, 347, 349, 353, 359, 367,
             373, 379, 383, 389, 397, 401, 409, 419,
             421, 431, 433, 439, 443, 449, 457, 461,
             463, 467, 479, 487, 491, 499, 503, 509,
             521, 523, 541, 547, 557, 563, 569, 571,
             577, 587, 593, 599, 601, 607, 613, 617,
             619, 631, 641, 643, 647, 653, 659, 661,
             673, 677, 683, 691, 701, 709, 719, 727)

    def __init__(self):
        self.hi, self.lo = 0, 0

    def prng_init(self, flag, player, turn, salt, uid):
        prime = StarsFile.prime
        i, j = (salt>>5) & 0x1f, salt & 0x1f
        if salt < (1<<10):
            i += (1<<5)
        else:
            j += (1<<5)
        self.hi, self.lo = prime[i], prime[j]

        seed = ((player%4)+1) * ((uid%4)+1) * ((turn%4)+1) + flag
        for i in xrange(seed):
            burn = self.prng()

    def prng(self):
        self.lo = (0x7fffffab * int(self.lo/-53668) + 40014 * self.lo)%(1<<32)
        if self.lo >= (1<<31): self.lo += 0x7fffffab - (1<<32)
        self.hi = (0x7fffff07 * int(self.hi/-52774) + 40692 * self.hi)%(1<<32)
        if self.hi >= (1<<31): self.hi += 0x7fffff07 - (1<<32)
        return (self.lo - self.hi) % (1<<32)


class Type0(Struct):
    """ End of file """
    info = Int(option=filetype('m', 'hst', 'xy'))


class Type8(Struct):
    """ Beginning of file """
    encrypted = False

    magic = Str(32, value="J3J3")
    game_id = Int(32)
    file_ver = Int()
    turn = Int()
    player = Int(5)
    salt = Int(11)
    filetype = Int(8)
    submitted = Bool()
    in_use = Bool()
    multi_turn = Bool()
    gameover = Bool()
    shareware = Bool()
    unused = Int(3, value=0)


class Type6(Struct):
    """ Race data """
    pass


class Type45(Struct):
    """ Score data """
    player = Int(5)
    unknown1 = Bool(value=True) # rare False?
    f_owns_planets = Bool()
    f_attains_tech = Bool()
    f_exceeds_score = Bool()
    f_exceeds_2nd = Bool()
    f_production = Bool()
    f_cap_ships = Bool()
    f_high_score = Bool()
    unknown2 = Bool(value=False) # rare True?
    f_declared_winner = Bool()
    unknown3 = Bool()
    year = Int()
    score = Int(32)
    resources = Int(32)
    planets = Int()
    starbases = Int()
    unarmed_ships = Int()
    escort_ships = Int()
    capital_ships = Int()
    tech_levels = Int()


class Type20(Struct):
    """ Waypoint - Server """
    x = Int()
    y = Int()
    planet_id = Int()
    unknown1 = Int(4)
    warp = Int(4)
    unknown2 = Int(8)


class Type30(Struct):
    """ Battle plans """
    u1 = Int(8)
    u2 = Int(8)
    u3 = Int(8)
    u4 = Int(8)
    name = CStr()


class Type40(Struct):
    """ In-game messages """
    unknown1 = Int(32)
    sender_id = Int(8)
    unknown2 = Int(40)
    text = CStr(16)


class Type17(Struct):
    """ Alien fleets """
    ship_id = Int(8)
    unknown1 = Int(8)
    player_id = Int(8)
    unknown2 = Int(40)
    x = Int()
    y = Int()
    unknown3 = Int(56)
    mass = Int(32)

class Type43(Struct):
    """ Mass packets """
    # optional sizes: 2, 4, and 18
    unknown1 = Int()
    x = Int()
    y = Int()
    unknown2 = Int()
    mass_ir = Int()
    mass_bo = Int()
    mass_ge = Int()
    unknown3 = Int(32)


class Type3(Struct):
    """ Delete waypoint """
    fleet_id = Int(9)
    unknown1 = Int(7)
    sequence_num = Int(8)
    unknown2 = Int(8)
