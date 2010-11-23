from __future__ import division, with_statement
from bisect import bisect
import sys, struct
import numbers


class ValidationError(Exception):
    pass


class Value(object):
    def __init__(self, field):
        self.field = field

    def __get__(self, obj, type=None):
        if obj is None:
            raise AttributeError
        return obj.__dict__[self.field.name]

    def __set__(self, obj, value):
        if not self.field.skip(obj, value):
            self.field.validate(obj, value)
        value = self.field.clean(value)
        obj.__dict__[self.field.name] = value


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
        new_cls.contribute_to_class = make_contrib(new_cls,
            attrs.get('contribute_to_class'))
        return new_cls


class Field(object):
    __metaclass__ = FieldBase

    counter = 0
    def __init__(self, size=16, value=None, max=None, option=None, **kwargs):
        self._counter = Field.counter
        Field.counter += 1
        self.size = size
        self.value = value
        self.max = max
        self.option = option

    def __cmp__(self, other):
        return cmp(self._counter, other._counter)

    def contribute_to_class(self, cls, name):
        self.name = name
        self.struct = cls
        cls.fields.insert(bisect(cls.fields, self), self)

    def parse(self, obj, seq):
        if self.skip(obj):
            setattr(obj, self.name, None)
            return
        if obj.byte >= len(seq):
            raise ValidationError("%s.%s: %s" % (self.struct.__name__, self.name, seq))
        size = self.size
        if isinstance(size, basestring):
            size = getattr(obj, size)
            size = size if size < 3 else 4
            size *= 8
        elif callable(size):
            size = size(obj)
        result = 0
        try:
            acc_bit = 0
            while size > 0:
                if size >= 8-obj.bit:
                    result += (seq[obj.byte]>>obj.bit) << acc_bit
                    obj.byte += 1
                    acc_bit += 8-obj.bit
                    size -= 8-obj.bit
                    obj.bit = 0
                else:
                    result += ((seq[obj.byte]>>obj.bit)&(2**size-1)) << acc_bit
                    obj.bit += size
                    size = 0
        except IndexError, e:
            raise ValidationError("%s %s: %s > %s" % (self.struct.__name__, seq, obj.byte, len(seq)))
        setattr(obj, self.name, result)

    def deparse(self, obj):
        extend = []
        if self.skip(obj):
            return extend
        value = getattr(obj, self.name) << obj.bit | obj.prev
        size = self.size
        if isinstance(size, basestring):
            size = getattr(obj, size)
            size = size if size < 3 else 4
            size *= 8
        size += obj.bit
        while size >= 8:
            value, tmp, size = value >> 8, value & 0xff, size - 8
            extend.append(tmp)
        obj.prev, obj.bit = value, size
        return extend

    def clean(self, value):
        return value

    def skip(self, obj, value=None):
        if value is None:
            if self.option and not self.option(obj):
                return True
            if isinstance(self.size, basestring):
                size = getattr(obj, self.size)
                if size is None or size == 0:
                    return True
        return False

    def validate(self, obj, value):
        if value is None:
            if self.option:
                if self.option(obj):
                    raise ValidationError
                return
            if isinstance(self.size, basestring):
                if getattr(obj, self.size) != 0:
                    raise ValidationError
                return
            raise ValidationError
        if self.value is not None and value != self.value:
            raise ValidationError("%s: %s != %s" % (self.name, value, self.value))


class Int(Field):
    def validate(self, obj, value):
        super(Int, self).validate(obj, value)
        if not isinstance(value, numbers.Integral):
            raise ValidationError("%s" % value)
        if self.max is not None and value > self.max:
            raise ValidationError("%s: %s > %s" % (self.name, value, self.max))
        size = self.size
        if isinstance(size, basestring):
            size = getattr(obj, size)
            if size is None or size == 0:
                if value is not None:
                    raise ValidationError("%s: %s" % (self.name, value))
                return
            size = size if size < 3 else 4
            size *= 8
        elif callable(size):
            size = size(obj)
        if not 0 <= value < 2**size:
            raise ValidationError


class Bool(Int):
    def __init__(self, **kwargs):
        kwargs.update(size=1)
        super(Bool, self).__init__(**kwargs)

    def clean(self, value):
        return bool(value)


class Str(Field):
    def __init__(self, *args, **kwargs):
        super(Str, self).__init__(*args, **kwargs)
        if self.size % 8 != 0:
            raise ValidationError

    def validate(self, obj, value):
        super(Str, self).validate(obj, value)
        if not isinstance(value, basestring):
            raise ValidationError
        if len(value) != self.size // 8:
            raise ValidationError("%s %s" % (value, self.size))

    def parse(self, obj, seq):
        if self.skip(obj):
            setattr(obj, self.name, None)
            return
        if obj.bit != 0:
            raise ValidationError
        if self.size > 8*(len(seq) - obj.byte):
            raise ValidationError
        result = ''.join(map(chr, seq[obj.byte:obj.byte+self.size//8]))
        obj.byte += self.size // 8
        setattr(obj, self.name, result)

    def deparse(self, obj):
        if self.skip(obj):
            return []
        if obj.prev != 0 or obj.bit != 0:
            raise ValidationError
        return map(ord, getattr(obj, self.name))


class CStr(Field):
    top = " aehilnorstbcdfgjkmpquvwxyz+-,!.?:;'*%$"

    def __init__(self, size=8, **kwargs):
        kwargs.update(size=size)
        super(CStr, self).__init__(**kwargs)
        if self.size % 8 != 0:
            raise ValidationError

    def validate(self, obj, value):
        super(CStr, self).validate(obj, value)
        if not isinstance(value, basestring):
            raise ValidationError

    def parse(self, obj, seq):
        if self.skip(obj):
            setattr(obj, self.name, None)
            return
        if obj.bit != 0:
            raise ValidationError
        realsize = sum(x<<(8*n) for n, x in
                       enumerate(seq[obj.byte:obj.byte+self.size//8]))
        obj.byte += self.size // 8
        if realsize == 0:
            setattr(obj, self.name, '')
            obj.byte += 1
            return
        if realsize > len(seq) - obj.byte:
            raise ValidationError
        result = self.decompress(seq[obj.byte:obj.byte+realsize])
        obj.byte += realsize
        setattr(obj, self.name, result)

    def deparse(self, obj):
        if self.skip(obj):
            return []
        if obj.prev != 0 or obj.bit != 0:
            raise ValidationError
        S = self.compress(getattr(obj, self.name))
        L = len(S)
        size = [L>>(8*n) & 0xff for n in xrange(self.size//8)]
        return size + S

    def decompress(self, lst):
        tmp = ((x>>i) & 0xf for x in lst for i in (4,0))
        result = []
        for x in tmp:
            if 0x0 <= x <= 0xA:
                C = self.top[x]
            elif 0xB <= x <= 0xE:
                x = ((x-0xB)<<4) + tmp.next()
                if x < 0x1A:
                    C = chr(x + 0x41)
                elif x < 0x24:
                    C = chr(x + 0x16)
                else:
                    C = self.top[x - 0x19]
            elif x == 0xF:
                try:
                    C = chr(tmp.next() + (tmp.next()<<4))
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
                result.extend(((tmp>>4) + 0xB, tmp & 0xF))
            else:
                tmp = ord(c)
                if 0x41 <= tmp < 0x5B:
                    tmp -= 0x41
                    result.extend(((tmp>>4) + 0xB, tmp & 0xF))
                elif 0x30 <= tmp < 0x3A:
                    tmp -= 0x16
                    result.extend(((tmp>>4) + 0xB, tmp & 0xF))
                else:
                    result.extend((0xF, tmp>>4, tmp & 0xF))
        if len(result) % 2 != 0:
            result.append(0xF)
        return [(result[i]<<4)+result[i+1] for i in xrange(0, len(result), 2)]


class Array(Field):
    def __init__(self, head=8, size=8, length=None, **kwargs):
        kwargs.update(size=size)
        super(Array, self).__init__(**kwargs)
        self.head = head
        self.length = length

    def parse(self, obj, seq):
        if self.skip(obj):
            setattr(obj, self.name, None)
            return
        if obj.bit != 0:
            raise ValidationError
        if self.length is None:
            reallength = sum(x<<(8*n) for n, x in
                             enumerate(seq[obj.byte:obj.byte+self.head//8]))
            obj.byte += self.head // 8
        elif callable(self.length):
            reallength = self.length(obj)
        else:
            reallength = self.length
        if callable(self.size):
            size = self.size(obj)
        else:
            size = self.size
        if reallength * size//8 > len(seq) - obj.byte:
            raise ValidationError
        result = seq[obj.byte:obj.byte + reallength*size//8]
        result = zip(*(iter(result),) * (size//8))
        result = tuple(sum(x<<(8*n) for n, x in enumerate(b)) for b in result)
        obj.byte += reallength * size//8
        setattr(obj, self.name, result)

    def deparse(self, obj):
        if self.skip(obj):
            return ()
        if obj.prev != 0 or obj.bit != 0:
            raise ValidationError
        value = getattr(obj, self.name)
        L = len(value)
        if callable(self.size):
            size = self.size(obj)
        else:
            size = self.size
        value = tuple(x>>(8*n) & 0xff for x in value
                      for n in xrange(size//8))
        if self.length is None:
            value = tuple((L>>(8*n)) & 0xff
                          for n in xrange(self.head//8)) + value
        return value

    def validate(self, obj, value):
        super(Array, self).validate(obj, value)
        if callable(self.size):
            size = self.size(obj)
        else:
            size = self.size
        if not all(0 <= x < 2**size for x in value):
            raise ValidationError
        if self.length is None:
            if not 0 <= len(value) < 2**self.head:
                raise ValidationError
        elif callable(self.length):
            if len(value) != self.length(obj):
                raise ValidationError
        else:
            if len(value) != self.length:
                raise ValidationError


class StructBase(type):
    def __new__(cls, name, bases, attrs):
        super_new = super(StructBase, cls).__new__
        parents = [b for b in bases if isinstance(b, StructBase)]
        if not parents:
            return super_new(cls, name, bases, attrs)

        module = attrs.pop('__module__')
        new_class = super_new(cls, name, bases, {'__module__': module})

        new_class.add_to_class('fields', [])
        for obj_name, obj in attrs.iteritems():
            new_class.add_to_class(obj_name, obj)
        if 'type' in attrs:
            new_class._registry[attrs['type']] = new_class

        return new_class

    def add_to_class(cls, name, value):
        if hasattr(value, 'contribute_to_class'):
            value.contribute_to_class(cls, name)
        else:
            setattr(cls, name, value)


class Struct(object):
    __metaclass__ = StructBase
    _registry = {}

    encrypted = True

    def __init__(self, sfile):
        self.file = sfile

    def __unicode__(self):
        return "{%s}" % (', '.join("%s: %r" % (f.name, getattr(self, f.name))
                                   for f in self.fields
                                   if getattr(self, f.name) is not None),)

    @property
    def bytes(self):
        seq, self.prev, self.bit = [], 0, 0
        for field in self.fields:
            seq.extend(field.deparse(self))
        if self.bit != 0 or self.prev != 0:
            raise ValidationError
        return tuple(seq)

    @bytes.setter
    def bytes(self, seq):
        self.byte, self.bit, self.length = 0, 0, len(seq)
        for field in self.fields:
            field.parse(self, seq)
            #print field.name, self.byte, getattr(self, field.name)
        if self.byte != len(seq) or self.bit != 0:
            raise ValidationError("%s %s (%s %s) %s" % (self.__class__.__name__,
                                                     len(seq), self.byte,
                                                     self.bit, seq))

    def adjust(self):
        return


class StarsFile(object):
    # Primes, but not really.  279 should be 269.
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
        self.structs = []
        self.stars = 0

    def prng_init(self, uid, turn, player, salt, flag):
        i, j = (salt>>5) & 0x1f, salt & 0x1f
        if salt < (1<<10):
            i += (1<<5)
        else:
            j += (1<<5)
        self.hi, self.lo = self.prime[i], self.prime[j]

        seed = ((player%4)+1) * ((uid%4)+1) * ((turn%4)+1) + flag
        for i in xrange(seed):
            burn = self.prng()

    def prng(self):
        self.lo = (0x7fffffab * int(self.lo/-53668) + 40014 * self.lo)%(1<<32)
        if self.lo >= (1<<31): self.lo += 0x7fffffab - (1<<32)
        self.hi = (0x7fffff07 * int(self.hi/-52774) + 40692 * self.hi)%(1<<32)
        if self.hi >= (1<<31): self.hi += 0x7fffff07 - (1<<32)
        return (self.lo - self.hi) % (1<<32)

    def crypt(self, seq):
        L = len(seq)
        oL = L
        if L % 4 != 0:
            seq = list(seq) + [0] * (4 - L%4)
            L = len(seq)
        tmp = struct.pack("%dB" % L, *seq)
        tmp = struct.unpack("%dI" % (L//4), tmp)
        tmp = [x^self.prng() for x in tmp]
        tmp = struct.pack("%dI" % (L//4), *tmp)
        tmp = struct.unpack("%dB" % L, tmp)
        return tmp[:oL]

    @property
    def bytes(self):
        seq = []
        for S in self.structs:
            if S.type is not None:
                L = len(S.bytes)
                seq.extend((L & 0xff, S.type<<2 | L>>8))
            seq.extend(self.crypt(S.bytes) if S.encrypted else S.bytes)
            S.adjust()
        return ''.join(map(chr, seq))

    @bytes.setter
    def bytes(self, data):
        index = 0
        self.structs = []
        while index < len(data):
            if self.stars > 0:
                stype, size = None, 4
            else:
                hdr = struct.unpack("H", data[index:index+2])[0]
                stype, size = (hdr & 0xfc00)>>10, hdr & 0x03ff
                index += 2

            S = self.dispatch(stype)
            self.structs.append(S)
            buf = struct.unpack("%dB" % size, data[index:index+size])
            if S.encrypted:
                buf = self.crypt(buf)
            S.bytes = buf
            S.adjust()
            index += size

    def dispatch(self, stype):
        if stype in Struct._registry:
            return Struct._registry[stype](self)
        instance = FakeStruct(self)
        instance.type = stype
        return instance


ftypes = ('xy', 'x', 'hst', 'm', 'h', 'r')

def filetypes(*args):
    def ftype_check(s):
        return s.file.type in args
    return ftype_check


class FakeStruct(Struct):
    bytes = None

    def __unicode__(self):
        return unicode(self.bytes)


class Star(Struct):
    type = None
    encrypted = False

    dx = Int(10)
    y = Int(12)
    name_id = Int(10)

    def adjust(self):
        self.file.stars -= 1


class Type0(Struct):
    """ End of file """
    type = 0
    encrypted = False

    info = Int(option=filetypes('hst', 'xy', 'm'))


class Type8(Struct):
    """ Beginning of file """
    type = 8
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
    unused = Int(3)

    def adjust(self):
        self.file.prng_init(self.game_id, self.turn, self.player,
                            self.salt, self.shareware)
        self.file.type = ftypes[self.filetype]


class Type7(Struct):
    """ Game definition """
    type = 7

    game_id = Int(32)
    size = Int()
    density = Int()
    num_players = Int()
    num_stars = Int()
    start_distance = Int()
    unknown1 = Int()
    flags1 = Int(8)
    unknown2 = Int(24)
    req_pct_planets_owned = Int(8)
    req_tech_level = Int(8)
    req_tech_num_fields = Int(8)
    req_exceeds_score = Int(8)
    req_pct_exceeds_2nd = Int(8)
    req_exceeds_prod = Int(8)
    req_capships = Int(8)
    req_highscore_year = Int(8)
    req_num_criteria = Int(8)
    year_declared = Int(8)
    unknown3 = Int()
    game_name = Str(32*8)

    def adjust(self):
        self.file.stars = self.num_stars


def type6_trigger(S):
    return S.optional_section

class Type6(Struct):
    """ Race data """
    type = 6

    player = Int(8)
    num_ship_designs = Int(8)
    planets_known = Int()
    visible_fleets = Int(12)
    station_designs = Int(4)
    unknown1 = Int(2, value=3)
    optional_section = Bool()
    race_icon = Int(5)
    unknown2 = Int(8) # 227 is computer control
    # optional section
    unknown3 = Int(32, option=type6_trigger) # not const
    password_hash = Int(32, option=type6_trigger)
    mid_G = Int(8, option=type6_trigger)
    mid_T = Int(8, option=type6_trigger)
    mid_R = Int(8, option=type6_trigger)
    min_G = Int(8, option=type6_trigger)
    min_T = Int(8, option=type6_trigger)
    min_R = Int(8, option=type6_trigger)
    max_G = Int(8, option=type6_trigger)
    max_T = Int(8, option=type6_trigger)
    max_R = Int(8, option=type6_trigger)
    growth = Int(8, option=type6_trigger)
    cur_energy = Int(8, option=type6_trigger)
    cur_weapons = Int(8, option=type6_trigger)
    cur_propulsion = Int(8, option=type6_trigger)
    cur_construction = Int(8, option=type6_trigger)
    cur_electronics = Int(8, option=type6_trigger)
    cur_biotech = Int(8, option=type6_trigger)
    # no idea yet
    whatever = Int(30*8, option=type6_trigger)
    col_per_res = Int(8, option=type6_trigger)
    res_per_10f = Int(8, option=type6_trigger)
    f_build_res = Int(8, option=type6_trigger)
    f_per_10kcol = Int(8, option=type6_trigger)
    min_per_10m = Int(8, option=type6_trigger)
    m_build_res = Int(8, option=type6_trigger)
    m_per_10kcol = Int(8, option=type6_trigger)
    leftover = Int(8, option=type6_trigger)
    energy = Int(8, max=2, option=type6_trigger)
    weapons = Int(8, max=2, option=type6_trigger)
    propulsion = Int(8, max=2, option=type6_trigger)
    construction = Int(8, max=2, option=type6_trigger)
    electronics = Int(8, max=2, option=type6_trigger)
    biotech = Int(8, max=2, option=type6_trigger)
    prt = Int(option=type6_trigger)
    imp_fuel_eff = Bool(option=type6_trigger)
    tot_terraform = Bool(option=type6_trigger)
    adv_remote_mine = Bool(option=type6_trigger)
    imp_starbases = Bool(option=type6_trigger)
    gen_research = Bool(option=type6_trigger)
    ult_recycling = Bool(option=type6_trigger)
    min_alchemy = Bool(option=type6_trigger)
    no_ramscoops = Bool(option=type6_trigger)
    cheap_engines = Bool(option=type6_trigger)
    only_basic_mine = Bool(option=type6_trigger)
    no_adv_scanners = Bool(option=type6_trigger)
    low_start_pop = Bool(option=type6_trigger)
    bleeding_edge = Bool(option=type6_trigger)
    regen_shields = Bool(option=type6_trigger)
    ignore = Int(2, value=0, option=type6_trigger)
    unknown4 = Int(8, value=0, option=type6_trigger)
    f1 = Bool(option=type6_trigger)
    f2 = Bool(option=type6_trigger)
    f3 = Bool(option=type6_trigger)
    f4 = Bool(option=type6_trigger)
    f5 = Bool(option=type6_trigger)
    p75_higher_tech = Bool(option=type6_trigger)
    f7 = Bool(option=type6_trigger)
    f_1kTlessGe = Bool(option=type6_trigger)
    # no idea yet
    whatever2 = Int(30*8, option=type6_trigger)
    unknown5 = Array(8, option=type6_trigger)
    # end optional section
    race_name = CStr(8)
    plural_race_name = CStr(8)


class Type45(Struct):
    """ Score data """
    type = 45

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
    year = Int() # or rank in .m files
    score = Int(32)
    resources = Int(32)
    planets = Int()
    starbases = Int()
    unarmed_ships = Int()
    escort_ships = Int()
    capital_ships = Int()
    tech_levels = Int()


class Type13(Struct):
    """ Authoritative Planet """
    type = 13

    planet_id = Int(11, max=998)
    player = Int(5)
    low_info = Bool(value=True) # add station design, if relevant
    med_info = Bool(value=True) # add minerals & hab
    full_info = Bool(value=True) # add real pop & structures
    const = Int(4, value=0)
    homeworld = Bool()
    f0 = Bool(value=True)
    station = Bool()
    terraformed = Bool()
    facilities = Bool() # turns on 8 bytes; rename
    artifact = Bool()
    surface_min = Bool()
    routing = Bool() # turns on 2 bytes
    f7 = Bool()
    s1 = Int(2, max=1)
    s2 = Int(2, max=1)
    s3 = Int(4, max=1)
    frac_ir_conc = Int('s1')
    frac_bo_conc = Int('s2')
    frac_ge_conc = Int('s3')
    ir_conc = Int(8)
    bo_conc = Int(8)
    ge_conc = Int(8)
    grav = Int(8)
    temp = Int(8)
    rad = Int(8)
    grav_orig = Int(8, option=lambda s: s.terraformed)
    temp_orig = Int(8, option=lambda s: s.terraformed)
    rad_orig = Int(8, option=lambda s: s.terraformed)
    apparent_pop = Int(12, option=lambda s: s.player < 16) # times 400
    apparent_defense = Int(4, option=lambda s: s.player < 16)
    s4 = Int(2, option=lambda s: s.surface_min)
    s5 = Int(2, option=lambda s: s.surface_min)
    s6 = Int(2, option=lambda s: s.surface_min)
    s7 = Int(2, option=lambda s: s.surface_min)
    ir_surf = Int('s4')
    bo_surf = Int('s5')
    ge_surf = Int('s6')
    population = Int('s7') # times 100
    frac_population = Int(8, max=99, option=lambda s: s.facilities)
    mines = Int(12, option=lambda s: s.facilities)
    factories = Int(12, option=lambda s: s.facilities)
    defenses = Int(8, option=lambda s: s.facilities)
    unknown3 = Int(24, option=lambda s: s.facilities)
    station_design = Int(4, max=9, option=lambda s: s.station)
    station_flags = Int(28, option=lambda s: s.station)
    routing_dest = Int(option=lambda s: s.routing and s.player < 16)


class Type14(Struct):
    """ Scanned Planet """
    type = 14

    planet_id = Int(11, max=998)
    player = Int(5)
    low_info = Bool() # add station design, if relevant
    med_info = Bool() # add minerals & hab
    full_info = Bool() # add real pop & structures
    const = Int(4, value=0)
    homeworld = Bool()
    f0 = Bool(value=True)
    station = Bool()
    terraformed = Bool()
    facilities = Bool(value=False) # turns on 8 bytes; rename
    artifact = Bool()
    surface_min = Bool()
    routing = Bool() # turns on 2 bytes
    f7 = Bool()
    s1 = Int(2, max=1, option=lambda s: s.med_info or s.full_info)
    s2 = Int(2, max=1, option=lambda s: s.med_info or s.full_info)
    s3 = Int(4, max=1, option=lambda s: s.med_info or s.full_info)
    frac_ir_conc = Int('s1')
    frac_bo_conc = Int('s2')
    frac_ge_conc = Int('s3')
    ir_conc = Int(8, option=lambda s: s.med_info or s.full_info)
    bo_conc = Int(8, option=lambda s: s.med_info or s.full_info)
    ge_conc = Int(8, option=lambda s: s.med_info or s.full_info)
    grav = Int(8, option=lambda s: s.med_info or s.full_info)
    temp = Int(8, option=lambda s: s.med_info or s.full_info)
    rad = Int(8, option=lambda s: s.med_info or s.full_info)
    grav_orig = Int(8, option=lambda s:
                        (s.med_info or s.full_info) and s.terraformed)
    temp_orig = Int(8, option=lambda s:
                        (s.med_info or s.full_info) and s.terraformed)
    rad_orig = Int(8, option=lambda s:
                        (s.med_info or s.full_info) and s.terraformed)
    apparent_pop = Int(12, option=lambda s: # times 400
                           (s.med_info or s.full_info) and s.player < 16)
    apparent_defense = Int(4, option=lambda s:
                               (s.med_info or s.full_info) and s.player < 16)
    s4 = Int(2, option=lambda s: s.full_info and s.surface_min)
    s5 = Int(2, option=lambda s: s.full_info and s.surface_min)
    s6 = Int(2, option=lambda s: s.full_info and s.surface_min)
    s7 = Int(2, option=lambda s: s.full_info and s.surface_min)
    ir_surf = Int('s4')
    bo_surf = Int('s5')
    ge_surf = Int('s6')
    station_design = Int(8, max=9, option=lambda s: s.station)
    last_scanned = Int(option=filetypes('h'))


class Type20(Struct):
    """ Waypoint """
    type = 20

    x = Int()
    y = Int()
    planet_id = Int()
    order = Int(4)
    warp = Int(4)
    unknown1 = Int(8)


class Type19(Struct):
    """ Orders-at Waypoint """
    type = 19

    x = Int()
    y = Int()
    planet_id = Int()
    order = Int(4)
    warp = Int(4)
    unknown1 = Int(8)
    ir_quant = Int(12)
    ir_order = Int(4)
    bo_quant = Int(12)
    bo_order = Int(4)
    ge_quant = Int(12)
    ge_order = Int(4)
    col_quant = Int(12)
    col_order = Int(4)
    fuel_quant = Int(12)
    fuel_order = Int(4)


class Type17(Struct):
    """ Alien Fleet """
    type = 17

    fleet_id = Int(9)
    player = Int(7)
    player2 = Int()
    info_level = Int(8)
    flags = Int(8)
    planet_id = Int()
    x = Int()
    y = Int()
    design_bits = Int()
    count_array = Array(size=lambda s: 16 - (s.flags & 0x8),
                        length=lambda s: bin(s.design_bits).count('1'))
    s1 = Int(2, option=lambda s: s.info_level >= 4)
    s2 = Int(2, option=lambda s: s.info_level >= 4)
    s3 = Int(2, option=lambda s: s.info_level >= 4)
    s4 = Int(2, option=lambda s: s.info_level >= 4)
    ironium = Int('s1')
    boranium = Int('s2')
    germanium = Int('s3')
    fuel = Int(size=lambda s: 8 * (s.s4 or 1),
               option=lambda s: s.info_level >= 4)
    dx = Int(8)
    dy = Int(8)
    warp = Int(4)
    unknown2 = Int(12)
    mass = Int(32)


# class Type3(Struct):
#     """ Delete waypoint """
#     type = 3

#     fleet_id = Int(9)
#     unknown1 = Int(7)
#     sequence_num = Int(8)
#     unknown2 = Int(8)


# class Type16(Struct):
#     """ Authoritative Fleet """
#     type = 16

#     fleet_id = Int(9)
#     player = Int(7)
#     player2 = Int()
#     unknown1 = Int()
#     planet_id = Int()
#     x = Int()
#     y = Int()
#     design_bits = Int()
#     count_array = Int(sizing)


# class Type30(Struct):
#     """ Battle plans """
#     type = 30

#     u1 = Int(8)
#     u2 = Int(8)
#     u3 = Int(8)
#     u4 = Int(8)
#     name = CStr()


# class Type40(Struct):
#     """ In-game messages """
#     type = 40

#     unknown1 = Int(32)
#     sender_id = Int(8)
#     unknown2 = Int(40)
#     text = CStr(16)


# class Type43(Struct):
#     """ Mass Packets / Debris / Wormholes / Mine Fields / Mystery Trader """
#     type = 43

#     # optional sizes: 2, 4, and 18
#     unknown1 = Int()
#     x = Int()
#     y = Int()
#     planet_id = Int(10)
#     unknown2 = Int(6)
#     mass_ir = Int()
#     mass_bo = Int()
#     mass_ge = Int()
#     unknown3 = Int(32)
