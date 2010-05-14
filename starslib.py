from __future__ import division, with_statement
import sys, struct

hi, lo = 0, 0 # really a 64-bit state variable

prime = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
    131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
    271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431,
    433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
    601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673,
    677, 683, 691, 701, 709, 719, 727)

class StarsStruct(object):
    def __init__(self, data):
        self._data = data

    def __str__(self):
        return str(struct.unpack("%dB" % len(self._data), self._data))

class BOF(StarsStruct):
    def __init__(self, data):
        self._data = data
        (self.magic,
         self.uid,
         self.version,
         self.turn,
         ps, f) = struct.unpack("4sI4H", data)
        self.player = ps & 0x001f
        self.salt = (ps & 0xffe0)>>5
        self.ftype = f & 0x00ff
        self.done = (f & 0x0100)>>8
        self.inuse = (f & 0x0200)>>9
        self.multi = (f & 0x0400)>>10
        self.g_over = (f & 0x0800)>>11
        self.share = (f & 0x1000)>>12
        self.other = (f & 0xe000)>>13
        prng_init(self.share, self.player, self.turn, self.salt, self.uid)

class Field6(StarsStruct):
    pass

class Field7(StarsStruct):
    def __init__(self, data):
        self._data = data
        assert len(data) % 4 == 0
        L = len(data)//4
        tmp = struct.unpack("%dI" % L, data)
        tmp = [i^prng() for i in tmp]
        tmp = struct.pack("%dI" % L, *tmp)
        self.info = struct.unpack("%dB" % 4*L, tmp)
        self.stars = self.info[10] + self.info[11]*256

    def __str__(self):
        return str(self.info)

class StarIdXY(StarsStruct):
    def __init__(self, data):
        self._data = data
        tmp = struct.unpack("I", data)[0]
        self.x = tmp & 0x000003ff
        self.y = (tmp & 0x003ffc00)>>10
        self.ind = (tmp & 0xffc00000)>>22

    def __str__(self):
        return "%3d: %4d, %6d" % (self.ind, self.x, self.y)

class EOF(StarsStruct):
    pass

fields = {8: BOF,
          7: Field7,
          0: EOF}

def prng():
    global hi, lo
    lo = (0x7fffffab * int(lo / -53668) + 40014 * lo) % (1<<32)
    if lo >= (1<<31): lo += 0x7fffffab - (1<<32)
    hi = (0x7fffff07 * int(hi / -52774) + 40692 * hi) % (1<<32)
    if hi >= (1<<31): hi += 0x7fffff07 - (1<<32)
    return (lo - hi) % (1<<32)

def prng_init(flag, player, turn, salt, uid):
    global hi, lo
    i, j = (salt>>5) & 0x1f, salt & 0x1f
    if salt < (1<<10):
        i += (1<<5)
    else:
        j += (1<<5)
    hi, lo = prime[i], prime[j]

    seed = ((player%4)+1) * ((uid%4)+1) * ((turn%4)+1) + flag
    #print hex(seed)
    for i in xrange(seed):
        #print hex(lo), hex(hi)
        burn = prng()
    #print hex(lo), hex(hi)

ind = 0
def read_struct():
    global ind, data
    hdr = struct.unpack("H", data[ind:ind+2])[0]
    ind += 2
    stype, size = (hdr & 0xfc00)>>10, hdr & 0x03ff
    #res = struct.unpack("%dB" % size, data[ind:ind+size])
    res = fields.get(stype, StarsStruct)(data[ind:ind+size])
    ind += size
    return res

def read_field(size):
    global ind, data
    tmp = data[ind:ind+size]
    ind += size
    return tmp

if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        data = f.read()

    while True:
        current = read_struct()
        print type(current), current
        if isinstance(current, EOF):
            break
        if isinstance(current, Field7):
            for i in xrange(current.stars):
                current = StarIdXY(read_field(4))
                print type(current), current
