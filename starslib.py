from __future__ import division, with_statement
import sys, struct

hi, lo = 0, 0 # really a 64-bit state variable

# Primes, but not really.  279 should be 269.
prime = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
    131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 279,
    271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431,
    433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
    601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673,
    677, 683, 691, 701, 709, 719, 727)

def prng():
    global hi, lo
    lo = (0x7fffffab * int(lo / -53668) + 40014 * lo) & 0xffffffff
    if lo >= 0x80000000: lo -= 0x80000055
    hi = (0x7fffff07 * int(hi / -52774) + 40692 * hi) & 0xffffffff
    if hi >= 0x80000000: hi -= 0x800000f9
    return (lo - hi) & 0xffffffff

def prng_init(flag, player, turn, salt, uid):
    global hi, lo
    i, j = (salt>>5) & 0x1f, salt & 0x1f
    if salt < 0x400:
        i += 0x20
    else:
        j += 0x20
    hi, lo = prime[i], prime[j]

    seed = ((player%4)+1) * ((uid%4)+1) * ((turn%4)+1) + flag
    #print hex(seed)
    for i in xrange(seed):
        #print hex(lo), hex(hi)
        burn = prng()
    #print hex(lo), hex(hi)

def crypt(lst):
    L = len(lst)
    oL = L
    if L % 4 != 0:
        lst = list(lst) + [0] * (4 - L%4)
        L = len(lst)
    tmp = struct.pack("%dB" % L, *lst)
    tmp = struct.unpack("%dI" % (L//4), tmp)
    tmp = [x^prng() for x in tmp]
    tmp = struct.pack("%dI" % (L//4), *tmp)
    tmp = struct.unpack("%dB" % L, tmp)
    return tmp[:oL]

def uint(lst):
    return sum(x<<(8*n) for n, x in enumerate(lst))

def decompress(lst):
    top = " aehilnorstbcdfgjkmpquvwxyz+-,!.?:;'*%$"
    tmp = ((x>>i) & 0xf for x in lst for i in (4,0))
    result = []
    for x in tmp:
        if 0x0 <= x <= 0xA:
            C = top[x]
        elif 0xB <= x <= 0xE:
            x = ((x-0xB)<<4) + tmp.next()
            if x < 0x1A:
                C = chr(x + 0x41)
            elif x < 0x24:
                C = chr(x + 0x16)
            else:
                C = top[x - 0x19]
        elif x == 0xF:
            try:
                C = chr(tmp.next() + (tmp.next()<<4))
            except StopIteration:
                break
        result.append(C)
    return ''.join(result)

def eof(lst):
    return lst

def bof(lst):
    tmp = struct.pack("16B", *lst)
    magic, uid, ver, turn, ps, f = struct.unpack("4sI4H", tmp)
    player = ps & 0x001f
    salt = (ps & 0xffe0)>>5
    share = (f & 0x1000)>>12
    prng_init(share, player, turn, salt, uid)
    return lst

def ixy(lst):
    tmp = struct.pack("4B", *lst)
    tmp = struct.unpack("I", tmp)[0]
    x = tmp & 0x000003ff
    y = (tmp & 0x003ffc00)>>10
    i = (tmp & 0xffc00000)>>22
    return i, x, y

process = {-1: ixy,
           0: eof,
           8: bof}

def parse(data):
    index, head = 0, True
    while index < len(data):
        if head:
            hdr = struct.unpack("H", data[index:index+2])[0]
            stype, size = (hdr & 0xfc00)>>10, hdr & 0x03ff
            index += 2
        else:
            stype, size = -1, 4
        buf = process.get(stype, crypt)(
            struct.unpack("%dB" % size, data[index:index+size]))
        yield (stype, size, buf)
        index += size
        if not head:
            stars -= 1
            if stars == 0: head = True
        if stype == 7:
            head, stars = False, buf[10] + 256*buf[11]


if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        data = parse(f.read())

    for field in data:
        print field
