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
    print hex(seed)
    for i in xrange(seed):
        print hex(lo), hex(hi)
        burn = prng()
    print hex(lo), hex(hi)

ind = 0
def read_struct():
    global ind, data
    hdr = struct.unpack("H", data[ind:ind+2])[0]
    ind += 2
    stype, size = hdr & 0xfc00, hdr & 0x03ff
    res = struct.unpack("%dB" % size, data[ind:ind+size])
    ind += size
    return res

def read_field(size):
    global ind, data
    ind += size
    return

with open(sys.argv[1], 'rb') as f:
    data = f.read()

bof = read_struct()
flag = (bof[15] & 0x10)>>4
turn = bof[10] + (1<<8)*bof[11]
salt = bof[12] + (1<<8)*bof[13]
player, salt = salt & 0x001f, (salt % 0xffe0)>>5
uid = bof[4] + (1<<8)*bof[5] + (1<<16)*bof[6] + (1<<24)*bof[7]
print flag, player, turn, salt, uid
prng_init(flag, player, turn, salt, uid)
mystery = read_struct()

m2 = []
for i in xrange(len(mystery)//4):
    rnd = prng()
    tmp = mystery[4*i:4*(i+1)]
    tmp = tmp[0] + (1<<8)*tmp[1] + (1<<16)*tmp[2] + (1<<24)*tmp[3]
    tmp = tmp ^ rnd
    m2.extend([0xff & (tmp>>(8*j)) for j in xrange(4)])

print m2[:32], map(chr, m2[32:])
