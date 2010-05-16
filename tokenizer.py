from __future__ import with_statement
import sys, struct

with open(sys.argv[1]) as f:
    data = f.read()

i, hdr = 0, True
while i < len(data):
    if hdr:
        head = struct.unpack("H", data[i:i+2])[0]
        i += 2
        size = head & 0x3ff
        ftyp = head >> 10
        if ftyp == 7:   # special case: next up are headless stellar
            hdr = False # coordinate fields; need decryption to know how many
    else:
        ftyp, size = -1, 4
        if len(data) - i - size == 4:
            hdr = True
    print ftyp, size, struct.unpack("%dB" % size, data[i:i+size])
    i += size
