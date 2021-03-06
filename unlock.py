from __future__ import absolute_import
import sys
import os
import re

from starslib import base

starsRE = re.compile(r'.*\.([hmxr]{1}[0-9]{1,2}|xy|hst)')

foo = base.StarsFile()

for root, dirs, files in os.walk(sys.argv[1]):
    for fname in files:
        if not starsRE.match(fname):
            continue

        with open(os.path.join(root, fname), 'rb') as f:
            foo.bytes = f.read()

        if not any(S.password_hash for S in foo.structs if S.type == 6):
            continue

        for S in foo.structs:
            if S.type == 6:
                S.password_hash = 0

        with open(os.path.join(root, fname), 'wb') as f:
            f.write(foo.bytes)
