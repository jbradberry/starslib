import sys
import os
import re

import structs

starsRE = re.compile('.*\.([hmxr]{1}[0-9]{1,2}|xy|hst)')

foo = structs.StarsFile()

for root, dirs, files in os.walk(sys.argv[1]):
    for fname in files:
        if not starsRE.match(fname):
            continue

        with open(os.path.join(root, fname), 'r') as f:
            foo.bytes = f.read()

        if not any(S.password_hash for S in foo.structs if S.type == 6):
            continue

        for S in foo.structs:
            if S.type == 6:
                S.password_hash = 0

        with open(os.path.join(root, fname), 'w') as f:
            f.write(foo.bytes)
