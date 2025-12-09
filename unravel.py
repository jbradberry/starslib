import sys
import os
import re

from starslib import base


starsRE = re.compile(r'.*\.([hmxr]{1}[0-9]{1,2}|xy|hst)')

fpath = sys.argv[1]
fpaths = []
if os.path.isfile(fpath):
    fpaths = [fpath]
else:
    for root, dirs, files in os.walk(fpath):
        fpaths.extend(os.path.join(root, fname) for fname in files
                      if starsRE.match(fname))


for fpath in fpaths:
    foo = base.StarsFile()

    with open(fpath, 'rb') as f:
        try:
            foo.bytes = f.read()
        except Exception:  # FIXME: do chained exceptions here?
            print("Problem found with {}".format(fpath))
            raise

    for S in foo.structs:
        print(S.type, str(S))
