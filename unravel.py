from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import re

from starslib import base
import six

starsRE = re.compile(r'.*\.([hmxr]{1}[0-9]{1,2}|xy|hst)')

foo = base.StarsFile()

fpath = sys.argv[1]
fpaths = []
if os.path.isfile(fpath):
    fpaths = [fpath]
else:
    for root, dirs, files in os.walk(fpath):
        fpaths.extend(os.path.join(root, fname) for fname in files
                      if starsRE.match(fname))


for fpath in fpaths:
    with open(fpath, 'rb') as f:
        foo.bytes = f.read()

    for S in foo.structs:
        print(S.type, six.text_type(S))
