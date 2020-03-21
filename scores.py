from __future__ import absolute_import
import getopt
import sys

from starslib import base


opts, files = getopt.getopt(sys.argv[1:], '', ['prefix='])

prefix = 'player_'
for o, a in opts:
    if o == '--prefix':
        prefix = a

sfiles = []
for fname in files:
    sfiles.append(base.StarsFile())
    with open(fname, 'r') as f:
        sfiles[-1].bytes = f.read()

scores = {}
for sf in sfiles:
    for S in sf.structs:
        if S.type == 8:
            year = S.turn
            if year not in scores:
                scores[year] = {}
        if S.type == 45:
            scores[year][S.player] = (S.score, S.resources, S.planets,
                                      S.starbases, S.unarmed_ships,
                                      S.escort_ships, S.capital_ships,
                                      S.tech_levels)
    if not scores[year]:
        del scores[year]

if not scores:
    sys.exit()
year = max(scores.keys())
for i, fext in enumerate(('score', 'resources', 'planets', 'starbases',
                          'unarmed', 'escorts', 'capital', 'tech')):
    with open(prefix+fext, 'a') as f:
        line = ' '.join(str(v[i]) for k, v in sorted(scores[year].items()))
        f.write("%s: %s\n" % (2400+year, line))
