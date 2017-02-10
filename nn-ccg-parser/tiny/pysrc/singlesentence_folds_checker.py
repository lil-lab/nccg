#-------------------------------------------------------------------------------
# UW SPF - The University of Washington Semantic Parsing Framework
# <p>
# Copyright (C) 2013 Yoav Artzi
# <p>
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or any later version.
# <p>
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
# <p>
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#-------------------------------------------------------------------------------
from optparse import OptionParser
from singlesentence import *
import sys

if __name__ == '__main__':
    # Parse command line arguments
    parser = OptionParser(usage = 'usage: %prog [-o output_suffix] folds')
    parser.add_option('-o', '--out-suffix', dest = 'out_suffix', help = 'Output file prefix. If none given, will output to stdout.')
    (options, args) = parser.parse_args()

    folds = []
    for file in args:
        folds.append((file, SingleSentenceDataset.parse(open(file).read())))

    upper_bounds = []
    for file, fold in folds:
        tokens_rest = set()
        for other_file, other_fold in folds:
            if other_fold != fold:
                for s in other_fold:
                    tokens_rest.update(s.nl().split())
        counter = 0
        if options.out_suffix:
            out = open(file + '.' + options.out_suffix, 'w')
        else:
            out = sys.stdout
            print >> out, '==============================================='
            print >> out, file
            print >> out, '-----------------------------------------------'
        for s in fold:
            tokens = set(s.nl().split())
            if not tokens.issubset(tokens_rest):
                diff = tokens.difference(tokens_rest)
                counter += 1
                print >> out, s.nl()
                print >> out, diff
                print >> out, '-----------------------------------------------'
        print >> out, '%d sentence contain at least one unknown token' % (counter)
        upper_bound = 1.0 - float(counter) / len(fold)
        upper_bounds.append(upper_bound)
        print >> out, 'Approximate upper-bound: %f' % (upper_bound)
    if  options.out_suffix is None:
        print >> sys.stdout, '==============================================='
        print >> sys.stdout, '==============================================='
    print >> sys.stdout, 'Average upper bound: %f' % (sum(upper_bounds) / len(upper_bounds))





