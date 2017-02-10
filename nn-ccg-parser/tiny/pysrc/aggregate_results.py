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
from user import *
from optparse import OptionParser
import matplotlib.pyplot as plt
import numpy as np
import os, math

def aggregate_files(files):
    # Map number of supervised to result dictionaries
    results = defaultdict(list)

    # Read the files
    for filename in files:
        num_supervised = int(os.path.basename(filename).split("_")[0])
        line = open(filename).readline().strip()
        if line == '':
            print >> sys.stderr, "skipping " + filename
        else:
            results[num_supervised].append(dict(map(lambda y: (y[0], float(y[1])), map(lambda x: x.split("="), line.split('\t')))))

    empty_f1 = dict()
    for num_supervised, res in results.items():
        f1s = map(lambda x: x['emptyF1'], res)
        empty_f1[num_supervised] = (np.average(f1s), np.std(f1s) / math.sqrt(len(f1s)))

    return empty_f1

def log(set_to_results):
    for n, pair in set_to_results.items():
        avg = pair[0]
        std = pair[1]
        print '%d :: %f +- %f' % (n, avg, std)



if __name__ == '__main__':
    # Parse command line arguments
    parser = OptionParser(usage = 'usage: %prog -f F1 -s F1 result_files')
    parser.add_option('-f', '--fully_supervised', dest = 'fully_supervised', type = 'float', help = 'Word-skipping F1 for supervised learning')
    parser.add_option('-s', '--semi-supervised-only', dest = 'semi_only', type = 'float', help = 'Word-skipping F1 for semi-supervised learning')
    parser.add_option('-l', '--semi-with-lex-seed', dest = 'semi_with_lex', type = 'float', help = 'Word-skipping F1 for semi-supervised learning with lexicon seed')
    parser.add_option('-n', '--no-learning-with-lex-seed', dest = 'no_learning_with_lex', type = 'float', help = 'Word-skipping F1 for no-learning with lexicon seed')
    (options, args) = parser.parse_args()

    fig = plt.figure()

    # Get semi supervised data
    if len(filter(lambda x: not x.endswith('.no_semi.out'), args)):
        empty_f1 = aggregate_files(filter(lambda x: not x.endswith('.no_semi.out'), args))
        # No supervised samples
        if options.semi_only:
            empty_f1[0] = (options.semi_only, 0)
        xy_pairs = sorted(empty_f1.items(), key = itemgetter(0))
        plt.errorbar(map(itemgetter(0), xy_pairs),
                     map(lambda x: x[1][0], xy_pairs),
                     yerr = map(lambda x: x[1][1], xy_pairs), color = 'blue')
        print 'With semi-supervised data:'
        log(empty_f1)

    # Get supervised data
    if len(filter(lambda x: x.endswith('.no_semi.out'), args)):
        empty_f1_no_semi = aggregate_files(filter(lambda x: x.endswith('.no_semi.out'), args))
        xy_pairs = sorted(empty_f1_no_semi.items(), key = itemgetter(0))
        plt.errorbar(map(itemgetter(0), xy_pairs),
                     map(lambda x: x[1][0], xy_pairs),
                     yerr = map(lambda x: x[1][1], xy_pairs), color = 'red')
        print 'Without semi-supervised data:'
        log(empty_f1_no_semi)

    # Line for the fully supervised case
    if options.fully_supervised:
        plt.axhline(options.fully_supervised)

    # Semi-supervised only with a seed lexicon
    if options.semi_with_lex:
        plt.plot(0, options.semi_with_lex, 'D')

    if options.no_learning_with_lex:
        plt.plot(0, options.no_learning_with_lex, 'D')


    plt.title("F1 with word skipping")
    plt.ylabel('F1')
    plt.xlabel('# supervised samples')
    plt.xlim([-10, 150])
    plt.show()


