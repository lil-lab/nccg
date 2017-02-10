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
import sys
from optparse import OptionParser, OptionGroup
from collections import defaultdict
from operator import itemgetter


'''
    Post process a GIZA output model to get a words and symbols.
'''



if __name__ == '__main__':
    parser = OptionParser(usage = 'usage: %prog -w WORD_VOC -s SYMBOL_VOC GIZA__MODEL')
    parser.add_option('-w', '--word-voc', dest = 'word_voc', help = 'Word vocabulary file')
    parser.add_option('-s', '--symbol-voc', dest = 'symbol_voc', help = 'Symbol vocabulary file')
    parser.add_option('-o', '--output', dest = 'output', help = 'Output file')
    parser.add_option('-t', '--top2err', dest = 'top2err', type = 'int', help = 'Output top-k to stderr for each symbol')
    parser.add_option('-c', '--source', dest = 'source', default = 'NL', help = 'Source side, either NL or LF. Default: NL.')
    (options, args) = parser.parse_args()

    if options.word_voc is None or options.symbol_voc is None:
        parser.error('Missing vocabulary')

    # Read symbol vocabulary
    symbols = {}
    for line in open(options.symbol_voc):
        s = line.strip().split(' ')
        symbols[int(s[0])] = s[1]

    # Read word vocabulary
    words = {}
    for line in open(options.word_voc):
        s = line.strip().split(' ')
        words[int(s[0])] = s[1]

    # Compute indices from direction
    if options.source == 'NL':
        word_index = 0
        symbol_index = 1
    elif options.source == 'LF':
        word_index = 1
        symbol_index = 0
    else:
        raise Exception('Invalid source option: ' + options.source)

    # Read gize model
    symbol_to_word_conf = defaultdict(list)
    for line in open(args[0]):
        s = line.strip().split(' ')
        word = 'null' if int(s[word_index]) == 0 else words[int(s[word_index])]
        symbol = 'null' if int(s[symbol_index]) == 0 else symbols[int(s[symbol_index])]
        score = float(s[2])
        symbol_to_word_conf[symbol].append((word, score))

    out = sys.stdout if options.output is None else open(options.output, 'w')
    for symbol, word_conf in symbol_to_word_conf.items():
        word_conf.sort(key = itemgetter(1), reverse = True)
        for word, score in word_conf:
            print >> out, '%s  ::  %s  ::  %.20f' % (symbol, word, score)

    # Output the top-k if needed
    if not options.top2err is None:
        k = options.top2err
        for symbol, word_conf in sorted(symbol_to_word_conf.items(), key = lambda x: str(x[0])):
            word_conf.sort(key = itemgetter(1), reverse = True)
            for word, score in word_conf[:k]:
                print >> sys.stderr, '%s  ::  %s  ::  %.20f' % (symbol, word, score)
            print >> sys.stderr

