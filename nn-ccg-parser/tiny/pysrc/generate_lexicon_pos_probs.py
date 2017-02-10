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
import nltk
from optparse import OptionParser

if __name__ == '__main__':
    # Parse command line arguments
    parser = OptionParser(usage = 'usage: %prog -l LEXICON SENTENCE_FILE')
    parser.add_option('-l', '--lexicon', dest = 'lexicon', help = 'Lexicon file')
    (options, args) = parser.parse_args()

    # Read the lexicon and construct a dictionary of (words -> syntax list)
    lexicon = defaultdict(list)
    for line in open(options.lexicon):
        s_line = line.strip().split('//')[0]
        if s_line != '':
            words = s_line.split(':-')[0].strip()
            syntax = s_line.split(':-')[1].split(':')[0].strip()
            lexicon[words].append(syntax)


    # Dictionary of POS to syntices to frequency
    pos_to_syntices = defaultdict(lambda: defaultdict(int))

    # Populate the POS to syntax mapping
    for sentence in open(args[0]):
        sentence = sentence.strip()
        tokens = sentence.split(' ')
        pos_tags = nltk.pos_tag(tokens)
        for start in range((len(tokens))):
            for end in range(start, len(tokens)):
                span = tokens[start:end + 1]
                span_pos = pos_tags[start:end + 1]
                span_text = ' '.join(span)
                syntices = lexicon[span_text]
                for syntax in syntices:
                    for pos in span_pos:
                        pos_to_syntices[pos[1]][syntax] += 1

    # Output the probability table, each line is P(syntax | pos)
    for pos, syntices in pos_to_syntices.items():
        total = sum(syntices.values())
        for syntax, count in syntices.items():
            print >> sys.stdout, '%s\t%s\t%f' % (pos, syntax, float(count) / float(total))



