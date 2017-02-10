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
import random

'''
    Samples a given set without replacing.
'''

if __name__ == '__main__':
    # Parse command line arguments
    parser = OptionParser(usage = 'usage: %prog -p output_prefix -n NUM_SAMPLES -s SIZE dataset_file')
    parser.add_option('-p', '--out-prefix', dest = 'out_prefix', help = 'Output file prefix')
    parser.add_option('-n', '--num-samples', type = "int", dest = 'num_samples', help = 'Number of samples')
    parser.add_option('-s', '--sample-size', type = 'int', dest = 'sample_size', help = 'Number of sentence in each sample')
    (options, args) = parser.parse_args()

    n = options.num_samples
    size = options.sample_size

    # Read the input file
    sentences = set()
    lines = open(args[0]).readlines()
    lines.reverse()
    while len(lines):
        line = lines.pop()
        if line.strip() != '':
            sentence = line.strip()
            label = lines.pop().strip()
            sentences.add((sentence, label))

    # Sample subsets without replacing, output them to separate files
    for sample_num in range(n):
        sample = random.sample(sentences, size)
        out = open(options.out_prefix + '_' + str(sample_num) + '.set', 'w')
        for sentence, label in sample:
            print >> out, sentence
            print >> out, label
            print >> out
        out.close()
