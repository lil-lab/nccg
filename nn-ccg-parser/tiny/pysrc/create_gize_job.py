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
from operator import itemgetter
import re, sys, os, stat
from collections import defaultdict
from singlesentence import SingleSentenceDataset
from tinyutils import *

'''
    Create a GIZA job to get initial co-occurrence scores.
'''

symbol_index_counter = 0
def next_symbol_index():
    global symbol_index_counter
    symbol_index_counter += 1
    return symbol_index_counter

word_index_counter = 0
def next_word_index():
    global word_index_counter
    word_index_counter += 1
    return word_index_counter

symbol_voc = defaultdict(next_symbol_index)
symbol_counters = defaultdict(int)
word_voc = defaultdict(next_word_index)
word_counters = defaultdict(int)

if __name__ == '__main__':
    parser = OptionParser(usage = 'usage: %prog SINGLE_SENTENCE_FILE')
    parser.add_option('-o', '--output-prefix', dest = 'output_prefix', help = 'Output prefix')
    parser.add_option('-g', '--giza-dir', dest = 'giza_dir', help = 'GIZA++ directory, containing binaries.')
    parser.add_option('-s', '--source', dest = 'source', default = 'NL', help = 'Source side, either NL or LF. Default: NL.')
    parser.add_option('-p', '--post-process-script', dest = 'post_process_script', help = 'Post-process Python script')
    (options, args) = parser.parse_args()

    # Read the data
    sentences = SingleSentenceDataset.parse(open(args[0]).read())

    if options.source == 'NL':
        source_is_words = True
    elif options.source == 'LF':
        source_is_words = False
    else:
        raise Exception('Invalid option: ' + option.source)

    bitext_counts = defaultdict(int)
    for s in sentences:
        sent = s.nl()
        label = s.lf()
        print >> sys.stderr, sent
        print >> sys.stderr, label
        print >> sys.stderr

        # Get all 1grams for the sentence
        words = sent.split(' ')
        for word in words:
            word_voc[word]
            word_counters[word] += 1

        # Get all constants from the label
        consts = re.findall('[^\s()]+:[a-z0-9*+,<>]+', label)
        for const in consts:
            symbol_voc[const]
            symbol_counters[const] += 1

        bitext_counts[(' '.join(map(lambda x: str(word_voc[x]), words)),
                       ' '.join(map(lambda x: str(symbol_voc[x]), consts)))] += 1

    # Write vocabulary files
    symbol_voc_filename = options.output_prefix + '.symbols.voc'
    out = open(symbol_voc_filename, 'w')
    for symbol, index in sorted(symbol_voc.items(), key = itemgetter(1)):
        out.write('%d %s %d\n' % (index, str(symbol), symbol_counters[symbol]))
    out.close()

    word_voc_filename = options.output_prefix + '.words.voc'
    out = open(word_voc_filename, 'w')
    for word, index in sorted(word_voc.items(), key = itemgetter(1)):
        out.write('%d %s %d\n' % (index, str(word), word_counters[word]))
    out.close()

    # Direction of model
    if source_is_words:
        source_voc_filename = word_voc_filename
        target_voc_filename = symbol_voc_filename
    else:
        target_voc_filename = word_voc_filename
        source_voc_filename = symbol_voc_filename

    # Write bitext file
    bitext_filename = options.output_prefix + '.bitext'
    out = open(bitext_filename, 'w')
    for bitext_pair, count in bitext_counts.items():
        print >> out, count
        print >> out, bitext_pair[0 if source_is_words else 1]
        print >> out, bitext_pair[1 if source_is_words else 0]
    out.close()

    # Write giza_config file
    giza_config_filename = options.output_prefix + '.giza.config'
    out = open(giza_config_filename, 'w')
    out.write('S: %s\n' % (source_voc_filename))
    out.write('T: %s\n' % (target_voc_filename))
    out.write('C: %s\n' % (bitext_filename))
    out.write('-CoocurrenceFile snt\n')
    out.write('-o giza_out\n')
    out.write('-model1dumpfreq 1\n')
    out.close()

    # Write run script
    run_script_filename = options.output_prefix + '.giza.sh'
    out = open(run_script_filename, 'w')
    out.write('%s/snt2cooc.out %s %s %s > snt\n' % (options.giza_dir, source_voc_filename, target_voc_filename, bitext_filename))
    out.write('%s/GIZA++ %s\n' % (options.giza_dir, giza_config_filename))
    out.write('python %s -w %s -s %s -o %s -t 3 -c %s giza_out.t1.5\n' %
              (options.post_process_script,
              word_voc_filename,
              symbol_voc_filename,
              options.output_prefix + '.giza.cooc',
              options.source))
    out.close()
    os.system('chmod +x %s' % (run_script_filename))








