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
import re, sys
from collections import defaultdict

'''
    Convert an experiment from the old typing system to the new one.
'''



if __name__ == '__main__':
    # Parse command line Arguments
    parser = OptionParser(usage = "usage: %prog data_files\ndata_files may include conversation, sentence pairs, lexicons and such")
    parser.add_option('-i', '--input-types', dest = "org_types", help = 'The original typing file')
    parser.add_option('-p', '--parallel-file', dest = 'parallel_types', help = 'A parallel file containing the new format')
    parser.add_option('-o', '--output-prefix', dest = 'output_prefix', help = 'Output prefix')
    parser.add_option('-d', '--output-debug', dest = 'debug_output', action = 'store_true', help = 'Output debug files')
    (options, args) = parser.parse_args()

    # Open the parallel files and verify that they have the same number of lines
    if len(open(options.org_types).readlines()) != len(open(options.parallel_types).readlines()):
        raise Exception('Original types file and the parallel files must have an identical number of lines')

    # Zip the parallel files and read them into a mapping structure
    preds = {}
    primitive_types = []
    preds_to_comments = {}
    for org_line, new_line in zip(open(options.org_types).readlines(), open(options.parallel_types).readlines()):
        # Get type from original file
        pred = None
        m = re.match('\((?P<pred>[a-z_0-9<>=]+:[a-z_]+)\s.+?(?:(?://(?P<comment>.+)$)|$)', org_line)
        if m:
            pred = m.groupdict()['pred']
            if m.groupdict()['comment']:
                preds_to_comments[pred] = m.groupdict()['comment']
        else:
            m = re.match('^(?P<type>(?:\([a-z_0-9]+\s[a-z_]+\))|(?:[a-z]+))', org_line)
            if m:
                primitive_types.append(org_line)
            elif org_line.strip() != '':
                print >> sys.stdout, 'Skipping line: %s' % (org_line.strip())
            continue


        # Get type from new file
        m = re.match('^(?P<pred>[a-z_0-9<>=]+:[<>,a-z_]+)', new_line)
        new_pred = m.groupdict()['pred']

        # Verify that the constant name didn't change
        if pred.split(':', 1)[0] != new_pred.split(':', 1)[0]:
            raise Exception('Predicate name changed: %s -> %s' % (pred, new_pred))

        preds[pred] = new_pred
        print >> sys.stderr, 'Updating: %s -> %s' % (pred, new_pred)

    print >> sys.stderr, 'Read %d predicates' % (len(preds))
    print >> sys.stderr, 'Read %d primitive types' % (len(primitive_types))

    # Output the primitive types file
    out = open(options.output_prefix + '.types', 'w')
    out.write('(\n\n')
    for line in primitive_types:
        out.write(line)
    out.write('\n)')
    out.close()

    # Output the predicates ontology file
    out = open(options.output_prefix + '.preds.ont', 'w')
    out.write('(\n\n')
    for old_pred, new_pred in preds.items():
        out.write(new_pred)
        if old_pred in preds_to_comments:
            out.write('\t //')
            out.write(preds_to_comments[old_pred])
        out.write('\n')
    out.write('\n)')
    out.close()

    preds['and:t'] = 'and:<t*,t>'
    preds['or:t'] = 'or:<t*,t>'
    preds['implies:t'] = 'implies:<e,<e,t>>'
    preds['not:t'] = 'not:<t,t>'
    preds['inc:ind'] = 'inc:<ind,ind>'

    def special_pred(pred):
        s = pred.split(':', 1)
        pred_name = s[0]
        ret_type = s[1]
        if pred_name == 'i':
            # Case index predicate (i:type -> i:<type[],<ind, type>>)
            return '%s:<%s[],<ind,%s>>' % (pred_name, ret_type, ret_type)
        elif pred_name == 'sub':
            # Case sub predicate (sub:type[] -> sub:<type[], <ind, type[]>>)
            return '%s:<%s,<ind,%s>>' % (pred_name, ret_type, ret_type)
        else:
            raise Exception('Unknown predicate %s' % (pred))

    # Iterate over the data files and replace using the mapping and regular expression. Collect constants while going over the files.
    constants = set()
    for f in args:
        print 'Processing %s ...' % (f)
        text = open(f).read()
        updated_text = re.sub('(?<=\()(?P<pred>[a-z_0-9<>=]+:[^\s()]+)(?=\s)', lambda m: preds[m.groupdict()['pred']] if m.groupdict()['pred'] in preds else special_pred(m.groupdict()['pred']), text)
        out = open(f, 'w')
        out.write(updated_text)
        out.close()
        constants.update(re.findall('(?<=\s)[^\s()$]+:[^\s()]+', updated_text))


    # Output the constants ontology files
    type_to_consts = defaultdict(list)
    for const in constants:
        type_to_consts[const.split(':')[1]].append(const)
    out = open(options.output_prefix + '.consts.ont', 'w')
    out.write('(\n\n')
    for type, consts in type_to_consts.items():
        for const in consts:
            out.write(const)
            out.write('\n')
        out.write('\n')
    out.write('\n)')
    out.close()

    if options.debug_output:
        out = open(options.output_prefix + '.debug', 'w')
        for old_p, new_p in preds.items():
            out.write('%s\t%s\n' % (old_p, new_p))
        out.close()



