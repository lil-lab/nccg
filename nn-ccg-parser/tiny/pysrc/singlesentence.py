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
from tinyutils import *
import sys

'''
    Data structure equivalent to the SingleSentence dataset in the Java code
'''



class SingleSentence:
    def __init__(self, nl, lf):
        self._nl = nl
        self._lf = lf
    def nl(self): return self._nl
    def lf(self): return self._lf
    def __str__(self): return self.to_file_string()
    def to_file_string(self):
        return self.nl() + '\n' + self.lf()
    @staticmethod
    def parse(text):
        s = text.split('\n')
        return SingleSentence(s[0], s[1])

class SingleSentenceDataset(list):
    def __init__(self, sentences):
        super(SingleSentenceDataset, self).__init__(sentences)
    def to_file_string(self):
        ret = ''
        for sentence in self:
            ret += sentence.to_file_string()
            ret += '\n\n'
        return ret
    @staticmethod
    def parse(text):
        chunks = filter(lambda x: x != '', text.split('\n\n'))
        sentences = []
        for chunk in chunks:
            sentence = SingleSentence.parse(chunk)
            sentences.append(sentence)
        return SingleSentenceDataset(sentences)

def preprocess_sentence(sentence):
    sentence._nl = preprocess_text(sentence.nl())

def preprocess_dataset(dataset):
    for sentence in dataset:
        preprocess_sentence(sentence)

def make_dataset_consistent_set(dataset):
    d = {}
    for s in dataset:
        if s.nl() in d and d[s.nl()].lf() != s.lf():
            print >> sys.stderr, 'inconsistent duplicate for ' + s.nl()
        d[(s.nl(), s.lf())] = s
    return SingleSentenceDataset(d.values())

def verify_dataset_consistent(dataset):
    d = {}
    for s in dataset:
        if s.nl() in d and d[s.nl()].lf() != s.lf():
            print >> sys.stderr, 'inconsistent duplicate for ' + s.nl() + ' -> ' + s.lf() + ' | expected: ' + d[s.nl()].lf()
        else:
            d[s.nl()] = s

