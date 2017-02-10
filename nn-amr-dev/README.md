Title: Cornell AMR Semantic Parser - README
Author: Yoav Artzi
Affiliation: Computer Science, Cornell University
Web: http://yoavartzi.com/amr

# [Cornell AMR Semantic Parser](http://yoavartzi.com/amr)

[TOC]


## Requirements

Java 8.

## Compiling

`ant dist`

## Parsing

Given a file `some.sentences`, which contains a sentence on each line, and a model file `amr.sp`:

`java -jar dist/amr-1.0.jar parse rootDir=. sentences=../../some.sentences modelFile=../../amr.sp`

## Preparing the data (for training and testing)

`ant dist`

`utils/config.sh`

`utils/prepData.sh`

## Attribution

@InProceedings{artzi-lee-zettlemoyer:2015:EMNLP,
  author    = {Artzi, Yoav  and  Lee, Kenton  and  Zettlemoyer, Luke},
  title     = {Broad-coverage CCG Semantic Parsing with AMR},
  booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing},
  month     = {September},
  year      = {2015},
  address   = {Lisbon, Portugal},
  publisher = {Association for Computational Linguistics},
  pages     = {1699--1710},
  url       = {http://aclweb.org/anthology/D15-1198}
}
