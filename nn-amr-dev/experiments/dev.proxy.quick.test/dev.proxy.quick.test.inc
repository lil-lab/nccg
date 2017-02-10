## Base resources
include=../resources.proxy.inc
include=../model.load.inc
## Data
type=filter.quick id=quickFilter len=8 chars=;,(,)
type=data.filter id=seed data=seed-lex filter=quickFilter
type=data.filter id=train data=proxy-train filter=quickFilter
type=data.filter id=test data=proxy-dev filter=quickFilter
## Learning and testing resources
include=../init.inc
include=../test.inc
include=../learn.inc
