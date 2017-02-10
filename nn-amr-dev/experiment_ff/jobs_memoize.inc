# Jobs
## Log initial model
#type=save model=model file=%{outputDir}/amr.pre.sp id=save.pre dep=init,listen
#type=log id=log.pre model=model logger=modelLogger dep=init,listen
## Dataset processing
#type=dataset.creation id=dataset.creator datasetCreator=data.neural.composite model=model
## Learning
#type=train.neural.parser id=train.neural.parser datasetCreator=data.neural.composite learner=learner model=model dep=dataset.creator
## Bootstrap neural model
type=bootstrap.neural.model id=bootstrap.neural.model learner=learner neuralModelFolder=%{neuralModelFolder} parser=baseParser model=model datasetCreator=data.neural.composite
## Memoize category embedding
type=memoize.category.embedding id=memoize.category.embedding parser=baseParser model=model testData=test dep=bootstrap.neural.model
## Log and test final model. The parser model has been logged already
#type=log id=log.post model=model logger=modelLogger dep=train
type=test id=test stats=exact exec=exec tester=tester data=test stats=statsExact,statsSmatch,statsUnderspecSmatch dep=memoize.category.embedding
type=save model=model file=%{outputDir}/amr.sp id=save dep=test
