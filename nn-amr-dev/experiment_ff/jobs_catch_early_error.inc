# Jobs
## Log initial model
#type=save model=model file=%{outputDir}/amr.pre.sp id=save.pre dep=init,listen
#type=log id=log.pre model=model logger=modelLogger dep=init,listen
## Dataset processing
#type=dataset.creation id=dataset.creator datasetCreator=data.neural.composite model=model
## Bootstrap data
type=bootstrap.feed.forward.neural.model id=bootstrap.feed.forward.neural.model learner=learner
## Learning
type=feed.forward.catch.early.error id=feed.forward.catch.early.error datasetCreator=data.neural.sparsefeature model=model data=train modelNewFeatures=modelNewFeatures
#type=train.ff.neural.parser id=train.ff.neural.parser datasetCreator=data.neural.sparsefeature learner=learner model=model modelNewFeatures=modelNewFeatures 
#dep=dataset.creator
## Bootstrap Neural model
#type=bootstrap.neural.model id=bootstrap.neural.model learner=learner neuralModelFolder=%{neuralModelFolder} parser=baseParser model=model datasetCreator=data.neural.composite
## Log and test final model. The parser model has been logged already
#type=log id=log.post model=model logger=modelLogger dep=train
#type=test id=test stats=exact exec=exec tester=tester data=test stats=statsExact,statsSmatch dep=train.ff.neural.parser
#type=save model=model file=%{outputDir}/amr.sp id=save dep=test
