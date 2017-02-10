# Jobs
## Log initial model
#type=save model=model file=%{outputDir}/amr.pre.sp id=save.pre dep=init,listen
#type=log id=log.pre model=model logger=modelLogger dep=init,listen
## Dataset processing
#type=dataset.creation id=dataset.creator datasetCreator=data.neural.composite model=model
## Learning
#type=train.ff.neural.parser id=train.ff.neural.parser datasetCreator=data.neural.sparsefeature learner=learner model=model  modelNewFeatures=modelNewFeatures
## Bootstrap Neural model
type=bootstrap.ensemble.feed.forward.neural.model id=bootstrap.neural.model model=model modelNewFeatures=modelNewFeatures ensemble=basicParser1,basicParser2 bootstrapFolders=end_1460309554378,end_1460653446723
#type=bootstrap.feed.forward.neural.model id=bootstrap.neural.model learner=learner neuralModelFolder=%{neuralModelFolder} datasetCreator=data.neural.sparsefeature learner=learner model=model modelNewFeatures=modelNewFeatures 
## Log and test final model. The parser model has been logged already
#type=log id=log.post model=model logger=modelLogger dep=train
type=test id=test stats=exact exec=exec tester=tester data=test stats=statsExact,statsSmatch dep=bootstrap.neural.model
#type=save model=model file=%{outputDir}/amr.sp id=save dep=test
