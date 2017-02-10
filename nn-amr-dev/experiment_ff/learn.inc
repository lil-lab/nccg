# Learning resources
## Learning tasks (to preform after each iteration)
# type=learn.task.test id=learningTester data=test tester=tester exec=exec smatch=%{smatchPath} smatchRounds=20
# type=learn.task.log id=learningLogger logger=modelLogger path=%{outputDir} model=model base=model-epoch
# type=learn.task.save id=learningSaver model=model dir=%{outputDir} prefix=model-epoch
# type=learn.task.snapshot id=learningSnapshot dir=%{outputDir} objects=adagrad
## Learner
# validation scorer
#type=test.stats.smatch id=validationStatsSmatch smatch=%{smatchPath} rounds=20 timeout=10000 underspec=true
#type=validation.statistics id=validationStats parser=baseParser model=model data=train beamSize=20 stats=validationStatsSmatch
### Backpropagation based learner
type=parser.feedfoward.dotproduct.shiftreduce.learner id=learner train=train baseParser=baseParser l2=%{l2} learningRate=%{learningRate} learningRateDecay=%{learningRateDecay} beamSize=%{beamSize} epoch=%{epoch} pretrain=false saveModelAfterLearning=true
