# Paths are relative to the .exp file using the parameters file
resDir=../../resources
types=%{resDir}/amr.types
specmap=%{resDir}/amr.specmap
stanfordModel=%{resDir}/stanford-models/english-bidirectional-distsim.tagger
nerTranslation=%{resDir}/amr.ner
nerConfig=%{resDir}/IllinoisNER.config
chartLogs=%{outputDir}/charts
# beam for CKY parser
gBeam=50
# beam for Neural Parser
beamSize=512
#epochs
epoch=3
#learning parameters
learningRate=0.05
learningRateDecay=0.0
l2=0.000001
seed=1234
conditionedBeam=50
gIter=10
logLevel=INFO
trainingMaxSentenceLength=35
smatchPath=../../utils/smatch/smatch.py
propBank=%{resDir}/propbank
learningEstimator=adagrad
