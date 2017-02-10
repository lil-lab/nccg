# Create training data using graph amr parser with cky base parser
type=data.neural.sparsefeatureandstate id=data.neural.sparsefeature trainingData=train baseParser=baseParser beamSize=%{beamSize} graphAmrParser=ckyAmrParser filterFactory=filterFactory
#type=data.neural.sparsefeatureandstate.exploration p=0.5 k=3 id=data.neural.sparsefeature trainingData=train baseParser=baseParser beamSize=%{beamSize} graphAmrParser=ckyAmrParser filterFactory=filterFactory
