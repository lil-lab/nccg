# Neural Shift Reduce CCG Semantic Parser for AMR Parsing
Neural Shift Reduce Parser for CCG Semantic Parsing (Misra and Artzi EMNLP 2016)

# Author
Developed and maintained by Dipendra Misra (dkm@cs.cornell.edu)

Uses the following code:

1. SPF and AMR code maintained by Yoav Artzi.
2. Stanford parser (BLAH)
3. CCG Supertagging (BLAH)

# Pre Requisite

 - Java 8.0.

# Instructions for using the code

It is best to use the code with Eclipse. Instructions below assume you are using Eclipse.

## Import the code in Eclipse

- Import all the java projects in nn-amr-dev. 

  - To import projects, first open eclipse and change the workspace folder to the root folder of ./nccg. 
  - Now go to File->Import->General->Existing Projects Into Workspace and select the nn-amd-dev folder in the root.
  - You should see the amr project. Select it. You should now see amr in the project explorer. Ignore any errors for now.
- Now import all the java projects in nn-ccg-dev in similar fashion.
- Close the following projects tiny, learn.ubl and learn.weakp (right click on the project and click on Close Project).
- If you see any error then please see the FAQ section or raise an issue.

## Understanding the code structure

- Neural Shift Reduce CCG Semantic parser (NCCG) builds on top of SPF (Semantic Parsing Framework). Please see SPF documentation
  to learn more.
- The NCC project is contained in the project ./parser.ccg.ff.shiftreduce. 
- There are three major components that create NCCG.

   - Parser: NCCG model is a feed-forward neural network that generates probability over actions given parsing configuration.
     For technical details, please see Section 3 in the [paper](http://www.cs.cornell.edu/~dkm/papers/ma-emnlp.2016.pdf).
     The model file is located in:
         ./edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser/NeuralDotProductShiftReduceParser.java
         
   - Creating dataset: NCCG parser is trained on configuration and gold action pairs that are generated using a CKY parser. 
     For technical details, please see Section 4 in the [paper](http://www.cs.cornell.edu/~dkm/papers/ma-emnlp.2016.pdf).
     This is described in the following package:
         ./edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset
         
   - Learning: NCCG is trained using backpropagation. For technical details, please see Section 4 in the [paper](http://www.cs.cornell.edu/~dkm/papers/ma-emnlp.2016.pdf). This is described in the following file:
         ./edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.learner/NeuralFeedForwardDotProductLearner.java

## AMR Parsing using Neural Shift Reduce CCG Semantic parser (NCCG)



## Build the model or use a saved model

   

## Run the code
