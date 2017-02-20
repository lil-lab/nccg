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


  - *Parser*: NCCG model is a feed-forward neural network that generates probability over actions given parsing configuration.
     For technical details, please see Section 3 in the [paper](http://www.cs.cornell.edu/~dkm/papers/ma-emnlp.2016.pdf).
     The model file is located in:
         ./edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser/NeuralDotProductShiftReduceParser.java
         
  - *Creating dataset*: NCCG parser is trained on configuration and gold action pairs that are generated using a CKY parser. 
     For technical details, please see Section 4 in the [paper](http://www.cs.cornell.edu/~dkm/papers/ma-emnlp.2016.pdf).
     This is described in the following package:
         ./edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset
         
  - *Learning*: NCCG is trained using backpropagation. For technical details, please see Section 4 in the [paper](http://www.cs.cornell.edu/~dkm/papers/ma-emnlp.2016.pdf). This is described in the following file:
         ./edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.learner/NeuralFeedForwardDotProductLearner.java

## AMR Parsing using Neural Shift Reduce CCG Semantic parser (NCCG)

In order to perform testing or learning with NCCG, you will have to build a jar file.
In this section, we will describe how to do this.

### Build the jar file

The main entry point for NCCG is the file BLAH. To build the jar file do the following:

1. Right click on `amr.neural/src/edu.uw.cs.lil.amr.neural.exp/AmrNeuralGenericExp.java`
   and right click on `export-->java-->Runnable jar file`.
   Give a file name to the jar such as NeuralAMRParsing.jar

   Select the following option for Library Handling: Copy required libraries into a sub-folder next to the generated JAR

2. You should now see a generated JAR file :) 

### Text Interface

1. AMR package comes with a clean text interface that allows one to modify hyperparamter, file names etc.
   without having to rebuild the JAR file. The text interface is defined in `nccg/nn-amr-dev/experiment_ff`
  
2. There are two kind of files-- one with .exp extension and one with .inc extension.
   .exp files are the experiment files that define the chief experiment setup and .inc are support
   files that define indvidual component.
   
3. `nccg/nn-amr-dev/experiment_ff/dev.proxy/dev.proxy.exp` defines one such experimental setup. 
   The exp file defines several variables e.g., globalLog=logs1/global.log defines the location of global log file.
   `dev.proxy.inc` defines other components such as parser and learning modules. `params.inc` defines
   several hyperparameter such as number of epochs.
   
4. Finally, `dev.proxy.exp` defines a job.inc which defines the job to run. This job can be a learning job or testing job
   or other user defined job.

### Perform Testing
In this section, we will talk on how to use a saved model and perform parsing on devset.

1. To perform testing, we will define a job as 


2. We will assume that we are performing distributed testing on Amazon AWS (you can do testing on single machine
   but it will take much longer). For ease, we have supplied the public AMI for NCCG given below: 
   
   Run the following command on an Amazon EC2.

 - Run the master instance
 
   1. Launch a master instance with >60GB RAM. Run the following command upon ssh:
   
   ```cd /home/ubuntu/nccg/nn-amr-dev/
   java -Xmx110g -jar ./NeuralAmrParserTest.jar ./experiment_ff/dev.proxy/dev.proxy.dist.exp
   ```

 - Run the worker instance
   
   1. Launch some x number of instances (say 20). Make sure these instances have the NCCG code.
      Paste the code below to run when the instance launch.
      
   ```#!/bin/bash
   cd /home/ubuntu/nccg/nn-amr-dev/
   screen -L
   java -Xmx110g -jar ./NeuralAmrParserTest.jar ./experiment_ff/worker1/worker.exp hostname=<id>   
    master=<id> masterPort=4444
   ```
   
   Supply the public IP address of master in place of <id>.
   
 2. On launching these instances, you can check the log in master. The final results will be printed in the log of master.
    These should match the numbers reported in the main paper upon completion.
 
### Perform Learning
