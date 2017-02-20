# Neural Shift Reduce CCG Semantic Parser for AMR Parsing
Contains implementation of Neural Shift Reduce Parser for CCG Semantic Parser of Misra and Artzi EMNLP 2016.

# Author
Developed and maintained by Dipendra Misra (dkm@cs.cornell.edu)

Uses the following external codebase:

1. Cornell SPF and AMR code maintained by Yoav Artzi (Artzi, 2016).
2. [https://deeplearning4j.org/](DeepLearning4j).
2. EasyCCG (Lewis and Steedman, 2014) for CCGBank categories.
3. SMATCH metric (Cai and Knight, 2013).
4. Illinois NER (Ratinov and Roth, 2009)
5. Stanford CoreNLP POS Tagger (Manning et al., 2014)

You don't have to install 1-5 above.

# Pre Requisite

 - Java 8.0.
 
# Section 1: Using Amazon AMI to do AMR Parsing

In this section, we will talk on how to use publically available Amazon AMI to perform parsing on devset.
Later sections describe how to do customized training and testing. First, login to [https://aws.amazon.com/](https://aws.amazon.com/) and use the following AMI:

We will launch a master instance and several worker instance to do test time parsing on the devset. 

 - **Run the master instance**
 
   Launch a master instance using the above AMI and run the following command upon ssh:
   
   ```
   cd /home/ubuntu/nccg/nn-amr-dev/
   java -Xmx60g -jar NeuralAmrParser_Test.jar ./experiment_ff/dev.proxy/dev.proxy.dist.exp
   ```
   
   You can find the log file in ```/home/ubuntu/nccg/nn-amr-dev/experiment_ff/dev.proxy/logs4/global.log```
   It may take some time for the master to start the distributed job when running the code for the first time.

 - **Run the worker instances**
   
   Launch some x number of instances (say 20) using the same public AMI.
      Paste the code below to run when the instance launch. It is a good idea to use spot instances for running workers.
      
   ```#!/bin/bash
   cd /home/ubuntu/nccg/nn-amr-dev/
   screen -L
   java -Xmx110g -jar ./NeuralAmrParserTest.jar ./experiment_ff/worker1/worker.exp hostname=<id>   
    master=<id> masterPort=4444
   ```
   
   Supply the public IP address of master in place of <id>. Above code runs with 110GB RAM which can be changed to any other number within the RAM limit.
  
  The results will be printed in the `dev.proxy/logs4/test.log`. The final number should match the numbers reported in the paper.

# Section 2: Using the source code with Eclipse

Instructions below assume you are using [Eclipse](http://www.eclipse.org/downloads/packages/eclipse-ide-java-developers/keplersr1) which is a powerful java IDE.

## Import the code in Eclipse

- Import all the java projects in nn-amr-dev. 

  - To import projects, first open eclipse and change the workspace folder to the root folder of ./nccg. 
  - Now go to File->Import->General->Existing Projects Into Workspace and select the nn-amd-dev folder in the root.
  - You should see the amr project. Select it. You should now see amr in the project explorer. Ignore any errors for now.
  - Now import all the java projects in nn-ccg-dev in similar fashion.
  - Close the following projects tiny, learn.ubl and learn.weakp (right click on the project and click on Close Project).
  - If you see any error then please see the FAQ section or raise an issue.

## Understanding the code structure

- Neural Shift Reduce CCG Semantic parser (NCCG) is developed on top of SPF (CCG Semantic Parsing Framework). Please see [https://github.com/cornell-lic/spf](SPF documentation) to learn more about SPF.
- The NCCG is contained in the java project `./parser.ccg.ff.shiftreduce`. 
- There are three major components that create NCCG.

  - *Parser*: NCCG model is a feed-forward neural network that generates probability over actions given parsing configuration.
     For technical details, please see Section 3 in the [paper](http://www.cs.cornell.edu/~dkm/papers/ma-emnlp.2016.pdf).
     The model file is located in:
         `./edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser/NeuralDotProductShiftReduceParser.java`
         
  - *Creating dataset*: NCCG parser is trained on configuration and gold action pairs that are generated using a CKY parser. 
     For technical details, please see Section 4 in the [paper](http://www.cs.cornell.edu/~dkm/papers/ma-emnlp.2016.pdf).
     This is described in the following package:
         `./edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset`
         
  - *Learning*: NCCG is trained using backpropagation. For technical details, please see Section 4 in the [paper](http://www.cs.cornell.edu/~dkm/papers/ma-emnlp.2016.pdf). This is described in the following file:
         `./edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.learner/NeuralFeedForwardDotProductLearner.java`

## Section 3: Custom Testing and Training

In order to perform testing or learning with NCCG, you will have to build a jar file.
In this section, we will describe how to do this.

### Build the jar file

To build the jar file do the following:

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

Use the jar file that is created and do testing as described in Section 1. You will have to of course
copy the jar and its library folder to the server and also ensure that worker instances have access to the jar.
This can be done by adding an rsync operation when running workers which copies the jar file from the master
or by creating a new AMI and launching workers using that AMI.

### Perform Learning

TO Come
