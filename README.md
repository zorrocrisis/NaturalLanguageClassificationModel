## **Natural Language Classification Model**
This project, originally an evaluation component for the Natural Language course (2023/2024), talking place in Instituto Superior Técnico, University of Lisbon, aimed to **simulate a participation in an evaluation forum** (e.g.: CLEF, SemEval, etc.), in which participants test their systems in specific tasks and in the same test sets. More specifically, this project is about **distinguishing between truthful and deceptive hotel reviews, and additionally determining their polarity (positive vs. negative)** - being given a file with a list of N reviews, this system returns another file with N predicted labels (TRUTHFULPOSITIVE, TRUTHFULNEGATIVE, DECEPTIVEPOSITIVE, or DECEPTIVENEGATIVE).

(geneic image of reviews??)
![pathfinding](https://github.com/user-attachments/assets/5d6efe70-b4eb-4c56-9da0-1b2ae98aad88)

The following document indicates how to access and utilise the source code. It also contains a brief analysis of the implementation and results, referring to the [official report]() for more detailed information.

## **Quick Start**
The project's source files can be downloaded from this repository. To open the program using Unity (v.2021.3.10f1), simply clone the repository and open the project utilising Unity Hub.

## **Introduction**
You will be given a training set (train.txt) in which each line has the following format (notice that there
is a tab between the label and the review):
label review
Example:
TRUTHFUL-POSITIVE The sheraton was a wonderful hotel! When me and my mom flew in we were
really tired so we decided to take a quick nap. We didnt want to get up! The beds are absolutely to die
for. I wanted to take it home with me. The service was great and this was probably one of the biggest if
not the biggest hotel ive ever stayed in. They had a really nice restaurant inside with excellent food.
You will also be given a test set (test_just_reviews.txt) in which each line has the format (no labels):
review
During the development of your project, you should create your own test(s) set(s) to evaluate your
models. In the paper you should report the results on your own test(s) set(s). However, for the
automatic evaluation of your project, you should run your best model on the given test set (notice that
it has no labels – test_just_reviews.txt) and return an output file (named results.txt), in which each line
has the format:
labelNotice that the line number in which the review appears in the test file should be the same line
number of the corresponding label in the results.txt (the automatic evaluation depends on this).
To build your model(s), you can use whatever you want, including taking advantage of code already
available (and we strongly advise you to do so), as long as you identify the source. The only constraint
is: you should implement your model in Python 3.

## **Model's Final Performance**
Automatic Evaluation (5 points):
• Accuracy will be the evaluation measure.
• If you beat a weak baseline (Jaccard) that results in an accuracy of 58.5% (on test_just_reviews.txt)
you will have 2.5 points.
• If you beat a stronger baseline, based on a Support Vector Classifier and a tf-idf that results in an
accuracy of 88.0% (on test_just_reviews.txt) you will have extra 2.5 points.


## **Analysis Overview**


## **Authors and Acknowledgements**

This project was developed by **[Miguel Belbute (zorrocrisis)](https://github.com/zorrocrisis)** with contributions from Guilherme Serpa and Peng Li.
The initial code was supplied by **[Prof. Pedro dos Santos](https://fenix.tecnico.ulisboa.pt/homepage/ist12886)**
