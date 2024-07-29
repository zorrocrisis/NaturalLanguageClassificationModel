## **Natural Language Classification Model**
This project, originally an evaluation component for the Natural Language course (2023/2024), talking place in Instituto Superior Técnico, University of Lisbon, aimed to **simulate a participation in an evaluation forum** (e.g.: CLEF, SemEval, etc.), in which participants test their systems in specific tasks and in the same test sets. In further detail, this project is about **distinguishing between truthful and deceptive hotel reviews, additionally determining their polarity (positive vs. negative)** - being given a file with a list of N reviews, this system returns another file with N predicted labels (TRUTHFULPOSITIVE, TRUTHFULNEGATIVE, DECEPTIVEPOSITIVE, or DECEPTIVENEGATIVE).

<p align="center">
  <img src="https://github.com/user-attachments/assets/970eb12e-2859-479c-89a4-824c4d121b0e"/>
</p>

The following document indicates how to access and utilise the source code. It also contains a brief analysis of the implementation and results, referring to the [official report](https://github.com/zorrocrisis/NaturalLanguageClassificationModel/blob/main/FinalReport.pdf) for more detailed information.

## **Quick Start**
The project's source files can be downloaded from this repository. To open the program using Unity (v.2021.3.10f1), simply clone the repository and open the project utilising Unity Hub.

## **Task Introduction**
- The **training set** (*train.txt*), supplied at the start of the poject, **contains multiple hotel reviews with the corresponding (correct) labels**. An example of these reviews can be found here:

*TRUTHFUL-POSITIVE The sheraton was a wonderful hotel! When me and my mom flew in we were really tired so we decided to take a quick nap. We didnt want to get up! The beds are absolutely to die for. I wanted to take it home with me. The service was great and this was probably one of the biggest if not the biggest hotel ive ever stayed in. They had a really nice restaurant inside with excellent food.*

- A **test set** (*test_just_reviews.txt*) was also supplied at the start of this project. However, unlike the training set, the test set **contains only a review per line, without the respective (correct) labels**. An example of these reviews can be found here:

*My family and I stayed here while we were visiting Chicago. This was a perfect location to many of the things we wanted to do and was easy to get around. It was in a safe neighborhood and was great accomodations. The staff was friendly and went the "extra mile" to make sure we had everything we needed. The beds were comfortable. We would definitely stay here again.*

- For the **automatic evaluation of the project, the best obtained model should be run on the supplied test set, returning an output file (named *results.txt*), in which each line contains both the predicted label and the corresponding review**. The line number in which the review appears in the test file has to be the same line number of the corresponding label in the *results.txt* (the automatic evaluation depended on this).

## **Implementations**
To address the task of **classifying hotel reviews which were labeled with regards to truthfulness and polarity**, resulting in **four possible labels** (TRUTHFULPOSITIVE, TRUTHFULNEGATIVE, DECEPTIVEPOSITIVE and DECEPTIVENEGATIVE), multiple models were developed in Python3.

More specifically, **five implementations** were developed. The **first contains and runs a diverse set of machine learning models to identify a clear baseline to improve upon**. The **second implementation proceeds to fine-tune the best model from the previous implementation**. The **third generates the final labels for the test run with the best achieved model**. The **fourth and fifth implementations contain and run deep learning models**.

## **Machine Learning Models**
Multiple machine learning (ML) classifiers were implemented, namely:

- **Multinomial Naïve Bayes (MNB)**
- **Logistic Regression (LR)**
- **Support Vector Machine (SVM)**
- **Stochastic Gradient Descent (SGD)**

The global pipeline for these models included **data preprocessing, vectorization, training and evaluation**.

#### **Preprocessing**
The preprocessing for the machine learning models was inspired by multiple applications found online (why reinvent the wheel?), involving **word tokenization**, **stop word removal**, **stemming** and **detokenization**.

In the first implementation, the NLTK’s word tokenizer (NLTK is a Natural Language Processing Python library) and the Porter stemmer were employed for all the models. However, the second implementation explored **multiple tokenizers** (e.g.: NLTK’s, Regexp’s, Treebank’s, Whitespace’s, WordPunct’s and TokTok’s) **and stemmers** (e.g.: Porter’s, Lancaster’s, Snowball’s and Regexp’s).

Regarding **stop word removal**, various options were also analysed in detail since this process can heavily impact the performance of a text classification task. After examining the default stop words pertaining to scikit-learn (a ML Python library) and NLTK, a decision was made to edit these lists. Considering one of the goals is to determine the polarity of the reviews, it was opted to **exclude from the stop words “fingerprints” belonging to a communication concept entitled negative language** (for more information, refer to the [full report](https://github.com/zorrocrisis/NaturalLanguageClassificationModel/blob/main/FinalReport.pdf)). This type of communication, often used when a person is plagued with negative emotions, is **characterised by sentences in the negative form** (e.g.: “I don’t recommend. . .”) or **passive-aggressive expressions** (e.g.: “Sure, it was fine but. . .”). Therefore, in some tests, the models would exclude words such as “but”, “no”, “not”, “couldn’t”, “never” and “however” to view the impact on the overall performance.

#### **Main Pipeline**
Pipelines were originated for each classifier, continuining the classification process after the preprocessing. These pipelines were formed by two major components: a **vectorizer** (TF-IDF vectorizer was usually utilised), which converts textual data into a numerical format, and the **classifier itself** (SVM, SGD, LR, and so on).


## **Deep Learning Models**
Two deep learning (DL) approaches were additionally considered:

- **Temporal Convolutional Network (TCN)**, using two different word embedding modes: a static word embedding using pre-trained Word2Vec mode and a random mode
- **Bidirectional Enconder Representations from Transformers (BERT)**

#### **Preprocessing**
For preprocessing data in DL, utilising a **tokenizer was discovered to be sufficient, as it did minimal text cleaning, not damaging each model’s ability to learn**. For the TCN, a tokenizer provided by the preprocessing text library within the Keras framework was exploited. Conversely, for BERT, a BertTokenizer from the transformer library developed by HuggingFace was leveraged, which is purposefully designed for data preprocessing tailored to BERT applications.

#### **Model's Implementation**
A variant of the **TCN embedding random model**, described in a thesis by Raihan et al. (for more information, refer to the [full report](https://github.com/zorrocrisis/NaturalLanguageClassificationModel/blob/main/FinalReport.pdf)), was implemented. The original model in the referenced paper employed a sigmoid activation function in the final layer to accommodate two distinct classes. To suit the requirement for four distinct classes, the **activation function was altered to a softmax**. For the same reason, the **loss function was also changed from a binary cross entropy to a categorical cross entropy**. From the same study conducted by Raihan et al. (for more information, refer to the [full report](https://github.com/zorrocrisis/NaturalLanguageClassificationModel/blob/main/FinalReport.pdf)), **TCN Word2Vec static model** was adapted to accommodate four distinct classes, employing the same modifications mentioned before.

Concerning **BERT**, a tutorial available on the "Into Deep Learning"’s website grounded the implementation's approach and in this case it was not necessary to make any alterations to the model (for more information, refer to the [full report](https://github.com/zorrocrisis/NaturalLanguageClassificationModel/blob/main/FinalReport.pdf)).

## **Experimental Setup and Results**
The **provided dataset**, containing 1400 examples of label-review pairs, **was split to originate a training set - 90% of the original data - and a development set - the remaining 10%. The latter was used to evaluate and fine-tune the various models during training**. Before each training run, the data was randomly shuffled while still maintaining the aforementioned proportions. Finally, the provided test set - 200 reviews without the "gold labels" - was utilized to perform the final evaluation.

To evaluate the models’ performances, **accuracy was highlighted as the primary evaluation metric**. In determining this metric, **the models' predicted labels were compared with the correct ones from the development set**. **Recognizing the potential for significant variance in performance due to arbitrary partitions, a crossvalidation computation with 5 "folds" (or smaller sets) was performed, which provided the mean accuracy and its corresponding standard deviation**. Although not directly evaluating the model, one should note how the suggested implementations also displays the **confusion matrices** related to each run and how the final model prints out the incorrect predictions, facilitating a more in-depth analysis of the results.

With regards to parameters, on the first implementation only the default values were applied to the global pipeline (for more information, refer to the [full report](https://github.com/zorrocrisis/NaturalLanguageClassificationModel/blob/main/FinalReport.pdf)). On the second implementation, **a grid search was performed on a set of defined parameters to fine-tune and improve the overall accuracy of the model which had displayed the best performance so far**. In the realm of DL models, no extensive parameter exploration was conducted. Consequently, the models were evaluated using their default parameters as provided in the original implementations (for more information, refer to the [full report](https://github.com/zorrocrisis/NaturalLanguageClassificationModel/blob/main/FinalReport.pdf)).

For the ML models, the "default" **TF-IDF + SVM pipeline achieved the highest accuracy at 81.36% on the first implementation**. Subsequently, TF-IDF + SVM was chosen to be further analysed - Figure 1 illustrates the impact of different stop words on both the default and a fine-tuned version of the model. Notably, **the fine-tuned TF-IDF + SVM without stop words outperformed all others variations of this model, achieving an average score of 85.50%** - Figure 2 and 3.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4ca849cb-3aeb-4a5b-87f2-77e87e3a789b"/>
</p>

<p align="center">
  <i>Figure 1: TF-IDF and SVM pipeline with default and finetuned parameters - average accuracy and standard deviation with varying stop words</i>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/1f5f769f-864d-4031-9ace-742f9bfe7052"/>
</p>

<p align="center">
  <i>Figure 2: Fine-tuned TF-IDF + SVM (best version) - confusion matrix</i>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/5b9369dd-f394-42e2-841f-8df5201c4713"/>
</p>

<p align="center">
  <i>Figure 3: Fine-tuned TF-IDF + SVM (best version) - accuracy per label</i>
</p>

Considering the DL models: **TCN embedding random achieved 69%, TCN Word2Vec static reached 71.17%, and the BERT model scored 74%**.

## **Discussion**
Regarding the **stop words’ analysis** (Figure 1), despite one verifying mostly positive variations on the average accuracy with the edited lists, **considering the standard deviation identified, one cannot confirm nor deny the impact of eliminating "negative communication" from the stop words**. This result can, however, be influenced by the **biaxial nature of this classification task**: aside from determining polarity, the models also had to consider truthfulness, whose correlation to negative language remains unexplored.

Figure 2’s confusion matrix showcases how **the fine-tuned model is greatly accurate in polarity classification**, with the most frequent errors in all labels corresponding to mislabelling on the truthfulness scale - TRUTHFULPOSITIVE is most often mistaken for DECEPTIVEPOSITIVE, TRUTHFULNEGATIVE is most frequently mistaken for DECEPTIVENEGATIVE and so on. A good example of this is review 690 which, due to its unusual punctuation and structure (e.g.: ",this is a very good place.amazing") often induces the model to mistake it for DECEPTIVE when it is TRUTHFUL. Contrarily, when the predicted polarity is incorrect its often due to a review containing a mixture of good and bad adjectives (e.g.: review 887) or a review whose classification contradicts its content (e.g.: review 1237 contains "Will definitely stay there again!" yet is classified as negative).

Nonetheless, the final model, which outperformed all other models - **fine-tuned TF-IDF + SVM without stop word removal** - **managed to achieve a satisfactory 91% accuracy per labels** (Figure
3)!

The **underperforming of the DL models** is possibly due to their large size and small data set, which poses an **overfitting risk**.

ML models allowed for quick and convenient parameter tuning while DL models were time-consuming and computationally expensive, making parameter exploration more challenging.

## **Future Work**
**To gain deeper insights into the influence of stop words associated with negative language, a specialized investigation focused on their impact solely regarding polarity classification could be conducted**. Furthermore, future research endeavors could concentrate on refining our DL approaches, expanding the dataset, or simplifying network architecture by reducing the number of layers and their complexity.

## **Best Model's Official Performance**
After the best classification model's submission, the automatic evaluation was performed, resulting in a **final obtained accuracy of 89.5%!**

This accuracy is slightly higher than the one obtained in testing (81.36%). Among other hypotheses, this could because, for the final submission, the original supplied dataset was entirely utilised to train the model, instead of having it split into a training and development set, thus giving more data for the model to train with.

## **Authors and Acknowledgements**
This project was developed by **[Miguel Belbute (zorrocrisis)](https://github.com/zorrocrisis)** and [Guilherme Pereira](https://github.com/the-Kob).
The initial code was supplied by **[Prof. Pedro dos Santos](https://fenix.tecnico.ulisboa.pt/homepage/ist12886)**
