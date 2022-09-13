-the required packages for the code are specified in requirements.txt
-as the dataset is sourced at the run time from the following link, access to internet is a requirement: 
https://alt.qcri.org/semeval2016/task6/data/uploads/semeval2016-task6-trainingdata.txt

-total runtime: 30-45 minutes, 
getting the BERT embeddings for the entire test & train data has been most intensive part taking taking out 15-20 minutes on Colab.

-as the model is evaluated only on 1 dataset no arguments are required
-running the command 
'python main.py' 
would load the dataset, train the model and then display the training progress, testing performance,
class wise F1-scores and confusion matrix for the test data in order without the need for any external prompts.




