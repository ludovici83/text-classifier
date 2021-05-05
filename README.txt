README

The document-classifier is structured in the following way:

1) A script "train.py" contains a function (train) that reads all the texts contained in a the folder and fits a Bag-of-Words + Naive Bayes classifier with the data. The data is split into a train set (66.66% of documents) and a test set (33.33% of documents). The accuracy on the test data is of roughly 93% . At the end of this script a the Naive Bayes classifier and the Count_Vectorizer (needed to transform the vocabulary into numerical features) is stored in disk. This function takes as only argument the folder that contains the category-subfolders (each sub-folder contains texts belonging to a single category and the class-label is the name of the subfolder; although as I mention later...some of the categories seem to be mis-labeled)

2) A second script named "classifier.py" contains the function classifier; taking as input the path to the trained model and count_vectorizer (the output from the first script) and the names of the paths containing the texts that want to be classified. 

3) A main script "main.py". This is the one you need to run to execute the train or classify functions. 

Examples of execution: 
3.1) To run the training script one needs to write in the command prompt: 
python main.py -a train -f <name_of_folder_with_texts>
...in this case it would be:
python main.py -a train -f dataset
3.2) To run the classifying script on a series of texts one needs to write:
python main.py -a classify -m <name_of_trained_model> -t <path_to_text_1> <path_to_text_2>


4) A jupyter notebook named "analysis.ipynb" explaning the thinking behind the decisions (why we are using a bag-of-words model? why a Naive Bayes algorithm for classification?, things like that)
