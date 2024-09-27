import pandas as pd
import sklearn
import nltk 
import numpy as np


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay


##Classifiers: 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


import matplotlib.pyplot as plt

##UI:
from tqdm import tqdm


PROMPT = f"""
You are a dataset generator responsible 
for generating facts about Vancouver, British Columbia, 
that are either true or false. 
This is meant to train a linear classifier
into detecting fake facts. 

EXAMPLE FACT:

The Port of Vancouver is the largest port in Canada 
and the third largest in 
North America in terms of total tonnage moved in and out of the port.

The cosmetic treatment Botox was invented in Vancouver.

EXAMPLE FAKE:

Vancouver is home to some of the largest tornadoes in the world.

Generate 100 samples of FACT and 100 samples of FAKE. 
You can output them in 
respective text lists.
"""



'''
filepath_facts, filepath_fakes assumes that it is a txt file, each line contains an element. Each element is not a string
and is delimited by a newline.
Returns the Facts, Fakes in a dataframe, with columns [sentence, fact].
NOTE: The .txt files are already utf-8 encoded.
'''
def aquire(filepath_facts: str, filepath_fakes: str):
    file_df = pd.DataFrame()

    file = open(filepath_facts, "r")
    for element in file:
        df_el = {'sentence': element.strip(), 'fact': True}
        file_df = file_df._append(df_el, ignore_index = True)
    
    file.close()

    file = open(filepath_fakes, "r")
    for element in file: 
        df_el = {'sentence': element.strip(), 'fact': False}
        file_df = file_df._append(df_el, ignore_index = True)
    
    file.close()

    return file_df


'''
Requires the nltk download.
returns in order: stop_words, lemmatizer, stemmer
'''
def download_and_get_nltk_tools():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    
    #Create tools
    stop_words = stopwords.words('english')
    lemmatize = WordNetLemmatizer()
    stemmer = PorterStemmer()

    return stop_words, lemmatize, stemmer

'''
This function is in charge of text cleaning each sentence in the dataframe. We use here to preprocessing techiniquese in NLP: Stemming and Lemmatization. 
This is done such that the counter (TF-IDF) records the abstraction of the word. Hopefully better in the downstream task that we have. 
Additionally, we remove stop words that are uncessary for our classification i.e ('it's, the, a, etc.)
'''
def text_cleaning(sentence: str, stop_words, stemmer, lemmatizer, use_stemming_lemmatization: True, remove_stop_words: True): 
    ## Remove stop words
    filtered_sentence = []
    sentence = sentence.lower()
    stop_words = set(stop_words)
    word_tokens = word_tokenize(sentence)

    for w in word_tokens:
        if remove_stop_words:
            if w not in stop_words:
                ## Lemmatize & Stem
                if use_stemming_lemmatization:
                    w = stemmer.stem(w)
                    w = lemmatizer.lemmatize(w) 
                filtered_sentence.append(w)
        else:
            if use_stemming_lemmatization:
                w = stemmer.stem(w)
                w = lemmatizer.lemmatize(w)
            filtered_sentence.append(w)
    

    sentence_filtered = ' '.join(filtered_sentence)

    return sentence_filtered

'''
Assumes that the file_dataframe has columns ['sentence', 'true']. Thus, converting the collenction of sentences to a matrix of TF-IDF features.
We then convert the output into a pandas dataframe for ease of visualization.
NOTE: We do ngrams (1-3) and TF-IDF counts as our feature extraction
'''    
def feature_extraction(file_dataframe):
    vectorizer = TfidfVectorizer(encoding='utf-8')
    ngram_vectorizer = CountVectorizer(ngram_range=(1,3))
    X_TFID = vectorizer.fit_transform(file_dataframe['sentence'])
    X_ngram = ngram_vectorizer.fit_transform(file_dataframe["sentence"])

    ##Convert the sparse numpy matrix to pandas DataFrame
    ## Then concat to the same dataframe
    X_df_TFID = pd.DataFrame(X_TFID.toarray(), columns=vectorizer.get_feature_names_out())
    X_df_ngram = pd.DataFrame(X_ngram.toarray(), columns=ngram_vectorizer.get_feature_names_out())
    X_concat = pd.concat([X_df_TFID, X_df_ngram], axis=1)


    
    X_concat['fact'] = file_dataframe['fact']
    
    return X_concat



'''
NOTE: We are using cross-validation here once for the hyperparamter tuning and another for the output scores. 
This is not necessary, however the output from doing cross-validation directly is nicer, and thus I do it twice. 
'''
def lr_classifier(scoring, X_train, y_train, X_test, y_test, perform_feature_selection: True):
    ##Before we do any classification we will do some feature selection first
    ##FEATURE SELECTION: We use recursive feature elimination with cross-validation. Thus, the function uses a validation set in deciding the best features. (NOT TEST SET)

    if perform_feature_selection:
        print("=== Recursive Feature Elimination ===")
        rfe_cv = RFECV(LogisticRegression(max_iter=1000), cv = 5)
        rfe_cv.fit(X_train, y_train)
        print("=== Support Column found ===")
        print(rfe_cv.support_)
        print(X_train.loc[:, rfe_cv.support_])
        X_train = X_train.loc[:, rfe_cv.support_] ##Remove all features not in feature selection
        X_test = X_test.loc[:, rfe_cv.support_] ##Remove all features in test set for equal dimension
    else:
        print("=== Skipping Feature Selection ===")
    
    
    
    ##Cross-validate score and hyperparameter tuning
    ##We do a grid search to do hyperparamater tuning (This can probably be better extended with random search)
    pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


    param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 1000]}
    grid_search = GridSearchCV(pipe_lr, param_grid=param_grid, cv = 5,
                               n_jobs=-1, return_train_score=True)
    grid_search.fit(X_train, y_train)
    print("=== Logistic Regression Best Parameters ===")
    print(grid_search.best_params_)


    
    pipe_lr_optimized = make_pipeline(StandardScaler(), LogisticRegression(C = grid_search.best_params_['logisticregression__C']))
    scores_lr = cross_validate(pipe_lr_optimized, X_train, y_train, return_train_score=True, scoring=scoring)
    
    ##Here we display the Confusion Matrix
    pipe_lr_optimized.fit(X_train, y_train)

    disp = ConfusionMatrixDisplay.from_estimator(
        pipe_lr_optimized, 
        X_test, 
        y_test,
        display_labels=["true", "false"],
        values_format="d",
        cmap=plt.cm.Blues,
        colorbar=False,)
    
    disp.ax_.set_title("Logistic Regression CM" + " FS:" + str(perform_feature_selection))
    
    score_dataframe = pd.DataFrame(scores_lr).mean()
    ## Do the final test score
    test_score = pipe_lr_optimized.score(X_test, y_test)
    score_dataframe['final_test_set_score'] = test_score 

    return score_dataframe
    

'''
NOTE: We do not use gridsearch here for hyperparameter tuning. As for this classifier hyperparamaters from sci-kit learn are minimal. Spcificng priors, for example, will 
also remove the priors calcualted on the given data. We do not want this. 
'''
def naive_classifier(scoring, X_train, y_train, X_test, y_test, perform_feature_selection: True):
    ##Feature Selection: (NOTE: For perumation importance we have to train first)
    ## Thus, we use train_test_split again to create a validation set
    if perform_feature_selection:
        print("=== Perumation importance ===")
        X_train_feature_use, X_val, y_train_feature_use, y_val = train_test_split(X_train, y_train, test_size=0.2)
        gaussian_nb = GaussianNB()
        gaussian_nb.fit(X_train_feature_use, y_train_feature_use)
        imps = permutation_importance(gaussian_nb, X_val, y_val)
        boolean_mask = np.array(imps.importances_mean, dtype = bool)
        X_train = X_train.loc[:, boolean_mask] ## Remove all features not deemed viable by the permutaion importance
        X_test = X_test.loc[:, boolean_mask]
    else:
        print("=== Feature Selection Skipped ===")

    
    ##Cross validation scoring:
    pipe_naive = make_pipeline(StandardScaler(), GaussianNB())

    scores_nb = cross_validate(pipe_naive, X_train, y_train, return_train_score=True, scoring=scoring)

    ##Display the Confusion Matrix
    pipe_naive.fit(X_train, y_train)

    
    disp = ConfusionMatrixDisplay.from_estimator(
        pipe_naive, 
        X_test, 
        y_test, 
        display_labels=["true", "false"],
        values_format="d",
        cmap=plt.cm.Blues,
        colorbar=False,
    )
    
    disp.ax_.set_title("Naive Bayes CM, " + "FS: " + str(perform_feature_selection))

    score_dataframe = pd.DataFrame(scores_nb).mean()
    ## Do the final test score
    test_score = pipe_naive.score(X_test, y_test)
    score_dataframe['final_test_set_score'] = test_score 

    return score_dataframe

    
def svc_classifier(scoring, X_train, y_train, X_test, y_test, perform_feature_selection: True):
    ##Feature Selection: NOTE: There is not convient feature selection algorithm that affords the complexity
    ## of recursive feature elimination. Thus, we will use RFE on logisitic regression and use those features on SVC

    if perform_feature_selection:
        print("=== Recursive Feature Elimination ===")
        rfe_cv = RFECV(LogisticRegression(max_iter=1000), cv = 5)
        rfe_cv.fit(X_train, y_train)
        print("=== Support Column found ===")
        print(rfe_cv.support_)
        print(X_train.loc[:, rfe_cv.support_])
        X_train = X_train.loc[:, rfe_cv.support_] ##Remove all features not in feature selection
        X_test = X_test.loc[:, rfe_cv.support_]
    else: 
        print("=== Skipping Feature Selection ===")
    
    pipe_svm = make_pipeline(StandardScaler(), SVC())

    param_grid = {
        "svc__gamma": [0.001, 0.01, 0.1, 1.0, 10, 100],
        "svc__C": [0.001, 0.01, 0.1, 1.0, 10, 100],
    }

    grid_search = GridSearchCV(
        pipe_svm, 
        param_grid=param_grid,
        cv = 5, # A validation set is created within the cross-folds part of this algorithm.
        n_jobs=-1,
        return_train_score=True, 
    )

    grid_search.fit(X_train, y_train)


    pipe_svm_optimized = make_pipeline(StandardScaler(), SVC(C = grid_search.best_params_['svc__C'], gamma= 
                                                             grid_search.best_params_['svc__gamma']))
    scores_svm = cross_validate(pipe_svm_optimized, X_train, y_train, return_train_score=True,scoring=scoring)

    ##Confusion Matrix on test set
    pipe_svm_optimized.fit(X_train, y_train)

    disp = ConfusionMatrixDisplay.from_estimator(
        pipe_svm_optimized,
        X_test,
        y_test,
        display_labels=["true", "false"], 
        values_format="d",
        cmap=plt.cm.Blues,
        colorbar=False,
    )
    
    disp.ax_.set_title("Support Vector Machine CM, " + "FS: " + str(perform_feature_selection))

    score_dataframe = pd.DataFrame(scores_svm).mean()
    ## Do the final test score
    test_score = pipe_svm_optimized.score(X_test, y_test)
    score_dataframe['final_test_set_score'] = test_score 

    return score_dataframe



def main():
    ## We consider three different cases:
    ## 1. No Pre-processing, No FeatureSelection
    ## 2. Pre-processing, No FeatureSelection
    ## 3. No Pre-processing, FeatureSelection
    ## 4. Pre,procssing, FeatureSelection
    scoring = ["accuracy",
           "f1", 
           "recall",
           "precision"]
    
    file_df = aquire('./facts.txt', './fake.txt')

    ##Parameters we consider
    hyperparameters = [{
        "use_stemming_lematization": False, 
        "remove_stop_words": False,
        "perform_feature_selection": False
    }, {
        "use_stemming_lematization": True,
        "remove_stop_words": True,
        "perform_feature_selection": False
        },{
        "use_stemming_lematization": False,
        "remove_stop_words": False,
        "perform_feature_selection": True
        },
    {
        "use_stemming_lematization": True,
        "remove_stop_words": True,
        "perform_feature_selection": True
        }]
    stop_words, lemmatizer, stemmer = download_and_get_nltk_tools()
    for paramaters in hyperparameters:
        file_df['sentence'] = file_df['sentence'].map(lambda element: text_cleaning(element, stop_words=stop_words, stemmer=stemmer, lemmatizer=lemmatizer, 
                                                                                    use_stemming_lemmatization=paramaters["use_stemming_lematization"], remove_stop_words=paramaters["remove_stop_words"]))
        X_df = feature_extraction(file_dataframe=file_df)
        '''
        NOTE: We are completing a few things here. One we make sure that our test class is of reasonable size. Two, we
        require that we have a random sample for our Train, test sets with a shuffle such that we do not have an imbalance in sampling.
        '''
        train, test = train_test_split(X_df, test_size=0.2, random_state=42, shuffle=True)

        X_train = train.drop(columns=["fact"])
        y_train = train['fact']

        X_test = test.drop(columns=["fact"])
        y_test = test["fact"]
        
        
        lr_dict = lr_classifier(scoring=scoring, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                perform_feature_selection=paramaters["perform_feature_selection"])
        nb_dict = naive_classifier(scoring=scoring, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                   perform_feature_selection=paramaters["perform_feature_selection"])
        svc_dict = svc_classifier(scoring=scoring, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                  perform_feature_selection=paramaters["perform_feature_selection"])

        print("==== Logistic Regression: =====")
        print(lr_dict)
        print("==== Naive Bayes: =====")
        print(nb_dict)
        print("==== Support Vector Machines: =====")
        print(svc_dict)

        

        ##Visulization of results
        models = ['logistic regression', 'Naive Bayes', 'SVC']
        test_accuracies = [lr_dict['final_test_set_score'], nb_dict['final_test_set_score'], svc_dict['final_test_set_score']]

        plt.figure(figsize=(8, 6))
        plt.bar(models, test_accuracies, color='skyblue')

        plt.xlabel('Models')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy for Each Model, '+ " pre-processing: " + str(paramaters["use_stemming_lematization"] and paramaters["remove_stop_words"])
                  + " FS: " + str(paramaters["perform_feature_selection"]))
        plt.ylim(0, 1)
        plt.show()


    return 0



if __name__ == "__main__":
    main()