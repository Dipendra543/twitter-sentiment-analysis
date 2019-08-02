import random
import pickle
from os import path
from statistics import mode

import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def labels(self):
        pass

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


def show_properties(corpus_name, show_fileid_details=False, file_id=None):
    try:
        print("Categories: ", corpus_name.categories())
        print("File Identifiers: ", corpus_name.fileids())

    except Exception as e:
        print(str(e))

    if show_fileid_details:
        if file_id is not None:
            get_file_id = corpus_name.fileids()[file_id]
            print("get_file_id value: ", get_file_id)
            print(list(corpus_name.words(get_file_id)))
        else:
            get_file_id = corpus_name.fileids()
            for each_file_id in get_file_id:
                # corpus_name.words(each_file_id)
                print(each_file_id)


def freq_dist(show_most_common=False, most_common_number=15, specific_word=None):
    if path.exists("pickled_alogs/documents.pickle"):
        with open("pickled_algos/documents.pickle", "rb") as f:
            documents = pickle.load(f)
    if path.exists("pickled_alogs/all_words.pickle"):
        with open("pickled_alogs/all_words.pickle", "rb") as f:
            all_words = pickle.load(f)
    else:
        allowed_word_types = ["J", "R", "V"]
        stop_words = nltk.corpus.stopwords.words('english')
        additional = [",", ".", "'", '"', "-", "?", ":", ')', '(', ]
        for each_word in additional:
            stop_words.append(each_word)
        with open("positive.txt", "r") as f:
            short_pos = f.read()

        with open("negative.txt", "r") as f:
            short_neg = f.read()

        documents = []
        all_words = []

        for each_review in short_pos.split("\n"):
            documents.append((each_review, "pos"))  # documents consists of list of tuples with (review,category)
            short_pos_words = word_tokenize(each_review)
            pos_tags = nltk.pos_tag(short_pos_words)
            for each_word in pos_tags:
                if each_word[1][0] in allowed_word_types:
                    all_words.append(each_word[0].lower())

        for each_review in short_neg.split("\n"):
            documents.append((each_review, "neg"))
            short_neg_words = word_tokenize(each_review)
            pos_tags = nltk.pos_tag(short_neg_words)
            for each_word in pos_tags:
                if each_word[1][0] in allowed_word_types:
                    all_words.append(each_word[0].lower())

        # Saving documents to pickle file
        save_documents = open("pickled_algos/documents.pickle", "wb")
        pickle.dump(documents, save_documents)
        save_documents.close()

        # Saving all_words to pickle file
        save_all_words = open("pickled_algos/all_words.pickle", "wb")
        pickle.dump(all_words, save_all_words)
        save_all_words.close()

    all_words = nltk.FreqDist(all_words)
    if show_most_common:
        try:
            print(all_words.most_common(most_common_number))
        except Exception as e:
            print(str(e))
    if specific_word is not None:
        try:
            print(f'Count of the word {specific_word} is: ', all_words[specific_word])

        except Exception as e:
            print(str(e))

    return documents, all_words


def find_features_helper(document, all_words=None, features_length=5000, save_features=False):
    if path.exists("pickled_algos/word_features5k.pickle"):
        with open("pickled_algos/word_features5k.pickle", "rb") as f:
            word_features = pickle.load(f)

    else:
        word_features = list(all_words.keys())[:features_length]
        if save_features:
            save_word_features = open("pickled_algos/word_features5k.pickle", "wb")
            pickle.dump(word_features, save_word_features)
            save_word_features.close()
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def find_features(documents, all_words, features_length=5000, print_features=False, save_features=False):
    if path.exists("pickled_algos/feature_sets.pickle"):
        with open("pickled_algos/feature_sets.pickle", "rb") as f:
            feature_sets = pickle.load(f)
    else:

        feature_sets = [(find_features_helper(rev, all_words, features_length, save_features=save_features), category)
                        for (rev, category) in documents]
        if save_features:
            with open("pickled_algos/feature_sets.pickle", "wb") as f:
                pickle.dump(feature_sets, f)
    if print_features:
        print("Printing Feature Set[0]", feature_sets[0])
    random.shuffle(feature_sets)
    return feature_sets


def save_classifier_to_file(classifier_name, classifier, overwrite=False):
    """set overwrite to True if you are working on a completely new dataset to overwrite the newly learned classifier
    """
    full_name = "pickled_algos/" + classifier_name + ".pkl"
    if not path.exists(full_name):
        # print(f"---------Writing Classifier {classifier_name} to Pickle File-------------")
        with open(full_name, "wb") as f:
            pickle.dump(classifier, f)

    else:
        if overwrite:
            with open(full_name, "wb") as f:
                pickle.dump(classifier, f)

        else:
            print(f"File {full_name} Already Exists. Set overwrite to 'True' to overwrite existing ")


def load_classifier(classifier_name):
    full_name = "pickled_algos/" + classifier_name + ".pkl"
    # print(f"------------Reading Classifier {classifier_name} from the Pickle File-----------")
    with open(full_name, "rb") as f:
        return pickle.load(f)


def load_or_train(file_name, training_set=None, save_classifier=False, overwrite=False):
    """
    !!Do not give the file extension .pkl in the file_name, just give the file name!!
    :param file_name: name of the file to load or train, without extension
    :param training_set: training set
    :param save_classifier: if true and file does not exist already, it will save the classifier
    :param overwrite: if True, this will overwrite the existing file if already exists
    :return:
    """
    if path.exists("pickled_algos/" + file_name + ".pkl"):
        return load_classifier(file_name)
    else:
        if file_name == "nb_original_classifier":  # 1
            nb_original_classifier = nltk.NaiveBayesClassifier.train(training_set)
            if save_classifier:
                save_classifier_to_file(file_name, nb_original_classifier, overwrite=overwrite)

            return nb_original_classifier

        if file_name == "svc_classifier":  # 1
            nb_original_classifier = SklearnClassifier(SVC(gamma='auto')).train(training_set)
            if save_classifier:
                save_classifier_to_file(file_name, nb_original_classifier, overwrite=overwrite)

            return nb_original_classifier

        if file_name == "nu_svc_classifier":  # 2
            nu_svc_classifier = SklearnClassifier(NuSVC(gamma='auto')).train(training_set)
            if save_classifier:
                save_classifier_to_file(file_name, nu_svc_classifier, overwrite=overwrite)

            return nu_svc_classifier

        if file_name == "linear_svc_classifier":  # 3
            linear_svc_classifier = SklearnClassifier(LinearSVC()).train(training_set)
            if save_classifier:
                save_classifier_to_file(file_name, linear_svc_classifier, overwrite=overwrite)

            return linear_svc_classifier

        if file_name == "sgd_classifier":  # 4
            sgd_classifier = SklearnClassifier(SGDClassifier()).train(training_set)
            if save_classifier:
                save_classifier_to_file(file_name, sgd_classifier, overwrite=overwrite)

            return sgd_classifier

        if file_name == "mnb_classifier":  # 5
            mnb_classifier = SklearnClassifier(MultinomialNB()).train(training_set)
            if save_classifier:
                save_classifier_to_file(file_name, mnb_classifier, overwrite=overwrite)

            return mnb_classifier

        if file_name == "bnb_classifier":  # 6
            bnb_classifier = SklearnClassifier(BernoulliNB()).train(training_set)
            if save_classifier:
                save_classifier_to_file(file_name, bnb_classifier, overwrite=overwrite)

            return bnb_classifier

        if file_name == "logistic_regression_classifier":  # 7
            logistic_regression_classifier = SklearnClassifier(
                LogisticRegression(solver='lbfgs', max_iter=300)).train(training_set)
            if save_classifier:
                save_classifier_to_file(file_name, logistic_regression_classifier, overwrite=overwrite)

            return logistic_regression_classifier


def classify(feature_sets, most_informative=15, save_classifier=False, show_most_informative=False,
             voted_classifier=False, overwrite=False):
    training_set = feature_sets[:10000]  # 10000 data for training
    testing_set = feature_sets[10000:]  # Remaining data is 662 for testing

    #  Original Naive Bayes Classifier
    nb_original_classifier = load_or_train("nb_original_classifier", training_set,
                                           save_classifier=save_classifier, overwrite=overwrite)
    accuracy = nltk.classify.accuracy(nb_original_classifier, testing_set) * 100
    print("Original Navie Bayes Accuracy: ", accuracy, "%")
    if show_most_informative:
        nb_original_classifier.show_most_informative_features(most_informative)

    # MultinomialNB Classifier
    mnb_classifier = load_or_train("mnb_classifier", training_set, save_classifier=save_classifier, overwrite=overwrite)
    accuracy_mnb = nltk.classify.accuracy(mnb_classifier, testing_set) * 100
    print("MNB_Classifier Accuracy: ", accuracy_mnb, "%", "\n")

    # GaussianNB Classifier
    # gnb_classifier = load_or_train("gnb_classifier", training_set, save_classifier=save_classifier)
    # accuracy_gnb = nltk.classify.accuracy(gnb_classifier, testing_set) * 100
    # print("GNB_Classifier Accuracy: ", accuracy_gnb, "%", "\n")

    # BernoulliNB Classifier
    bnb_classifier = load_or_train("bnb_classifier", training_set, save_classifier=save_classifier, overwrite=overwrite)
    accuracy_bnb = nltk.classify.accuracy(bnb_classifier, testing_set) * 100
    print("BNB_Classifier Accuracy: ", accuracy_bnb, "%", "\n")

    #  Logistic Regression Classifier
    logistic_regression_classifier = load_or_train("logistic_regression_classifier",
                                                   training_set, save_classifier=save_classifier, overwrite=overwrite)
    accuracy_lr = nltk.classify.accuracy(logistic_regression_classifier, testing_set) * 100
    print("logistic_regression_classifier Accuracy: ", accuracy_lr, "%", "\n")

    # SGD Classifier
    sgd_classifier = load_or_train("sgd_classifier", training_set, save_classifier=save_classifier, overwrite=overwrite)
    accuracy_sgd = nltk.classify.accuracy(sgd_classifier, testing_set) * 100
    print("sgd_classifier Accuracy: ", accuracy_sgd, "%", "\n")

    # SVC Classifier
    # Not used because of low accuracy
    # svc_classifier = load_or_train("svc_classifier", training_set, save_classifier=save_classifier,
    # overwrite=overwrite)
    # accuracy_svc = nltk.classify.accuracy(svc_classifier, testing_set) * 100
    # print("svc_classifier Accuracy: ", accuracy_svc, "%", "\n")

    # Linear SVC Classifier
    linear_svc_classifier = load_or_train("linear_svc_classifier", training_set, save_classifier=save_classifier,
                                          overwrite=overwrite)
    accuracy_linear_svc = nltk.classify.accuracy(linear_svc_classifier, testing_set) * 100
    print("linear_svc_classifier Accuracy: ", accuracy_linear_svc, "%", "\n")

    # NU_SVC Classifier
    nu_svc_classifier = load_or_train("nu_svc_classifier", training_set, save_classifier=save_classifier,
                                      overwrite=overwrite)
    accuracy_nu_svc = nltk.classify.accuracy(nu_svc_classifier, testing_set) * 100
    print("nu_svc_classifier Accuracy: ", accuracy_nu_svc, "%", "\n")

    if voted_classifier:
        voted_classifier = VoteClassifier(nb_original_classifier,
                                          nu_svc_classifier,
                                          linear_svc_classifier,
                                          sgd_classifier,
                                          mnb_classifier,
                                          bnb_classifier,
                                          logistic_regression_classifier)

    print("voted_classifier accuracy percent:",
          (nltk.classify.accuracy(voted_classifier, testing_set)) * 100, "%", "\n")


def find_sentiment(text):
    feats = find_features_helper(text)
    nb_original_classifier = load_or_train("nb_original_classifier")
    mnb_classifier = load_or_train("mnb_classifier")
    bnb_classifier = load_or_train("bnb_classifier")
    logistic_regression_classifier = load_or_train("logistic_regression_classifier")
    sgd_classifier = load_or_train("sgd_classifier")
    # svc_classifier = load_or_train("svc_classifier")
    linear_svc_classifier = load_or_train("linear_svc_classifier")
    nu_svc_classifier = load_or_train("nu_svc_classifier")

    voted_classifier = VoteClassifier(nb_original_classifier,
                                      nu_svc_classifier,
                                      linear_svc_classifier,
                                      sgd_classifier,
                                      mnb_classifier,
                                      bnb_classifier,
                                      logistic_regression_classifier)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)


# if __name__ == '__main__':
#     # show_properties(movie_reviews, show_fileid_details=True, file_id=0)
#     # mr_doc, mr_all_words = freq_dist(show_most_common=True)
#     # mr_features = find_features(mr_doc, mr_all_words, print_features=False, save_features=True)
#
#     # In classify function, set overwrite to True only if you are dealing with new dataset or have changed
#     # the features and you need to overwrite the existing pickle classifier files
#     # classify(mr_features, save_classifier=True, voted_classifier=False, overwrite=False)
#     print(find_sentiment("This movie was awesome! The acting was great, "
#                          "plot was wonderful, and there were pythons...so yea!"))
#     print(find_sentiment("This movie is fantastic!"))
#     # print(nltk.pos_tag(nltk.word_tokenize("This movie is not good")))
