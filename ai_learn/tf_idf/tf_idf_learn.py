from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
def test_CountVectorizer():

    # 1. give a simple dataset
    simple_train = ["don't call you tonight", "Call me isn't a cab", 'please call me PLEASE!']
    # 2. to conduct a CountVectorizer object
    cv = CountVectorizer()
    #cv = CountVectorizer()
    cv.fit(simple_train)
    # to print the vocabulary of the simple_train
    print(cv.vocabulary_)
    # 3. 4.transform training data into a 'document-term matrix' (which is a sparse matrix) use “transform()”
    train_data = cv.transform(simple_train)
    # (the index of the list , the index of the dict ) the frequency of the list[index]
    print(cv.get_feature_names_out())
    print(train_data)

    train_data = train_data.toarray()
    print(train_data)

    # 7. transform testing data into a document-term matrix (using existing vocabulary)
    simple_test = ["please don't call me"]
    test_data = cv.transform(simple_test).toarray()
    # 8. examine the vocabulary and document-term matrix together
    print(test_data)

def test_tfidf_filter_vec():
    simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']
    cv = TfidfVectorizer()
    cv.fit(simple_train)
    print(cv.vocabulary_)
    train_data = cv.transform(simple_train)
    print(cv.get_feature_names_out())
    print(train_data)

    train_data = train_data.toarray()
    print(train_data)


if __name__ == '__main__':
    test_CountVectorizer()
    # test_tfidf_filter_vec()

