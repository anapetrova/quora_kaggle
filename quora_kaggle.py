import numpy as np
import string
from sklearn.svm import LinearSVC
import csv
from nltk.corpus import stopwords

punctuation = set(string.punctuation)
stopwords = stopwords.words("english")

def read_file(f):
    '''
    Read CSV file in, return last 3 elements of the row
    '''
    with open(f, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        csvreader.next()
        for row in csvreader: # why unicode?
            yield row[-3:]

def write_file(test_f, write_f, model):
    '''
    Write Kaggle test file
    '''
    with open(test_f, 'r') as test, open(write_f, 'w') as write:
        csvreader = csv.reader(test)
        csvreader.next()
        write.write('test_id,is_duplicate\n')
        for row in csvreader:
            feature_vector = np.asmatrix(text_feature(row[1], row[2]))
            result = model.predict(feature_vector)[0]
            write.write('{},{}\n'.format(row[0], result))

def get_words(sentence):
    '''
    Returns uncapitalized words given a sentence
    '''
    try:
        words = [w for w in ''.join([c for c in sentence.lower() if not c in punctuation]).split()]
        return words
    except AttributeError: # no sentence
        return []

def text_feature(q1, q2):
    '''
    Extract text features
    Input:
        q1, q2: Question strings
    Output:
        Feature vector
    '''
    # Number of shared ngrams..?
    features = []
    q1_words = get_words(q1)
    q2_words = get_words(q2)
    q1_set = set(q1_words)
    q2_set = set(q2_words)
    q1_bigrams = set(zip(q1_words, q1_words[1:]))
    q2_bigrams = set(zip(q2_words, q2_words[1:]))

    # Unigrams
    features.append(len(q1_set & q2_set)) # Number of shared words
    features.append(len(q1_set ^ q2_set)) # Number of different words

    # Bigrams
    features.append(len(q1_bigrams & q2_bigrams))
    features.append(len(q1_bigrams ^ q2_bigrams))

    features.append(abs(len(q1) - len(q2))) # Absolute difference in length
    return features

# Data processing and feature extraction
data = list(read_file('train.csv'))
X = [text_feature(row[0], row[1]) for row in data]
y = [int(row[2]) for row in data]
X = np.asmatrix(X)
y = np.asarray(y)
X_train, y_train = X[:len(X)/2], y[:len(y)/2]
X_val, y_val = X[len(X)/2:], y[len(y)/2:]

# Train and score model
model = LinearSVC(C=1)
model.fit(X_train, y_train)
model.score(X_val, y_val)
print "Score: {}".format(model.score(X_val, y_val))

write_file('test.csv', 'submission.csv', model)
