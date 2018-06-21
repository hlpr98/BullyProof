import csv
import re
from random import shuffle

import nltk
import pandas as pd
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.decomposition import KernelPCA
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


def train(train_features, train_labels):
    print("Training........\n")
    clf = svm.SVC()
    clf.fit(train_features, train_labels)
    joblib.dump(clf, 'model.pkl')


def test(test_features, test_labels):
    print("Testing........\n")

    clf = joblib.load('model.pkl')
    prediction = clf.predict(test_features)

    file = open('result2.csv', 'w')
    writer = csv.writer(file)
    for i in range(len(prediction)):
        if prediction[i] == test_labels[i]:
            writer.writerows([[prediction[i], test_labels[i]]])
        else:
            writer.writerows([['wrong', test_labels[i]]])

    print(classification_report(test_labels, prediction))
    print("\nAccuracy = ")
    print(accuracy_score(test_labels, prediction))


def find_features(list_of_words, all_words):
    words = set(list_of_words)
    features = []
    for w in all_words:
        features.append((w in words))
    return features


def find_features_tuples(list_of_tuples, all_tuples):
    tuples = set(list_of_tuples)
    features = []
    for w in all_tuples:
        features.append((w in tuples))

    return features


def dimention_reduction_train(features):
    print("Dimenstionality reduction training.........")
    kpca = KernelPCA(n_components=50)
    kpca.fit(features)
    joblib.dump(kpca, 'KernelPCA.pkl')


def dimention_reduction(features):
    print("Reducing dimentions........")
    kpca = joblib.load('KernelPCA.pkl')

    return kpca.transform(features)


if __name__ == '__main__':

    train_feature = []
    train_labels = []
    test_feature = []
    test_labels = []
    labels = []
    features = []

    # dataset = pd.read_csv("train.csv",sep = '\t', delimiter = None);
    # # dataset.pop(index)
    # dataset.pop('userid')
    # dataset.pop('asker')
    # dataset.pop('severity1')
    # dataset.pop('bully1')
    # dataset.pop('severity2')
    # dataset.pop('bully2')
    # dataset.pop('severity3')
    # dataset.pop('bully3')
    # print(dataset);
    # dataset.to_csv('out.csv',sep = ',');

    df = pd.read_csv("out.csv", sep=',', delimiter=None, index_col=['index'])

    array = df.loc[:, ['ans1', 'ans2', 'ans3']].as_matrix();

    for i in range(array.shape[0]):
        if array[i][0] == 'yes' or array[i][1] == 'yes' or array[i][2] == 'yes':
            labels.append(1);
        else:
            labels.append(-1);
        i = i + 1;

    # print(labels);

    df.pop('ans1')
    df.pop('ans3')
    df.pop('ans2')

    data_post_array = []
    data_ques_array = []
    data_ans_array = []

    with open('out.csv') as f:
        data = csv.reader(f, delimiter=',')
        for row in data:
            data_post_array.append(row[1])
            data_ques_array.append(row[2])
            data_ans_array.append(row[3])

    # print(re.sub('[!@#$%,+-^&*~"/|><:]','', data_post_array[1]))

    data_post_array.pop(0);
    data_ans_array.pop(0);
    data_ques_array.pop(0);

    # Try to remove " ' ", not yet done.

    for i in range(len(data_post_array)):
        data_post_array[i] = re.sub('[!@#$%,+-^&*~"/|><:().?]', '', data_post_array[i])
        data_ques_array[i] = re.sub('[!@#$%,+-^&*~"/|><:().?]', '', data_ques_array[i])
        data_ans_array[i] = re.sub('[!@#$%,+-^&*~"/|><:().?]', '', data_ans_array[i])
        i = i + 1

    # print(word_tokenize(data_post_array[2012]))
    # print(data_ans_array[2012])

    stop_words = set(stopwords.words("english"))
    # print(stop_words)

    for i in range(len(data_post_array)):
        post = ''
        for w in word_tokenize(data_post_array[i]):
            if w not in stop_words:
                post = post + w + ' '

        data_post_array[i] = post

    for i in range(len(data_ques_array)):
        post = ''
        for w in word_tokenize(data_ques_array[i]):
            if w not in stop_words:
                post = post + w + ' '

        data_ques_array[i] = post

    for i in range(len(data_ans_array)):
        post = ''
        for w in word_tokenize(data_ans_array[i]):
            if w not in stop_words:
                post = post + w + ' '

        data_ans_array[i] = post

    # print(word_tokenize(data_ques_array[0]))

    trigram_post = []
    unigram_post = []
    for i in range(len(data_post_array)):
        temp1 = []
        temp2 = []
        for w in ngrams(word_tokenize(data_post_array[i]), 3):
            temp1.append(w)
        for w in ngrams(word_tokenize(data_post_array[i]), 0):
            temp2.append(w)
        trigram_post.append(temp1)
        unigram_post.append(temp2)

    # print(unigram_post[0])
    # print(trigram_post[0])

    trigram_ques = []
    unigram_ques = []
    for i in range(len(data_ques_array)):
        temp1 = []
        temp2 = []
        for w in ngrams(word_tokenize(data_ques_array[i]), 3):
            temp1.append(w)
        for w in ngrams(word_tokenize(data_ques_array[i]), 0):
            temp2.append(w)
        trigram_ques.append(temp1)
        unigram_ques.append(temp2)

    trigram_ans = []
    unigram_ans = []
    for i in range(len(data_ans_array)):
        temp1 = []
        temp2 = []
        for w in ngrams(word_tokenize(data_ans_array[i]), 3):
            temp1.append(w)
        for w in ngrams(word_tokenize(data_ans_array[i]), 0):
            temp2.append(w)
        trigram_ans.append(temp1)
        unigram_ans.append(temp2)

    # for w in trigram_post[0]:
    # 	print(w)
    # print(unigram_ans[0][0][0])

    print("Makeing unigrams and trigrams........")

    features2 = []

    # print(trigram_ans[201][0])

    for i in range(len(data_post_array)):
        to_send = []
        to_send2 = []
        for j in range(len(unigram_post[i])):
            to_send.append(unigram_post[i][j][0].lower())

        for j in range(len(unigram_ques[i])):
            to_send.append(unigram_ques[i][j][0].lower())

        for j in range(len(unigram_ans[i])):
            to_send.append(unigram_ans[i][j][0].lower())

        for j in range(len(trigram_post[i])):
            to_send2.append(trigram_post[i][j])

        for j in range(len(trigram_ques[i])):
            to_send2.append(trigram_ques[i][j])

        for j in range(len(trigram_ans[i])):
            to_send2.append(trigram_ans[i][j])

        # features.append(np.array([unigram_post[i],unigram_ques[i],unigram_ans[i],trigram_post[i],trigram_ques[i],trigram_ans[i]],dtype = object))
        features.append(to_send)
        features2.append(to_send2)

    # print(features2[0])
    # print(len(features[201]))
    # print("\n")
    # print(len(features2[201]))

    print("Finding FreqDist......")

    all_words = []
    for i in range(len(features)):
        for j in features[i]:
            all_words.append(j.lower());

    all_tuples = []
    for i in range(len(features2)):
        for j in features2[i]:
            all_tuples.append(j);

    # # print(all_words);
    # print(len(all_words))
    # print(len(all_tuples))

    all_words = nltk.FreqDist(all_words)
    all_tuples = nltk.FreqDist(all_tuples)

    # print(len(all_words))
    # print(len(all_tuples))

    # print(all_words["night"])

    # print(all_words)

    # word_features = list(all_words.keys())[:5000]
    # tuple_features = list(all_tuples.keys())[:10000]

    # joblib.dump(word_features,'words1.pkl')
    # joblib.dump(tuple_features,'tuples1.pkl')

    word_features = joblib.load("words.pkl")
    tuple_features = joblib.load("tuples.pkl")

    # print(tuple_features[0])

    # print(find_features(features[2018],word_features))
    # print(find_features_tuples(features2[2018],tuple_features))

    # feature_set = [(find_features(f,word_features)) for f in features]
    print("Making word and tuples as features.......")

    feature_set = []
    for i in range(len(features)):
        dum = []
        for w in find_features(features[i], word_features):
            dum.append(w)
        for w in find_features_tuples(features2[i], tuple_features):
            dum.append(w)
        # dum.append(1)
        # dum.append(1)
        feature_set.append(dum)

    # print(len(feature_set[2018]))
    # print(len(find_features(features[2018],word_features)))
    # print(len(find_features_tuples(features2[2018],tuple_features)))

    # print(feature_set[2018])
    # print(len(features))
    # print(len(feature_set))

    # vectorizer = TfidfVectorizer(min_df=5,
    #                                 max_df = 0.8,
    #                                 sublinear_tf=True,
    #                                 use_idf=True)
    # train_vectors = vectorizer.fit_transform(features)
    # # test_vectors = vectorizer.transform(test_data)

    # feature_set_to_shuffle = []
    # for i in range(len(feature_set)):
    # 	dum = []
    # 	dum.append(feature_set[i])
    # 	dum.append(labels[i])
    # 	feature_set_to_shuffle.append(dum)

    print("Standardizing.....")

    feature_set = StandardScaler().fit_transform(feature_set)

    print("Shuffling the dataset.......")

    feature_set_to_shuffle = list(zip(feature_set, labels))

    # print(feature_set_to_shuffle[12187])
    shuffle(feature_set_to_shuffle)
    # print(feature_set_to_shuffle[12187])

    feature_set, labels = zip(*feature_set_to_shuffle)

    # train_features = feature_set[:(round(.80 * len(data_post_array)))];
    # test_features = feature_set[(round(.80 * len(data_post_array))):];
    # train_labels = labels[:(round(.80 * len(data_post_array)))];
    # test_labels = labels[(round(.80 * len(data_post_array))):];

    test_features = feature_set;
    test_labels = labels;

    print("Dimenstionality reduction......")
    # pca = KernelPCA(n_components = 20)
    # pca.fit(train_features)

    # pca.fit_transform(train_features)
    # pca.fit_transform(test_features)

    # dimention_reduction_train(train_features)
    # train_features = dimention_reduction(train_features)
    test_features = dimention_reduction(test_features)

    # print(len(train_features[2223]))

    # # print(train_features[0])

    # print(len(train_features))
    # print(len(test_features))
    # print(len(train_labels))
    # print(len(test_labels))

    # train(train_features,train_labels)
    test(test_features, test_labels)
