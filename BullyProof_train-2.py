import numpy as np
import os
import pandas as pd
import sys
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams
import csv
from random import shuffle
import random




def train(train_features, train_labels):
	# print("Training........\n")
	clf = svm.SVC()
	clf.fit(train_features,train_labels)
	joblib.dump(clf, 'model.pkl')


def test(test_features, test_labels):
	print("Testing........\n")

	clf = joblib.load('model.pkl')
	prediction = clf.predict(test_features)
	# toWrite = ''
	file = open('result1.csv','w')
	file2 = open('res.csv','w')
	writer = csv.writer(file)
	writer1 = csv.writer(file2)
	for i in range(len(prediction)):
		# if prediction[i]==test_labels[i]:
		writer.writerows([[prediction[i]]])
			# if test_labels[i]==1 and prediction[i]==1:
				# writer1.writerows([[i]])

		# else:
			# writer.writerows([['wrong',test_labels[i]]])


	print(classification_report(test_labels,prediction))
	print("\nAccuracy = ")
	print(accuracy_score(test_labels,prediction))


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
	# print("Dimenstionality reduction training.........")
	kpca = KernelPCA(n_components = 50)
	kpca.fit(features)
	joblib.dump(kpca,'KernelPCA.pkl')

def dimention_reduction(features):
	# print("Reducing dimentions........")
	kpca = joblib.load('KernelPCA.pkl')

	return kpca.transform(features)	



if __name__ == '__main__' : 

	train_feature = []
	train_labels = []
	test_feature = []
	test_labels = []
	labels = []
	features = []

	# dataset = pd.read_csv("train4.csv",sep = ',', delimiter = None);
	# # dataset.pop(index)
	# dataset.pop('_unit_id')
	# dataset.pop('_golden')
	# dataset.pop('_unit_state')
	# dataset.pop('_trusted_judgments')
	# dataset.pop('_last_judgment_at')
	# dataset.pop('question1')
	# dataset.pop('question1:confidence')
	# dataset.pop('question2')
	# dataset.pop('question2:confidence')
	# dataset.pop('cptn_time')
	# dataset.pop('img_url')
	# dataset.pop('owner_id')
	# dataset.pop('follows')
	# dataset.pop('followed_by')
	# dataset.pop('id')

	# print(dataset);
	# dataset.to_csv('out2.csv',sep = ',');


	df = pd.read_csv("out2.csv",sep = ',', delimiter = None)

	array = df.loc[:,['cyberaggression','cyberbullying']].as_matrix();

	count1 = 0
	count2 = 0
	for i in range(array.shape[0]):
		if (array[i][1]>1 and array[i][1]>array[i][0]) or (array[i][1]>2 and array[i][1]==array[i][0]):
			labels.append(1);
			count1 = count1+1
		else:
			count2 = count2+1
			labels.append(-1);

	# print(count2)
	# print(count1)
	# print(count1/(count1+count2))
	# # exit(0)
	posts = []
	likes = []
	shares = []

	count = 0;
	with open('out2.csv') as f:
		data = csv.reader(f,delimiter = ',')
		for row in data:
			posts.append([])

			for i in range(195):
				posts[count].append(row[i+1])
			
			# likes.append(row[196].split(" ")[0])
			posts[count].append(row[197])
			# shares.append(row[198])
			count = count+1

	posts.pop(0)
	# likes.pop(0)
	# shares.pop(0)

	# for i in range(len(likes)):
	# 	likes[i] = int(likes[i])
	# 	shares[i] = int(shares[i])

	

	for i in range(len(posts)):
		for j in range(len(posts[i])):
			posts[i][j] = re.sub('[!@#$%,+-^&*~"/|><:().?]','', posts[i][j])	
		
	stop_words = set(stopwords.words("english"))
	
	for i in range(len(posts)):
		for j in range(len(posts[i])):
			post = ''
			for w in word_tokenize(posts[i][j]):
				if w not in stop_words:
					if w.find("color")==-1 and w.find("font")==-1:
						post = post + w + ' '

			posts[i][j] = post


	
	
	features_u = []
	features_t = []
	for i in range(len(posts)):
		temp1 = []
		temp2 = []

		for j in range(len(posts[i])):
			for w in ngrams(word_tokenize(posts[i][j]),3):
				temp1.append(w)
			for w in ngrams(word_tokenize(posts[i][j]),0):
				temp2.append(w)

		to_append_u = []
		to_append_t = []
		for w in temp2:
			if(w[0]!='empety'):
				to_append_u.append(w[0].lower())
		for w in temp1:
			if(w[0]!='empety'):
				to_append_t.append(w)

		features_u.append(to_append_u)
		features_t.append(to_append_t)


	# print(features_u[0])
	# print("\n")
	# print(features_t[0])


	# print(len(features_u[0]))
	# print(len(features_u[277]))

	# print(len(features_t[0]))
	# print(len(features_t[277]))

	
	# print("Finding FreqDist......")

	all_words = []
	for i in range(len(features_u)):
		for j in features_u[i]: 
			all_words.append(j.lower());

	all_tuples = []
	for i in range(len(features_t)):
		for j in features_t[i]:
			all_tuples.append(j);



	# # # print(all_words);
	# print(len(all_words))
	# print(len(all_tuples))


	all_words = nltk.FreqDist(all_words)
	all_tuples = nltk.FreqDist(all_tuples)


	# print(len(all_words))
	# print(len(all_tuples))

	# # print(all_words["night"])

	# # print(all_words)

	# word_features = list(all_words.keys())[3000:7000] #3000:10000
	# tuple_features = list(all_tuples.keys())[3000:14000]	#3000:16000

	# joblib.dump(tuple_features,"tuples.pkl")
	# joblib.dump(word_features,"words.pkl")

	word_features = joblib.load("words.pkl")
	tuple_features = joblib.load("tuples.pkl")

	# print(word_features[0])
	# print(tuple_features[0])


	# # print(find_features(features[2018],word_features))
	# # print(find_features_tuples(features2[2018],tuple_features))

	# # feature_set = [(find_features(f,word_features)) for f in features]
	# print("Making words and tuples as features.......")

	feature_set = []
	for i in range(len(features_u)):
		dum = []
		for w in find_features(features_u[i],word_features):
			dum.append(w)
		for w in find_features_tuples(features_t[i],tuple_features):
			dum.append(w)
		# dum.append(likes[i])
		# dum.append(shares[i])
		feature_set.append(dum)

	# print(len(feature_set[201]))
	# print(len(find_features(features_u[201],word_features)))
	# print(len(find_features_tuples(features_t[201],tuple_features)))


	# print(feature_set[27])
	# print(len(features_t))
	# print(len(feature_set))

	# # vectorizer = TfidfVectorizer(min_df=5,
 # #                                 max_df = 0.8,
 # #                                 sublinear_tf=True,
 # #                                 use_idf=True)
	# # train_vectors = vectorizer.fit_transform(features)
	# # # test_vectors = vectorizer.transform(test_data)


	# # feature_set_to_shuffle = []
	# # for i in range(len(feature_set)):
	# # 	dum = []
	# # 	dum.append(feature_set[i])
	# # 	dum.append(labels[i])
	# # 	feature_set_to_shuffle.append(dum)

	# print("Standardizing.....")


	feature_set = StandardScaler().fit_transform(feature_set)

	# print("Shuffling the dataset.......")

	# feature_set_to_shuffle = list(zip(feature_set,labels))

	# # print(feature_set_to_shuffle[12187])
	# shuffle(feature_set_to_shuffle)
	# # print(feature_set_to_shuffle[12187])

	# feature_set,labels = zip(*feature_set_to_shuffle)

	# train_features = feature_set[:(round(.75 * len(feature_set)))];
	# test_features = feature_set[(round(.75 * len(feature_set))):];
	# train_labels = labels[:(round(.75 * len(feature_set)))];
	# test_labels = labels[(round(.75 * len(feature_set))):];

	test_features = feature_set
	test_labels = labels


	# train2 = []
	# train2l = []
	# for i in range(len(feature_set)):
	# 	if labels[i]==1:
	# 		# print(labels[i])
	# 		train2.append(feature_set[i])
	# 		train2l.append(labels[i])
	# # print(train2l[19])

	# for i in range(round(.40 * len(feature_set))):
	# 	train2.append(feature_set[random.randint(1,len(feature_set)-5)])
	# 	train2l.append(labels[random.randint(1,len(feature_set)-5)])

	# train_features = train2
	# train_labels = train2l
	# print(len(train_labels))

	# print("Dimenstionality reduction......")
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
	test(test_features,test_labels)


