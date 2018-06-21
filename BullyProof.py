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
import smtplib

# def send_mail(count):


# 	# creates SMTP session
# 	s = smtplib.SMTP('smtp.gmail.com:587')
	 
# 	# start TLS for security
# 	s.starttls()
# 	s.ehlo()
# 	# Authentication
# 	s.login("BullyProof@gmail.com", "%BullyProof#")
	 
# 	# message to be sent
# 	message = "Unfortunately your friend is bullied" + str(count) + " times in this week itself"
	 
# 	# sending the mail
# 	s.sendmail("BullyProof@gmail.com", "Friend@gmail.com", message)
	 
# 	# terminating the session
# 	s.quit()


def test(test_features):
	# print("Testing........\n")

	count = 0
	clf = joblib.load('model.pkl')
	prediction = clf.predict(test_features)
	file = open('result.csv','w')
	writer = csv.writer(file)
	for i in range(len(prediction)):
		writer.writerows([[prediction[i]]])
		if prediction[i]==1:
			count = count + 1
			print("Tweet session",str(i+1),": "," Instance of Bully")
		else:
			print("Tweet session",str(i+1),": "," Not an instance of Bully")

	return count


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


def dimention_reduction(features):
	kpca = joblib.load('KernelPCA.pkl')

	return kpca.transform(features)	



if __name__ == '__main__' : 

	test_feature = []
	features = []

	posts = []

	count = 0;
	with open('tweets.csv') as f:
		data = csv.reader(f,delimiter = ',')
		for row in data:
			posts.append([])

			for i in range(195):
				posts[count].append(row[i+1])
			
			posts[count].append(row[197])
			count = count+1

	# Removing all the special characters
	for i in range(len(posts)):
		for j in range(len(posts[i])):
			posts[i][j] = re.sub('[!@#$%,+-^&*~"/|><:().?]','', posts[i][j])	
		
	stop_words = set(stopwords.words("english"))
	
	# Removing all the stop words

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


	# print("Finding FreqDist......")

	all_words = []
	for i in range(len(features_u)):
		for j in features_u[i]: 
			all_words.append(j.lower());

	all_tuples = []
	for i in range(len(features_t)):
		for j in features_t[i]:
			all_tuples.append(j);


	all_words = nltk.FreqDist(all_words)
	all_tuples = nltk.FreqDist(all_tuples)


	word_features = joblib.load("words.pkl")
	tuple_features = joblib.load("tuples.pkl")

	
	# print("Making words and tuples as features.......")

	feature_set = []
	for i in range(len(features_u)):
		dum = []
		for w in find_features(features_u[i],word_features):
			dum.append(w)
		for w in find_features_tuples(features_t[i],tuple_features):
			dum.append(w)
		feature_set.append(dum)


	# print("Standardizing.....")
	
	feature_set = StandardScaler().fit_transform(feature_set)

	
	test_features = feature_set

	# print("Dimenstionality reduction......")
	test_features = dimention_reduction(test_features)

	bully_count = test(test_features)
	# send_mail(bully_count)


