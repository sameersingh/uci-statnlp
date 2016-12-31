#!/bin/python

def train_classifier(X, y):
	from sklearn.linear_model import LogisticRegression
	cls = LogisticRegression()
	cls.fit(X, y)
	return cls

def evaluate(X, yt, cls):
	from sklearn import metrics
	yp = cls.predict(X)
	#print metrics.classification_report(yt, yp)
	acc = metrics.accuracy_score(yt, yp)
	print "  Accuracy", acc
	#f1 = metrics.f1_score(yt, yp)
	#print "  F1", f1