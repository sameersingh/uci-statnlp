#!/bin/python

def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains a logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import GridSearchCV
	cls = LogisticRegression()
	param_grid = {
	 	'C': [100],
	 	'penalty': ['l2'],
	 	'max_iter': list(range(400, 600, 200)),
	 	'solver': ['lbfgs']
	 }
	param_search = GridSearchCV(cls, param_grid=param_grid, refit=True, verbose=3, cv=3)
	param_search.fit(X, y)
	print("printing grid scores")
	print(param_search.cv_results_)
	import matplotlib.pyplot as plt
	print(param_grid['C'])
	print(param_search.cv_results_['mean_test_score'])
	plt.plot(param_grid['C'], param_search.cv_results_['mean_test_score'])

	import seaborn as sns

	return param_search

def evaluate(X, yt, cls):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	print("  Accuracy", acc)
