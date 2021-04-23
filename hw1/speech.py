import matplotlib.pyplot


class Data: pass

def read_files(tarfname):
	"""Read the training and development data from the speech tar file.
	The returned object contains various fields that store the data, such as:

	train_data,dev_data: array of documents (array of words)
	train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
	train_labels,dev_labels: the true string label for each document (same length as data)

	The data is also preprocessed for use with scikit-learn, as:

	count_vec: CountVectorizer used to process the data (for reapplication on new data)
	trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
	le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
	target_labels: List of labels (same order as used in le)
	trainy,devy: array of int labels, one for each document
	"""
	import tarfile
	# from spacy.lang.en import English
	# nlp = English()
	tar = tarfile.open(tarfname, "r:gz")
	speech = Data()
	print("-- train data")
	speech.train_data, speech.train_fnames, speech.train_labels = read_tsv(tar, "train.tsv")

	# print("printing train data")
	# print(speech.train_data)
	# print("printing type of train data")
	# print(type(speech.train_data))
	# from nltk.stem import WordNetLemmatizer
	# import nltk
	# wordnet_lemmatizer = WordNetLemmatizer()
	# speech_train_lemma = []
	# for i in range(len(speech.train_data)):
	# 	tokenization = nltk.word_tokenize(str(speech.train_data[i]))
	# 	temp = []
	# 	for w in enumerate(tokenization):
	# 		temp.append(wordnet_lemmatizer.lemmatize(w[1]))
	# 	strings = str(temp)
	# 	str1 = ""
	# 	speech_train_lemma.append(str1.join(strings))

	print("-- dev data")
	speech.dev_data, speech.dev_fnames, speech.dev_labels = read_tsv(tar, "dev.tsv")
	print(len(speech.dev_data))
	print("-- transforming data and labels")

	from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
	from scipy.sparse import vstack

	speech.count_vect = CountVectorizer()
	speech.trainX = speech.count_vect.fit_transform(speech.train_data)

	# tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
	# speech.trainX = tfidf_transformer.fit_transform(speech.trainX)

	#speech.trainX = vstack(speech.trainX1, speech.trainX2)

	# print("trainX2 1st item")
	# print(speech.trainX2[1])
	# print("trainX2 type")
	# print(type(speech.trainX2))
	speech.devX = speech.count_vect.transform(speech.dev_data)
	# tfidf_transformer_dev = TfidfTransformer(smooth_idf=True, use_idf=True)
	# speech.devX = tfidf_transformer_dev.fit_transform(speech.devX)
	from sklearn import preprocessing
	speech.le = preprocessing.LabelEncoder()
	speech.le.fit(speech.train_labels)
	speech.target_labels = speech.le.classes_
	speech.trainy = speech.le.transform(speech.train_labels)
	speech.devy = speech.le.transform(speech.dev_labels)
	tar.close()
	return speech

def read_unlabeled(tarfname, speech):
	"""Reads the unlabeled data.

	The returned object contains three fields that represent the unlabeled data.

	data: documents, represented as sequence of words
	fnames: list of filenames, one for each document
	X: bag of word vector for each document, using the speech.vectorizer
	"""
	import tarfile
	tar = tarfile.open(tarfname, "r:gz")
	unlabeled = Data()
	unlabeled.data = []
	unlabeled.fnames = []
	for m in tar.getmembers():
		if "unlabeled" in m.name and ".txt" in m.name:
			unlabeled.fnames.append(m.name)
			unlabeled.data.append(read_instance(tar, m.name))
	unlabeled.X = speech.count_vect.transform(unlabeled.data)
	print(unlabeled.X.shape)
	tar.close()
	return unlabeled

def read_tsv(tar, fname):
	member = tar.getmember(fname)
	print(member.name)
	tf = tar.extractfile(member)
	data = []
	labels = []
	fnames = []
	for line in tf:
		line = line.decode("utf-8")
		(ifname,label) = line.strip().split("\t")
		#print ifname, ":", label
		content = read_instance(tar, ifname)
		labels.append(label)
		fnames.append(ifname)
		data.append(content)
	return data, fnames, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, speech):
	"""Writes the predictions in Kaggle format.

	Given the unlabeled object, classifier, outputfilename, and the speech object,
	this function write the predictions of the classifier on the unlabeled data and
	writes it to the outputfilename. The speech object is required to ensure
	consistent label names.
	"""
	yp = cls.predict(unlabeled.X)
	labels = speech.le.inverse_transform(yp)
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	for i in range(len(unlabeled.fnames)):
		fname = unlabeled.fnames[i]
		# iid = file_to_id(fname)
		f.write(str(i+1))
		f.write(",")
		#f.write(fname)
		#f.write(",")
		f.write(labels[i])
		f.write("\n")
	f.close()

def file_to_id(fname):
	return str(int(fname.replace("unlabeled/","").replace("labeled/","").replace(".txt","")))

def write_gold_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the truth.

	You will not be able to run this code, since the tsvfile is not
	accessible to you (it is the test labels).
	"""
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(ifname,label) = line.strip().split("\t")
			# iid = file_to_id(ifname)
			i += 1
			f.write(str(i))
			f.write(",")
			#f.write(ifname)
			#f.write(",")
			f.write(label)
			f.write("\n")
	f.close()

def write_basic_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the naive baseline.

	This baseline predicts OBAMA_PRIMARY2008 for all the instances.
	You will not be able to run this code, since the tsvfile is not
	accessible to you (it is the test labels).
	"""
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(ifname,label) = line.strip().split("\t")
			i += 1
			f.write(str(i))
			f.write(",")
			f.write("OBAMA_PRIMARY2008")
			f.write("\n")
	f.close()

def read_instance(tar, ifname):
	inst = tar.getmember(ifname)
	ifile = tar.extractfile(inst)
	content = ifile.read().strip()
	return content

if __name__ == "__main__":
	print("Reading data")
	tarfname = "data/speech.tar.gz"
	#speech = read_files(tarfname)
	import pickle
	#pickle.dump(speech, open('speech_pickle', 'wb'))
	print("loading speech pickle")
	speech = pickle.load(open('speech_pickle', 'rb'))

	import self_training
	#self_training(speech.trainX, speech.trainy, speech.devX, speech.devy)

	# print(speech.trainX)
	#
	# print("Training classifier")
	# import classify
	# cls = classify.train_classifier(speech.trainX, speech.trainy)
	# print("Evaluating")
	# classify.evaluate(speech.trainX, speech.trainy, cls)
	# classify.evaluate(speech.devX, speech.devy, cls)

	print("Reading unlabeled data")
	#unlabeled = read_unlabeled(tarfname, speech)
	#pickle.dump(unlabeled, open('unlabeled_pickle', 'wb'))

	print("loading unlabeled pickle")
	unlabeled = pickle.load(open('unlabeled_pickle', 'rb'))
	# print("Unlabeled X")
	# print(unlabeled.X)
	# print("Unlabeled Data")
	# print(unlabeled.data)
	# print("Unlabeled Fnames")
	# print(unlabeled.fnames)

	#Semi Supervised portion
	from sklearn.linear_model import LogisticRegression
	import matplotlib.pyplot as plt
	import scipy.sparse
	import pandas as pd
	from scipy.sparse import hstack

	# speech.trainX = pd.DataFrame.sparse.from_spmatrix(speech.trainX)
	# speech.trainy = pd.DataFrame.sparse.from_spmatrix(speech.trainy)
	# unlabeled.X = pd.DataFrame.sparse.from_spmatrix(unlabeled.X)
	# cls = LogisticRegression(max_iter=1000)
	# cls.fit(speech.trainX,speech.trainy)

	# preds_probs = cls.predict_proba(speech.devX)
	# plt.hist(preds_probs)
	# plt.title("Prediction Probability on Development Data (19 unique classes)")
	# plt.xlabel("Probability of Correct Prediction")
	# plt.ylabel("Counts")
	# plt.show()
	# plt.savefig("pred_prob")

	preds_probs = cls.predict_proba(unlabeled.X)
	print("preds probs")
	print(preds_probs)


	#print("Writing pred file")
	#write_pred_kaggle_file(unlabeled, cls, "data/speech-pred.csv", speech)



	# You can't run this since you do not have the true labels
	#print "Writing gold file"
	#write_gold_kaggle_file("data/speech-unlabeled.tsv", "data/speech-gold.csv")
	#write_basic_kaggle_file("data/speech-unlabeled.tsv", "data/speech-basic.csv")
