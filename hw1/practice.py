

list_1 = ["practicing practices practiced", "I love to see the temple. I'm going there someday."]

from nltk.stem import WordNetLemmatizer
import nltk
wordnet_lemmatizer = WordNetLemmatizer()
new_list = []
for i in range(len(list_1)):
	tokenization = nltk.word_tokenize(str(list_1[i]))
	temp = []
	for w in enumerate(tokenization):
		temp.append(wordnet_lemmatizer.lemmatize(w[1]))
		print(temp[w[0]])
	strings = str(temp)
	str1 = ""
	new_list.append(str1.join(strings))
	print(new_list[i])

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import vstack, hstack

count_vect = CountVectorizer()
trainX1 = count_vect.fit_transform(new_list)
print(trainX1)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
trainX2 = tfidf_transformer.fit_transform(trainX1)
print(trainX2)

# trainX = vstack((trainX1, trainX2))

print("trying to join")
# print(trainX)



# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer()
# vectors = vectorizer.fit_transform(list_1)
# feature_names = vectorizer.get_feature_names()
# dense = vectors.todense()
# denselist = dense.tolist()
# print(denselist)


