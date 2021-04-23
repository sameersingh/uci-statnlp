def self_train(Xtrain, Ytrain, Xtest, Ytest):

	import matplotlib.pyplot as plt
	from sklearn.metrics import f1_score
	print("Xtrain")
	print(Xtrain)
	print("Ytrain")
	print(Ytrain)
	print("Xtest")
	print(Xtest)
	print("Ytest")
	print(Ytest)

	from sklearn.linear_model import LogisticRegression
	cls = LogisticRegression()
	cls.fit(Xtrain,Ytrain)
	Yhat_train = cls.predict(Xtrain)
	Yhat_test = cls.predict(Xtest)

	preds_probs = cls.predict_proba(Xtest)
	preds = cls.predict(Xunlabel)

	# Store predictions and probabilities in dataframe
	df_pred_prob = pd.DataFrame([])
	df_pred_prob['preds'] = preds
	df_pred_prob['prob_0'] = prob_0
	df_pred_prob['prob_1'] = prob_1
	df_pred_prob.index = X_unlabeled.index
	#
	# # Separate predictions with > 90% probability
	high_prob = pd.concat([df_pred_prob.loc[df_pred_prob['prob_0'] > 0.90],
	 					   df_pred_prob.loc[df_pred_prob['prob_1'] > 0.90]],
	 					  axis=0)

	# return cls

