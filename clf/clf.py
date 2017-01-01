#coding:utf-8
import jieba
punct = set(u''' `!@#$%^&*()_+-={}[]！@#￥%……&*（）——+-=《》，。<>,./?？、:!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…\\''')
filterpunt = lambda s: ''.join(filter(lambda x: x not in punct, s))
stop_words = open("stop_words.txt", "r").read().decode("utf8").split('\n')

def cut_words(line):
	words = list(jieba.cut(line.strip(), cut_all = True))
	# 去除jieba分词留下了标点符号
	words = [filterpunt(word).lower() for word in words if not (len(word) == 1 and word.isalpha()) and word not in stop_words and len(filterpunt(word)) > 0]
	return words

# 预处理：分词、去标点、分类属性二元化
def preprocess(raw_data):

	labels = []
	data = []
	label_data_dict = {} # 标签对应的数据集，key:label, value:此label对应的所有data下标形成的list

	for line in raw_data:
		if line[0] == '(' and line[1].isdigit() and line[5] == ')':
			label = int(line[1])
			if label == 0:
				continue;
			words = cut_words(line[7:])
			if len(words) > 0:
				#分类属性二元化
				if label <= 2:
					label = 0 # negtive
				else:
					label = 1 # positive

				labels.append(label)
				data.append(words)
				if label not in label_data_dict:
					label_data_dict[label] = []
				label_data_dict[label].append(len(data) - 1)

	return data, labels, label_data_dict

# 根据ratio划分数据集和测试集
def divide_dataset(data, labels, label_data_dict, ratio=0.5):
	train_data = []; train_label = []; test_data = []; test_label = []
	for key in label_data_dict.keys():
		i = 0
		while i < len(label_data_dict[key]) * ratio:
			train_data.append(data[label_data_dict[key][i]])
			train_label.append(labels[label_data_dict[key][i]])
			i += 1
		while i < len(label_data_dict[key]):
			test_data.append(data[label_data_dict[key][i]])
			test_label.append(labels[label_data_dict[key][i]])
			i += 1
	return train_data, train_label, test_data, test_label


if __name__ == '__main__':
	import os
	files = os.listdir("douban")
	raw_data = []
	i = 0
	for file in files:
		if i < 500: # 对模型进行tuning时，可以适当减少读入的文件数量
			raw_data.extend(open("douban/"+file, "r").readlines())
			i += 1
	
	# 1.预处理
	data, labels, label_data_dict = preprocess(raw_data)

	# 2.切割数据集使各类别的数据量相同
	min_label_datasize = 0
	for key in label_data_dict.keys():
		if min_label_datasize == 0 or len(label_data_dict[key]) < min_label_datasize:
			min_label_datasize = len(label_data_dict[key])

	for key in label_data_dict.keys():
		label_data_dict[key] = label_data_dict[key][:min_label_datasize]

	print "data size of every class:", min_label_datasize

	train_data, train_label, test_data, test_label = divide_dataset(data, labels, label_data_dict, ratio=0.7)

	# 3.特征提取及向量化
	# print '************** 1. CountVectorizer **************'  
	# from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
	# # 数据集已经过分词，analyzer直接返回原向量即可， 根据max_features阶段词汇表
	# count_vectorizer = CountVectorizer(analyzer = lambda x : x, max_features=10000) 
	# # fit_transform = fit + transform, fit: 训练模型（生成词汇表等） transform：将词向量转变成数值向量 
	# # 只对训练集fit， 对训练集和测试集transform
	# train_features = count_vectorizer.fit_transform(train_data) 
	# test_features = count_vectorizer.transform(test_data)

	print '************** 2. CountVectorizer + TfidfTransformer **************'  
	from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
	count_vectorizer = CountVectorizer(analyzer = lambda x : x, max_features=10000) 
	train_features = count_vectorizer.fit_transform(train_data)
	test_features = count_vectorizer.transform(test_data)
	# 用tf-idf对word count向量进行加权
	tfidftransformer = TfidfTransformer()
	train_features = tfidftransformer.fit_transform(train_features)
	test_features = tfidftransformer.transform(test_features)

	# print '**************************** 3. TfidfVectorizer ****************************'  
	# from sklearn.feature_extraction.text import TfidfVectorizer  
	# tfidf_vectorizer = TfidfVectorizer(sublinear_tf = True,  
	#                                    max_df = 0.5,
	#                                    analyzer = lambda x : x);  
	# train_features = tfidf_vectorizer.fit_transform(train_data);  
	# test_features = tfidf_vectorizer.transform(test_data);  

	# print '**************************** 4. HashingVectorizer ****************************'  
	# from sklearn.feature_extraction.text import HashingVectorizer  
	# hash_vectorizer = HashingVectorizer(analyzer = lambda x : x, non_negative = True,  
	#                                n_features = 10000)  
	# train_features = hash_vectorizer.fit_transform(train_data)  
	# test_features = hash_vectorizer.transform(test_data);  	
	

	from sklearn.metrics import classification_report # for report
	# 4. 训练和评估分类模型
	print '************** Naive Bayes **************'  
	from sklearn.naive_bayes import MultinomialNB  
	nb_clf = MultinomialNB(alpha = 0.01)   
	nb_clf.fit(train_features, train_label)
	pred = nb_clf.predict(test_features)
	print(classification_report(test_label, pred))


	print '************** KNN **************'  
	from sklearn.neighbors import KNeighborsClassifier 
	knn_clf = KNeighborsClassifier()   
	knn_clf.fit(train_features, train_label)
	pred = knn_clf.predict(test_features)
	print(classification_report(test_label, pred))

	print '************** SVM **************'  
	from sklearn.svm import SVC 
	svm_clf = SVC(kernel = 'linear') #default kernel function is rbf  
	svm_clf.fit(train_features, train_label)  
	pred = svm_clf.predict(test_features);  
	print(classification_report(test_label, pred))


	

