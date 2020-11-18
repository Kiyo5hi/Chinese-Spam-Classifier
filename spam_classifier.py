from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import jieba
import os
import re
from joblib import dump, load

class SpamClassifier:

    def __init__(self, root_path="./data", data_path="labeled.txt", stop_words_path="stop.txt", train_size=0.8):
        super().__init__()
        self.__root_path = root_path
        self.__data_path = data_path
        self.__stop_words_path = stop_words_path
        self.__train_size = train_size
        self.__count_vect = CountVectorizer()
        self.__tfidf_transformer = TfidfTransformer()

    def __text_parse_helper(self, txt):
        txt = "".join(re.findall(u'[\u4e00-\u9fa5]+', txt))
        corpus = jieba.lcut(txt)
        return " ".join([tok for tok in corpus if (tok not in self.__stop_words and len(tok) >= 2)])

    def load_data(self):
        if (os.path.isfile("stop_words.joblib")):
            self.__stop_words = load("stop_words.joblib")
        else:
            with open(os.path.join(self.__root_path, self.__stop_words_path), mode="r", encoding="utf-8") as f:
                self.__stop_words = f.read().split("\n")
                dump(self.__stop_words, "stop_words.joblib")

        if (os.path.isfile("corpus.joblib") and os.path.isfile("label_list.joblib")):
            self.__label_list = load("label_list.joblib")
            self.__corpus = load("corpus.joblib")
        else:
            with open(os.path.join(self.__root_path, self.__data_path), mode="r", encoding="utf-8") as f:
                data = f.readlines()
                self.__label_list = [i[0] for i in data]
                self.__corpus = [self.__text_parse_helper(i[2:-1]) for i in data]
                dump(self.__label_list, "label_list.joblib")
                dump(self.__corpus, "corpus.joblib")

    def vectorize_data(self):
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            self.__corpus, self.__label_list, train_size=self.__train_size)

        X_train_counts = self.__count_vect.fit_transform(self.__X_train)
        self.__X_train_tfidf = self.__tfidf_transformer.fit_transform(
            X_train_counts)

        X_test_counts = self.__count_vect.transform(self.__X_test)
        self.__X_test_tfidf = self.__tfidf_transformer.transform(X_test_counts)

    def train(self):
        self.__clf = MultinomialNB().fit(self.__X_train_tfidf, self.__y_train)

    def test(self):
        predicted = self.__clf.predict(self.__X_test_tfidf)
        print(metrics.classification_report(self.__y_test, predicted))

    def predict(self, *txt):
        corpus = [self.__text_parse_helper(i) for i in txt]
        X_counts = self.__count_vect.transform(corpus)
        X_tfidf = self.__tfidf_transformer.transform(X_counts)
        predicted = self.__clf.predict(X_tfidf)
        return predicted
