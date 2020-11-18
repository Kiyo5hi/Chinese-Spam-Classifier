from spam_classifier import SpamClassifier

if __name__ == "__main__":
    nb = SpamClassifier()
    nb.load_data()
    nb.vectorize_data()
    nb.train()
    nb.test()
    print(nb.predict("店内今天到货，请选购", "今晚是双十一，要打折，半夜把我叫起来！", "双十一大促，全场8折"))
    