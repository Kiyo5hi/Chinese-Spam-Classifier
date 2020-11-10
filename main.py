from spam_classifier import SpamClassifier

if __name__ == "__main__":
    nb = SpamClassifier()
    nb.load_data()
    nb.vectorize_data()
    nb.train()
    nb.test()
