import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

# Download the movie reviews dataset
nltk.download("movie_reviews")

# Function to extract features from text
def extract_features(words):
    return dict([(word, True) for word in words])

# Function to train the sentiment analysis classifier
def train_sentiment_classifier():
    # Load positive and negative movie reviews
    positive_reviews = [(extract_features(movie_reviews.words(fileids=[f])), "Positive") for f in movie_reviews.fileids("pos")]
    negative_reviews = [(extract_features(movie_reviews.words(fileids=[f])), "Negative") for f in movie_reviews.fileids("neg")]

    # Split the dataset into training and testing sets
    train_set = positive_reviews[:800] + negative_reviews[:800]
    test_set = positive_reviews[800:] + negative_reviews[800:]

    # Train a Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(train_set)

    # Display the accuracy of the classifier on the test set
    accuracy = nltk_accuracy(classifier, test_set)
    print(f"Accuracy of the classifier: {accuracy:.2%}")

    return classifier

# Function to predict sentiment for a given text
def predict_sentiment(classifier, text):
    words = text.split()
    features = extract_features(words)
    return classifier.classify(features)

def main():
    # Train the sentiment analysis classifier
    sentiment_classifier = train_sentiment_classifier()

    # Test the classifier with custom text
    test_text = "I love this product! It's amazing."
    sentiment = predict_sentiment(sentiment_classifier, test_text)
    print(f"\nSentiment Prediction for '{test_text}': {sentiment}")

if __name__ == "__main__":
    main()
