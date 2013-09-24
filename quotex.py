from features.features import *
from nltk.classify import MaxentClassifier

########## HELPER FUNCTIONS ##########

def get_features(words):
    '''
    Function that aggregates active features for the maxent classifier and returns
    a feature dict in the format expected by NLTK.
    '''
    features = {} # Start with empty feature dict

    # Put features here (found in classify.features)
    features['contains_quotes'] = contains_quotes(words)
    features['first_quote_index'] = first_quote_index(words)
    features['last_word_%s' % last_word(clean_text(words))] = True
    features['said_near_source'] = said_near_source(words)
    features['num_words_between_quotes'] = num_words_between_quotes(words)
    for word in words_near_quotes(words):
        features['%s_near_quote' % word] = True

    return features

########## MAIN ##########

if __name__ == '__main__':
    toclassify = open('data/input.txt', 'rU').readlines()

    # Load the training data in the format (features, bool_quote_or_not)
    training = [(get_features(i), j.strip() == 'True') for i, j in
                    (line.split('|') for line in
                        (open('data/train.txt', 'rU').readlines()))]

    classifier = MaxentClassifier.train(training, algorithm='iis', trace=0, max_iter=10)

    for item in toclassify:
        item = item.strip() # A little cleanup

        # Run the classifier, make a guess (quote or no quote) and attach a probability
        classify = classifier.prob_classify(get_features(item))
        guess = classify.max()
        certainty = float(classify.prob(guess))

        print guess, certainty