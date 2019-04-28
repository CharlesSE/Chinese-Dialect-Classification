import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from nltk.tag.stanford import StanfordPOSTagger as POSTag


char = False  # True for char granularity, false for word granularity
simplified = True  # True for simplified character coprus, false for traditional character corpus
TRAIN_PATH = './DMT/TRAININGSET-DMT_SIMP-VARDIAL2019/train.txt' if simplified else './DMT/TRAININGSET-DMT_TRAD-VARDIAL2019/train.txt'
TEST_PATH = './DMT/TRAININGSET-DMT_SIMP-VARDIAL2019/dev.txt' if simplified else './DMT/TRAININGSET-DMT_TRAD-VARDIAL2019/dev.txt'

path_to_model = './stanford-postagger/models/chinese-distsim.tagger'
path_to_jar = './stanford-postagger/stanford-postagger.jar'
ngrams = [1, 2, 3, 4] # Which ngram features that will be created


def build_dataframe(train_path, test_path, char=False):
    df_train = pd.read_csv(train_path, sep='\t', names=['Sentence', 'Dialect'])
    df_test = pd.read_csv(test_path, sep='\t', names=['Sentence', 'Dialect'])

    # Change the class names so that M becomes 0 and T becomes 1
    df_train.loc[df_train["Dialect"] == 'M', "Dialect"] = 0
    df_train.loc[df_train["Dialect"] == 'T', "Dialect"] = 1
    df_test.loc[df_test["Dialect"] == 'M', "Dialect"] = 0
    df_test.loc[df_test["Dialect"] == 'T', "Dialect"] = 1

    if char:
        df_train['Sentence'] = df_train['Sentence'].str.replace(' ', '')
        df_test['Sentence'] = df_test['Sentence'].str.replace(' ', '')
    else:
        # Remove trailing space
        df_train['Sentence'] = df_train['Sentence'].str.rstrip()
        df_test['Sentence'] = df_test['Sentence'].str.rstrip()

    x_train_df, y_train_df, x_test_df, y_test_df = df_train['Sentence'], df_train['Dialect'], df_test['Sentence'], df_test['Dialect']
    y_train_df = y_train_df.astype('int')
    y_test_df = y_test_df.astype('int')

    return x_train_df, y_train_df, x_test_df, y_test_df


def train_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train)
    score = model.score(x_val, y_val)
    predict = model.predict(x_val)
    print(type(model).__name__ + ":\t", "%.3f" % score)
   
    confusion_matrix = metrics.confusion_matrix(y_val, predict, [0, 1])
    M_weight = confusion_matrix[0][0]/len(y_val)
    T_weight = confusion_matrix[1][1]/len(y_val)

    print(confusion_matrix)
    return model, (M_weight, T_weight)


def pos_tag(dataframe, char=False):
    pos_tagger = POSTag(path_to_model, path_to_jar)

    if not char:
        dataframe = dataframe.apply(lambda x: x.split(" "))  # Split into words

    tagged = pos_tagger.tag_sents(dataframe)

    for i in range(len(tagged)):  # For each sentence
        for j in range(len(tagged[i])):  # For each word in that sentence
            tagged[i][j] = tagged[i][j][1].split("#")[1]

    sent_labels = pd.DataFrame([sent] for sent in tagged)
    sent_labels = sent_labels[0]  # Flatten
    label_strings = [" ".join(label) for label in sent_labels.values]

    return_df = pd.DataFrame(label_strings)
    return_df = return_df.astype(str)
    return return_df[0]


def pos_tagging_classification():
    print("\nPOS:")
    x_train_df, y_train_df, x_test_df, y_test_df = build_dataframe(TRAIN_PATH, TEST_PATH, char=char)
    x_train_df = pos_tag(x_train_df, char=char)
    x_test_df = pos_tag(x_test_df, char=char)


    tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 10), analyzer='char' if char else 'word')
    train_idf = tfidf.fit_transform(x_train_df)
    test_idf = tfidf.transform(x_test_df)

    x_train, y_train, x_val, y_val, x_test, y_test = train_idf, y_train_df, test_idf[1000:], y_test_df[1000:], test_idf[:1000], y_test_df[:1000]

    weighted_prediction(x_train, y_train, x_val, y_val, x_test, y_test)


def weighted_prediction(x_train, y_train, x_val, y_val, x_test, y_test):
    models = [MultinomialNB(), BernoulliNB(), LinearSVC(),
              RandomForestClassifier(n_estimators=100)]
    for i in range(len(models)):
        models[i] = train_model(models[i], x_train, y_train, x_val, y_val)  # model, (M_weight, T_weight)

    labels = []

    for i in range(len(y_val)):  # For each sentence in the validation set
        votes = {'M': 0, 'T': 0} # Each model will vote
        for model in models:
            prediction = model[0].predict(x_test[i])

            if prediction == 0:
                votes['M'] += model[1][0]
            else:
                votes['T'] += model[1][1]
        label = 0 if votes['M'] > votes['T'] else 1  # Choose the label with the highest score
        labels.append(label)

    label_df = pd.DataFrame(labels)

    correct_labels = 0
    for i in range(len(label_df)):
        if label_df[0][i] == y_test[i]:
            correct_labels += 1

    print("Weighted prediction:", correct_labels/len(y_test))


def main():
    # Preprocess data to get dataframe for test and training data
    x_train_df, y_train_df, x_test_df, y_test_df = build_dataframe(TRAIN_PATH, TEST_PATH, char=char)

    for n in ngrams:
        print("\nNgram:", n)
        tfidf = TfidfVectorizer(min_df=1, ngram_range=(n, n), analyzer='char' if char else 'word')

        #Fit the vectorizer
        train_idf = tfidf.fit_transform(x_train_df)
        test_idf = tfidf.transform(x_test_df)

        x_train, y_train, x_val, y_val, x_test, y_test = train_idf, y_train_df, test_idf[1000:], y_test_df[1000:], test_idf[:1000], y_test_df[:1000]
        weighted_prediction(x_train, y_train, x_val, y_val, x_test, y_test)

    pos_tagging_classification()


if __name__ == "__main__": main()
