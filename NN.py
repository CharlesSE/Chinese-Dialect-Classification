import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import model_from_json


char = True
TRAIN_PATH = './DMT/TRAININGSET-DMT_SIMP-VARDIAL2019/train.txt'
TEST_PATH = './DMT/TRAININGSET-DMT_SIMP-VARDIAL2019/dev.txt'

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


def deep_learning():
    y_train2 = to_categorical(y_train)
    y_test2 = to_categorical(y_test)
    try:
        if not char:
            with open('model2.json', 'r') as f:
                model = model_from_json(f.read())
            model.load_weights('model_weights.h5')
            print("Model loaded from disk")
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            with open('model.json_char', 'r') as f:
                model = model_from_json(f.read())
            model.load_weights('model_weights_char.h5')
            print("Model loaded from disk")
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    except:
        model = Sequential()
        num_cols = len(tfidf.get_feature_names())
        model.add(Dense(250, activation='relu', input_shape=(num_cols,)))
        model.add(Dense(250, activation='relu'))
        model.add(Dense(250, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        early_stopping_monitor = EarlyStopping(patience=2)
        model.fit(x_train, y_train2, epochs=5, validation_split=0.1, callbacks=[early_stopping_monitor])
        #save_model(model)

    scores = model.evaluate(x_test, y_test2, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    if not char:
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model_weights.h5")
    else:
        with open("model.json_char", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model_weights_char.h5")

    print("Saved model to disk")


x_train_df, y_train_df, x_test_df, y_test_df = build_dataframe(TRAIN_PATH, TEST_PATH, char=char)
# Define vectorizer
tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 3), analyzer='char' if char else 'word')


# Fit the vectorizer
train_idf = tfidf.fit_transform(x_train_df)
test_idf = tfidf.transform(x_test_df)

x_train, y_train, x_test, y_test = train_idf, y_train_df, test_idf, y_test_df
deep_learning()