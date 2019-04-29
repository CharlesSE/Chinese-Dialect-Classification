import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv2D, Flatten, Conv1D, GlobalMaxPooling1D, BatchNormalization, SpatialDropout1D, Dropout, GlobalAveragePooling1D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer


char = False
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


def cnn():
    batch_size = 32
    try:
        if not char:
            with open('cnn_model.json', 'r') as f:
                model = model_from_json(f.read())
            model.load_weights('cnn_model_weights.h5')
            print("Model loaded from disk")
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            with open('cnn_model.json_char', 'r') as f:
                model = model_from_json(f.read())
            model.load_weights('cnn_model_weights_char.h5')
            print("Model loaded from disk")
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    except:
        model = Sequential()
        model.add(Embedding(max_features, 150, input_length=x_train.shape[1], dropout=0.2))
        model.add(Conv1D(64, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(10, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping_monitor = EarlyStopping(patience=4)
        model.fit(x_train, y_train, epochs=10, validation_split=0.1, callbacks=[early_stopping_monitor], verbose=1,
                  batch_size=batch_size)
        save_model(model)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    if not char:
        with open("cnn_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("cnn_model_weights.h5")
    else:
        with open("cnn_model.json_char", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("cnn_model_weights_char.h5")

    print("Saved model to disk")


x_train_df, y_train_df, x_test_df, y_test_df = build_dataframe(TRAIN_PATH, TEST_PATH, char=char)
x_df = pd.concat([x_train_df, x_test_df])
max_features = 30000
tokenizer = Tokenizer(nb_words=max_features, char_level=char, split=' ')
tokenizer.fit_on_texts(x_df.values)
X1 = tokenizer.texts_to_sequences(x_df.values)
X1 = pad_sequences(X1)
Y1 = pd.get_dummies(y_train_df).values
Y2 = pd.get_dummies(y_test_df).values

x_train, y_train, x_test, y_test = X1[:-2000], Y1, X1[-2000:], Y2

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
cnn()