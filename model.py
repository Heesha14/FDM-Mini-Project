import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

def predict_classification(gender, car, realty, child, income, income_type, educ_type, fam_stat, hou_type, age, exp, occ, fam_mem, paid_off, past_dues, no_loan):
    dataset = pd.read_csv('C:/Users/Raisul Zulfikar/Desktop/Credit Card/Clean_dataset.csv')

    dataset = dataset.iloc[:, 1:18]
    # print(dataset.head())
    # Your code goes here
    X = dataset.drop('target', axis=1).values # Input features (attributes)
    y = dataset['target'].values # Target vector
    # print('X shape: {}'.format(np.shape(X)))
    # print('y shape: {}'.format(np.shape(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    rf.fit(X_train, y_train)
    prediction_test = rf.predict(X=X_test)

    # Accuracy on Test
    print("Training Accuracy is: ", rf.score(X_train, y_train))
    # Accuracy on Train
    print("Testing Accuracy is: ", rf.score(X_test, y_test))

    # print(rf.predict([[1, 1, 1, 0, 112500, 0, 0, 0, 0, 58.83, 3.106, 16, 2, 1, 1, 1]]))

    pickle.dump(rf, open('model.pkl', 'wb'))

    model = pickle.load(open('model.pkl', 'rb'))
    output = model.predict([[gender, car, realty, child, income, income_type, educ_type, fam_stat, hou_type, age, exp, occ,
                             fam_mem, paid_off, past_dues, no_loan]])

    # print('Customer is : ', output)

    return output
