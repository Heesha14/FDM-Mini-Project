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

def predict_classification(gender, car, realty, child, income, income_type, educ_type, fam_stat, hou_type, age, exp, occ, fam_mem, month):

    dataset = pd.read_csv('Credit_Card_final.csv')

    # print(dataset.shape)
    dataset = dataset.iloc[:, 1:16]
    # print(dataset.head())
    # Your code goes here
    X = dataset.drop('STATUS', axis=1).values # Input features (attributes)
    y = dataset['STATUS'].values # Target vector
    # print('X shape: {}'.format(np.shape(X)))
    # print('y shape: {}'.format(np.shape(y)))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    random_model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    random_model.fit(x_train, y_train)
    prediction_test = random_model.predict(X=x_test)

    # Accuracy on Test
    print("Training Accuracy is: ", random_model.score(x_train, y_train))
    # Accuracy on Train
    print("Testing Accuracy is: ", random_model.score(x_test, y_test))

    # print(rf.predict([[1,1,1,0,112500,4,4,1,1,59,3,16,2,30]]))

    pickle.dump(random_model, open('model.pkl', 'wb'))

    model = pickle.load(open('model.pkl', 'rb'))
    output = model.predict([[gender, car, realty, child, income, income_type, educ_type, fam_stat, hou_type, age, exp, occ, fam_mem, month]])

    # print('Customer is : ', output)

    return output
