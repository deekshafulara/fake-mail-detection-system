from flask import Flask, render_template, request, jsonify
import sklearn
import pickle
import pandas as pd

# ----------------------------------------------Preprocessing of this code---------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


raw_mail = pd.read_csv("mail_data.csv")

mail_data = raw_mail.where((pd.notnull(raw_mail)), '')

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

X = mail_data['Message']
Y = mail_data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)


# -------------------------------------------------------------------------------------------------------------------------------------






# -------------------------------------------------------------Setting up the flask-----------------------------------


app = Flask(__name__, template_folder='templates', static_folder='staticFiles')

model = pickle.load(open('spam_det_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/sendd', methods=['GET', 'POST'])
def predict():

    a = (request.form.get('mail_check'))

    c = (request.form.get('out_type'))

    if(a == ""):
        return "Enter Some value of mail"
    if(c == None):
        return "Select the output type"

    a = str(request.form.get('mail_check'))


    input_mail = [a]
    input_mail = feature_extraction.transform(input_mail)


    result = model.predict(input_mail)



    if(c == "Json_format"):
        bb = "Warning it's a spam message"
        if(result[0] == 1):
            bb = "It's a safe message"
        
        ann = {"Message" : a, "Ans_format" : c, "Status" : bb}
        return jsonify(ann)

    else:

        if(result[0] == 1):
            return render_template('index.html', label = 1)
        else:
            return render_template('index.html', label = -1)
    
    

if __name__ =='__main__':
    app.run(debug=True)

 