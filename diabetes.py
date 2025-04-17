import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from flask import Flask, render_template
from dotenv import load_dotenv

from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from tensorflow import keras

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap5(app)

class LabForm(FlaskForm):
    preg = StringField('# Pregnancies', validators=[DataRequired()])
    glucose = StringField('Glucose', validators=[DataRequired()])
    blood = StringField('Blood pressure', validators=[DataRequired()])
    skin = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin', validators=[DataRequired()])
    bmi = StringField('BMI', validators=[DataRequired()])
    dpf = StringField('DPF score', validators=[DataRequired()])
    age = StringField('Age', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        # get the form data for the patient and put into array
        X_test = np.array([[float(form.preg.data),
                            float(form.glucose.data),
                            float(form.blood.data),
                            float(form.skin.data),
                            float(form.insulin.data),
                            float(form.bmi.data),
                            float(form.dpf.data),
                            float(form.age.data)]])

        # get diabetes dataset
        data = pd.read_csv('./diabetes.csv', sep=',')
        X = data.values[:, :8]
        y = data.values[:, 8]

        # scaling
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_test = scaler.transform(X_test)

        # load model
        model = keras.models.load_model('pima_model.keras')

        # predict
        prediction = model.predict(X_test)
        res = prediction[0][0]
        res = round(res * 100, 2)

        return render_template('result.html', res=res)

    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run()
