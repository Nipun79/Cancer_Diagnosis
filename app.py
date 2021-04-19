import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
#dir_path = os.path.dirname(os.path.realpath(__file__))

# STATIC_FOLDER = dir_path + '/static'
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/liver')
def liver():
    return render_template('liver.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        input_features = [x for x in request.form.values()]
        features_value = [np.array(input_features)]
        features_name = [['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase','Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']]
        df = pd.DataFrame(features_value, columns=features_name)
        model = pickle.load(open('liver_saved_model.pkl', 'rb')) 
        prediction=model.predict(df)
        if prediction == 1:
            res_val = " Liver cancer"
        else:
            res_val = "no Liver cancer"
    return render_template('liver.html',prediction='Patient has {}'.format(res_val))  

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')
@app.route('/predict_kidney',methods=['POST'])
def predict_kidney():
    if request.method=='POST':
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]
        features_name=['age', 'bp', 'al','su','rbc','pc', 'pcc','ba', 'bgr', 'bu', 'sc','pot','wc', 'htn', 'dm','cad','pe','ane']
        df = pd.DataFrame(features_value, columns=features_name)
        model = pickle.load(open('kidney.pkl', 'rb')) 
        prediction=model.predict(df)
        if prediction == 1:
            res_val = "Kidney Disease"
        else:
            res_val = "no Kidney Disease"
    return render_template('kidney.html',prediction='Patient has {}'.format(res_val))

@app.route('/heart_disease')
def heart_disease():
    return render_template('heart_disease.html')
@app.route('/predict_heart_disease',methods=['POST'])
def predict_heart_disease():
    if request.method=='POST':
        input_features = [x for x in request.form.values()]
        features_value = [np.array(input_features)]
        features_name=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']  
        df = pd.DataFrame(features_value, columns=features_name)
        model = pickle.load(open('loaded_heart_disease_model.pkl', 'rb')) 
        prediction=model.predict(df)
        if prediction == 1:
            res_val = "Heart Disease"
        else:
            res_val = "no Heart Disease"
    return render_template('heart_disease.html',prediction='Patient has {}'.format(res_val))


@app.route('/breast_cancer')
def breast_cancer():
    return render_template('breast_cancer.html')

@app.route('/predict_breast_cancer',methods=['POST'])
def predict_breast_cancer():
    if request.method=='POST':
        input_features = [x for x in request.form.values()]
        features_value = [np.array(input_features)]
    
        features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    
        df = pd.DataFrame(features_value, columns=features_name)
        model = pickle.load(open('breast_Cancer_prediction_svc_model.pkl', 'rb'))
        prediction = model.predict(df)
        
        if prediction == 1:
            res_val = "breast cancer "
        else:
            res_val = "no breast cancer"
        

    return render_template('breast_cancer.html', prediction='Patient has {}'.format(res_val))




@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')
@app.route('/predict_diabetes',methods=['POST'])
def predict_diabetes():
    if request.method=='POST':
        input_features = [x for x in request.form.values()]
        features_value = [np.array(input_features)]
        features_name = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
        df = pd.DataFrame(features_value, columns=features_name)
        model = pickle.load(open('diabetes_saved_model.pkl', 'rb')) 
        prediction=model.predict(df)
        if prediction == 1:
            res_val = "Patient Has Diabetes"
        else:
            res_val = "Patient Has No Diabetes"
    return render_template('diabetes.html',prediction='Patient has {}'.format(res_val))  

if __name__ == "__main__":
    app.run(debug=True)