from flask import Flask,render_template,jsonify,request
#import requests as req
import pickle
from flask_cors import CORS,cross_origin
#from icecream import ic
import numpy as np
#for aws besntalk use main.py and flask will be application
application=Flask(__name__)
app=application

@app.route('/',methods=['GET'])
@cross_origin()
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def index():
    if request.method=='POST':
        try:
            gre_score=float(request.form['gre_score'])
            toefl_score=float(request.form['toefl_score'])
            university_rating = float(request.form['university_rating'])
            sop = float(request.form['sop'])
            lor = float(request.form['lor'])
            cgpa = float(request.form['cgpa'])
            research = 1.0 if request.form['research']=='yes' else 0.0

            scaler_file='scale.bin'
            model_file='first_model_ready.pickle'

            scaler=pickle.load(open(scaler_file,'rb'))
            model=pickle.load(open(model_file,'rb'))

            data=scaler.transform(np.array([gre_score,toefl_score,university_rating,sop,lor,cgpa,research]).reshape(1,-1))
            #ic(data)

            value=model.predict(data)
            #ic(value)

            return render_template('results.html',prediction=round(100*value[0],2))
        except Exception as e:
            #ic(e)
            return 'something is wrong'
    else:
        return render_template('index.html')





if __name__=='__main__':
    app.run(debug=True)













