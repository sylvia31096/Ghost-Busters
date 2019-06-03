from flask import render_template,request
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

from app import app	

from sklearn.preprocessing import LabelEncoder 
import pickle
import numpy as np


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')

@app.route('/predict_monster',methods = ['POST'])
def predict_monster():
	if request.method == 'POST':
		
		with open((os.path.join(THIS_FOLDER, "python_gnb_model.pkl")), "rb") as file_handler:
				gnb_model = pickle.load(file_handler)
		with open((os.path.join(THIS_FOLDER, "color_encodings.pkl")), "rb") as file_handler:
				le_color = pickle.load(file_handler)
		with open((os.path.join(THIS_FOLDER, "target_class_encodings.pkl")), "rb") as file_handler:
				le_y = pickle.load(file_handler)
		print("data:",request.form)
		bone_length = float(request.form.get("bone_length"))
		rotting_flesh = float(request.form.get("rotting_flesh"))
		hair_length = float(request.form.get("hair_length"))
		has_soul = float(request.form.get("has_soul"))

		
		

		color = le_color.transform(np.array([request.form.get("color")]))

		

		X = np.array([bone_length,rotting_flesh,hair_length,has_soul,color])
		 
			
		
		
		prediction = le_y.inverse_transform(gnb_model.predict([X]))
		#return jsonify(output.tolist())
		return render_template("index.html",prediction=prediction)