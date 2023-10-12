from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
import os

app = Flask(__name__)

# define the route for the quiz form
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    if username == 'admin' and password == '123456':
        return redirect(url_for('success'))
    else:
        error = 'Invalid username or password. Please try again.'
        return render_template('login.html', error=error)

@app.route('/success')
def success():
    return render_template('detectionPage.html')

@app.route('/result', methods=["POST"])
def result():
    model = tf.keras.models.load_model("./modelFile/eyeDisease.h5")
    CLASSES = [ 'cataract', 'diabetic_retinopathy', 'glaucoma', 'normal' ]
    
    file = request.files['image']
    file_contents = file.read()
    filename = os.path.join('static', file.filename)
    with open('static/css/img/result.jpg', 'wb') as f:
        f.write(file_contents)
        
    img = tf.keras.preprocessing.image.load_img(BytesIO(file_contents), target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    predicted_class = np.argmax(preds)

    # return the predicted class to the HTML page
    result = CLASSES[predicted_class]
    return render_template('detectionPage.html', result=result)

if __name__ == '__main__':
    app.run(debug = True)