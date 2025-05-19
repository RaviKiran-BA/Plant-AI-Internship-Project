import os
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from info import disease_info
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('models/model.h5', compile=False)

# Prediction function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    preds = model.predict(x)
    return preds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/product')
def product():
    return render_template('product.html')  # About Us page template


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)
        
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        

        #Mapping original disease names to user-friendly versions
        disease_mapping = {
        'Apple_apple_scab': 'Apple Apple Scab',
        'Apple_black_rot': 'Apple Black Rot',
        'Apple_cedar_apple_rust': 'Apple Cedar Apple Rust',
        'Apple_healthy': 'Apple Healthy',
        'Cassava_bacterial_blight': 'Cassava Bacterial Blight',
        'Cassava_brown_streak': 'Cassava Brown Streak',
        'Cassava_green_mottle': 'Cassava Green Mottle',
        'Cassava_healthy': 'Cassava Healthy',
        'Cassava_mosaic_disease': 'Cassava Mosaic Disease',
        'Cherry_healthy': 'Cherry Healthy',
        'Cherry_powdery_mildew': 'Cherry Powdery Mildew',
        'Chilli_healthy': 'Chilli Healthy',
        'Chilli_leaf_curl': 'Chilli Leaf Curl',
        'Chilli_leaf_spot': 'Chilli Leaf Spot',
        'Chilli_whitefly': 'Chilli Whitefly',
        'Chilli_yellowish': 'Chilli Yellowish',
        'Coffee_cercospora_leaf_spot': 'Coffee Cercospora Leaf Spot',
        'Coffee_healthy': 'Coffee Healthy',
        'Coffee_red_spider_mite': 'Coffee Red Spider Mite',
        'Coffee_rust': 'Coffee Rust',
        'Corn_cercospora_leaf_spot_gray_leaf_spot': 'Corn Cercospora Leaf Spot Gray Leaf Spot',
        'Corn_common_rust': 'Corn Common Rust',
        'Corn_healthy': 'Corn Healthy',
        'Corn_northern_leaf_blight': 'Corn Northern Leaf Blight',
        'Cotton_diseased': 'Cotton Diseased',
        'Cotton_healthy': 'Cotton Healthy',
        'Cucumber_diseased': 'Cucumber Diseased',
        'Cucumber_healthy': 'Cucumber Healthy',
        'Grape_black_rot': 'Grape Black Rot',
        'Grape_esca_black_measles': 'Grape Esca Black Measles',
        'Grape_healthy': 'Grape Healthy',
        'Grape_leaf_blight_isariopsis_leaf_spot': 'Grape Leaf Blight Isariopsis Leaf Spot',
        'Grapevine_black_rot': 'Grapevine Black Rot',
        'Grapevine_esca': 'Grapevine Esca',
        'Grapevine_healthy': 'Grapevine Healthy',
        'Grapevine_leaf_blight': 'Grapevine Leaf Blight',
        'Guava_diseased': 'Guava Diseased',
        'Guava_healthy': 'Guava Healthy',
        'Jamun_diseased': 'Jamun Diseased',
        'Jamun_healthy': 'Jamun Healthy',
        'Lemon_diseased': 'Lemon Diseased',
        'Lemon_healthy': 'Lemon Healthy',
        'Mango_diseased': 'Mango Diseased',
        'Mango_healthy': 'Mango Healthy',
        'Mulberry_healthy': 'Mulberry Healthy',
        'Mulberry_leaf_rust': 'Mulberry Leaf Rust',
        'Mulberry_leaf_spot': 'Mulberry Leaf Spot',
        'Pea_downy_mildew': 'Pea Downy Mildew',
        'Pea_healthy': 'Pea Healthy',
        'Pea_leaf_minner': 'Pea Leaf Minner',
        'Pea_powdery_mildew': 'Pea Powdery Mildew',
        'Peach_bacterial_spot': 'Peach Bacterial Spot',
        'Peach_healthy': 'Peach Healthy',
        'Pepper_bell_bacterial_spot': 'Pepper Bell Bacterial Spot',
        'Pepper_bell_healthy': 'Pepper Bell Healthy',
        'Pomegranate_diseased': 'Pomegranate Diseased',
        'Pomegranate_healthy': 'Pomegranate Healthy',
        'Potato_early_blight': 'Potato Early Blight',
        'Potato_healthy': 'Potato Healthy',
        'Potato_late_blight': 'Potato Late Blight',
        'Rice_brown_spot': 'Rice Brown Spot',
        'Rice_healthy': 'Rice Healthy',
        'Rice_hispa': 'Rice Hispa',
        'Rice_leaf_blast': 'Rice Leaf Blast',
        'Rice_neck_blast': 'Rice Neck Blast',
        'Rose_healthy': 'Rose Healthy',
        'Rose_rust': 'Rose Rust',
        'Rose_sawfly': 'Rose Sawfly',
        'Soyabean_bacterial_blight': 'Soyabean Bacterial Blight',
        'Soyabean_caterpillar': 'Soyabean Caterpillar',
        'Soyabean_diabrotica_speciosa': 'Soyabean Diabrotica Speciosa',
        'Soyabean_downy_mildew': 'Soyabean Downy Mildew',
        'Soyabean_healthy': 'Soyabean Healthy',
        'Soyabean_mosaic_virus': 'Soyabean Mosaic Virus',
        'Soyabean_rust': 'Soyabean Rust',
        'Soyabean_southern_blight': 'Soyabean Southern Blight',
        'Strawberry_healthy': 'Strawberry Healthy',
        'Strawberry_leaf_scorch': 'Strawberry Leaf Scorch',
        'Sugarcane_bacterial_blight': 'Sugarcane Bacterial Blight',
        'Sugarcane_healthy': 'Sugarcane Healthy',
        'Sugarcane_red_rot': 'Sugarcane Red Rot',
        'Sugarcane_red_stripe': 'Sugarcane Red Stripe',
        'Sugarcane_rust': 'Sugarcane Rust',
        'Tea_algal_leaf': 'Tea Algal Leaf',
        'Tea_anthracnose': 'Tea Anthracnose',
        'Tea_bird_eye_spot': 'Tea Bird Eye Spot',
        'Tea_brown_blight': 'Tea Brown Blight',
        'Tea_healthy': 'Tea Healthy',
        'Tea_red_leaf_spot': 'Tea Red Leaf Spot',
        'Tomato_bacterial_spot': 'Tomato Bacterial Spot',
        'Tomato_early_blight': 'Tomato Early Blight',
        'Tomato_healthy': 'Tomato Healthy',
        'Tomato_late_blight': 'Tomato Late Blight',
        'Tomato_leaf_mold': 'Tomato Leaf Mold',
        'Tomato_mosaic_virus': 'Tomato Mosaic Virus',
        'Tomato_septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
        'Tomato_spider_mites_two_spotted_spider_mite': 'Tomato Spider Mites Two-Spotted Spider Mite',
        'Tomato_target_spot': 'Tomato Target Spot',
        'Tomato_yellow_leaf_curl_virus': 'Tomato Yellow Leaf Curl Virus',
        'Wheat_brown_spot': 'Wheat Brown Spot',
        'Wheat_healthy': 'Wheat Healthy',
        'Wheat_septoria': 'Wheat Septoria',
        'Wheat_stripe_rust': 'Wheat Stripe Rust',
        'Wheat_yellow_rust': 'Wheat Yellow Rust'
    }


        
        # Get the predicted disease
        predicted_index = np.argmax(preds)
        predicted_confidence = np.max(preds) * 100  # Convert to percentage
        original_disease = list(disease_mapping.keys())[predicted_index]
        res = disease_mapping[original_disease]
        
        return render_template('result.html', prediction=res, confidence=predicted_confidence)
    return None

@app.route('/suggest')
def cure():
    disease = request.args.get('disease')
    
    # Default to "Healthy" if disease not found in data
    if disease not in disease_info:
        disease = "Healthy" 
    
    disease_details = disease_info[disease]
    
    # Pass the additional fields to the template
    return render_template(
        'suggest.html', 
        disease=disease, 
        description=disease_details.get('description'), 
        cause=disease_details.get('cause'), 
        cure=disease_details.get('cure'), 
        symptoms=disease_details.get('symptoms', []), 
        prevention=disease_details.get('prevention', []), 
        treatment=disease_details.get('treatment', [])
    )



             

if __name__ == '__main__':
    app.run(debug=True)

       