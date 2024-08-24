from keras.models import load_model
from flask import Flask, render_template, request
from keras_preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

app = Flask(__name__)

CLASS_NAMES =['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

model = load_model(r'C:\Tomato-Disease-Detection\models\beta_1')

def preprocess_image(image, target_size):
    # Convert the image to RGB if it's not already
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize(target_size)
    image = img_to_array(image)
     
    
    # Expand dimensions to match the model input shape
    image = np.expand_dims(image, axis=0)
    
    return image

def prediction(image):
    # Preprocess the image to match the model input requirements
    img = preprocess_image(image, target_size=(256, 256))
    
    # Predict using the model
    predictions = model.predict(img)
    
    
    # Get the class with the highest probability
    class_prediction = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print('Predictions : ', class_prediction)
    return (class_prediction, confidence)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file:
            image = Image.open(file.stream)  # Open the image file using PIL
            result, confidence = prediction(image)
            
            return render_template('index.html', result=result, confidence=confidence*100)
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
