from flask import Flask, render_template, request
from PIL import Image
import io
from io import BytesIO
import base64
import numpy as np
from celeba_classify import run_model
from similarity import get_similarities


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pca')
def about():
    return render_template('chart_1763.html')

@app.route('/example')
def upload():
    return render_template('example.html')

@app.route('/similarity', methods=['GET', 'POST'])
def similarity():
    if request.method == 'POST':
    # #Retrieve uploaded files
    #     input_file1 = request.files['image1']
    #     image1_data = input_file1.read()
    #     # Convert image data to a PIL Image object
    #     image1 = Image.open(io.BytesIO(image1_data))
    #     # Convert PIL Image to NumPy array
    #     image1_array = np.array(image1)
    #
    #     input_file2 = request.files['image2']
    #     image2_data = input_file2.read()
    #     # Convert image data to a PIL Image object
    #     image2 = Image.open(io.BytesIO(image2_data))
    #     # Convert PIL Image to NumPy array
    #     image2_array = np.array(image2)
    #     sim = get_similarities(image1_array, image2_array)

        input_file1 = request.files['image1']
        image_data1 = Image.open(io.BytesIO(input_file1.read()))
        if image_data1.mode != 'RGB':  # PNG imgs are RGBA
            image_data1 = image_data1.convert('RGB')
        with BytesIO() as buffer:
            image_data1.save(buffer, 'png')
            data1 = base64.encodebytes(buffer.getvalue()).decode('utf-8')
        image1_np = np.array(image_data1)

        input_file2 = request.files['image2']
        image_data2 = Image.open(io.BytesIO(input_file2.read()))
        if image_data2.mode != 'RGB':  # PNG imgs are RGBA
            image_data2 = image_data2.convert('RGB')
        with BytesIO() as buffer:
            image_data2.save(buffer, 'png')
            data2 = base64.encodebytes(buffer.getvalue()).decode('utf-8')
        image2_np = np.array(image_data2)

        #sim = get_similarities(image1_np, image2_np)
        sim = get_similarities(image_data1, image_data2)
        return render_template('results_sim.html', image_data1=data1, image_data2=data2, output=sim)



    else:
        # Render the upload template for GET requests
        return render_template('similarity.html')

# @app.route('/classify', methods=['GET', 'POST'])
# def classify_image():
#     if request.method == 'POST':
#         # Get the uploaded image file from the request
#         image_file = request.files['image']
#         image_data = Image.open(io.BytesIO(image_file.read()))
#         with BytesIO() as buffer:
#             image_data.save(buffer, 'png')
#             data = base64.encodebytes(buffer.getvalue()).decode('utf-8')
#         image_np = np.array(image_data)
#
#         features_output = run_model(image_np)
#         # Render the results template with the output string
#         return render_template('results.html',  image_data=data, output=features_output)
#     else:
#         # Render the upload template for GET requests
#         return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)