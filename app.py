from flask import Flask, request, jsonify
import tempfile
import os
import cv2
from function import SimpleFacerec
import requests
import logging

app = Flask(__name__)

sfr = SimpleFacerec()

logging.basicConfig(level=logging.DEBUG)

@app.route('/upload_reference', methods=['POST'])
def upload_reference():
    logging.debug("Request headers: %s", request.headers)
    logging.debug("Request data: %s", request.data)

    if not request.is_json:
        logging.debug("Request is not JSON")
        return jsonify({'error': 'Invalid content type. Expected application/json'}), 400

    data = request.get_json()
    logging.debug("Request JSON: %s", data)

    if not data or 'reference_image_url' not in data:
        return jsonify({'error': 'No reference image URL provided'}), 400

    reference_image_url = data['reference_image_url']
    logging.debug("Downloading reference image from URL: %s", reference_image_url)

    try:
        response = requests.get(reference_image_url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error("Failed to download reference image: %s", e)
        return jsonify({'error': 'Failed to download reference image', 'details': str(e)}), 400

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file_path = temp_file.name
    with open(temp_file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    temp_file.close()

    load_result = sfr.load_reference_image(temp_file_path)
    if load_result == "no face found":
        os.unlink(temp_file_path)
        return jsonify({'error': 'No face found in the reference image'}), 400

    os.unlink(temp_file_path)
    return jsonify({'message': 'Reference image uploaded successfully'})

@app.route('/detect', methods=['POST'])
def detect():
    if 'imagefiles' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    imagefiles = request.files.getlist('imagefiles')
    if len(imagefiles) == 0:
        return jsonify({'error': 'No files uploaded'}), 400

    results = []
    for image_file in imagefiles:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file_path = temp_file.name
        image_file.save(temp_file_path)
        temp_file.close()

        result = sfr.compare_with_reference(temp_file_path)
        results.append(result)

        os.unlink(temp_file_path)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
