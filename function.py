import face_recognition
import cv2
import cloudinary
import cloudinary.uploader
import os
import numpy as np
from datetime import datetime

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []  # saving the features of the image
        self.known_face_names = []  # saving the name of the image 
        self.frame_resizing = 0.25  # Resize frame for faster processing
        self.reference_face_encoding = None

        # Configure Cloudinary
        cloudinary.config(
            cloud_name="hcvu40dvj",
            api_key="523379171599888",
            api_secret="pD2VU84Ew_KMETn0o-6kdbjPFnU"
        )

    def load_reference_image(self, image_path):
        """
        Load and encode the reference image.
        :param image_path: Path to the reference image
        """
        img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.reference_face_encoding = face_recognition.face_encodings(rgb_img)[0]

    def compare_with_reference(self, image_path):
        """
        Compare a new image with the reference image.
        :param image_path: Path to the new image
        :return: String indicating match result and Cloudinary URL if not match
        """
        img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_img)

        # Check if at least one face is found
        if not face_encodings:
            return "no face found"

        # Compare the first face found in the new image to the reference face encoding
        new_face_encoding = face_encodings[0]
        match = face_recognition.compare_faces([self.reference_face_encoding], new_face_encoding)[0]

        if match:
            return "match"
        else:
            # Upload the image to Cloudinary
            upload_result = cloudinary.uploader.upload(image_path)
            url = upload_result['url']
            return f"not match - image uploaded to {url}"
