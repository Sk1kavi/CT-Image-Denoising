import os
from flask import Flask, render_template, request
import pydicom
import numpy as np
from skimage import restoration, exposure
from numpy.fft import fft2, ifft2, fftshift
import matplotlib
matplotlib.use('Agg')  # Set the backend for server-side rendering
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'  # Path to save the images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Function to compute SNR
def compute_snr(image, noise):
    signal = np.mean(image)
    noise_std = np.std(noise)
    return signal / noise_std

# Function to denoise the CT image
def denoise_ct_image(dicom_path):
    dicom_file = pydicom.dcmread(dicom_path)
    image = dicom_file.pixel_array

    if image.ndim == 3:
        image_slice = image[image.shape[0] // 2, :, :]
    else:
        image_slice = image

    transformed_image = exposure.rescale_intensity(image_slice.astype(np.float32))
    denoised_poisson_image = restoration.denoise_nl_means(transformed_image, h=0.1, fast_mode=True)
    denoised_poisson_image = np.maximum(denoised_poisson_image - 1.0 / 8.0, 0)

    f_transform = fft2(denoised_poisson_image)
    f_transform_shifted = fftshift(f_transform)

    mask = np.ones(f_transform_shifted.shape, dtype=np.float32)
    mask[80:200, 80:200] = 0

    f_transform_filtered = f_transform_shifted * mask
    final_denoised_image = np.abs(ifft2(fftshift(f_transform_filtered)))
    final_denoised_image = restoration.denoise_tv_chambolle(final_denoised_image, weight=0.05)

    final_denoised_image = (final_denoised_image - np.min(final_denoised_image)) / (
        np.max(final_denoised_image) - np.min(final_denoised_image)
    )

    # Compute SNR
    snr_before = compute_snr(image_slice, image_slice - denoised_poisson_image)
    snr_after = compute_snr(final_denoised_image, image_slice - final_denoised_image)

    return image_slice, final_denoised_image, snr_before, snr_after

# Route for the front page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the homepage
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        noised_image, denoised_image, snr_before, snr_after = denoise_ct_image(file_path)

        # Save noised and denoised images to static folder
        noised_image_path = os.path.join(app.config['STATIC_FOLDER'], 'noised_image.png')
        denoised_image_path = os.path.join(app.config['STATIC_FOLDER'], 'denoised_image.png')
        plt.imsave(noised_image_path, noised_image, cmap='gray')
        plt.imsave(denoised_image_path, denoised_image, cmap='gray')

        return render_template(
            'home.html', 
            snr_before=snr_before, 
            snr_after=snr_after, 
            noised_image_url='static/noised_image.png',
            denoised_image_url='static/denoised_image.png'
        )

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
