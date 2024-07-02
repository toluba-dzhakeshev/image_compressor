import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris 
from sklearn.decomposition import TruncatedSVD, PCA
from skimage import io

from sklearn.preprocessing import StandardScaler


st.title('Image compressor')
st.write('Provide url of the image and get compressed one')

url_image = st.text_input('Enter the URL of the image', '')

if url_image:
    try:
        image = io.imread(url_image)[:,:,0]
        
        U, sigma_values, V = np.linalg.svd(image)
        sigma = np.zeros(shape = image.shape)
        np.fill_diagonal(sigma, sigma_values)
        
        top_k = st.number_input('Enter Top_k value', 0, min(image.shape))

        trunc_U = U[:, :top_k]
        trunc_sigma = sigma[:top_k,:top_k]
        trunc_V = V[:top_k,:]

        compressed_image = trunc_U@trunc_sigma@trunc_V
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(image, cmap='grey')
        ax[0].set_title('Original image')

        ax[1].imshow(compressed_image, cmap='grey')
        ax[1].set_title('Compressed image')

        st.pyplot(fig)
    
    except:
        st.write(f'Error, something wrong')

else:
    st.stop()
