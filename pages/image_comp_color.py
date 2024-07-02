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


st.title('Image compressor including color')
st.write('Provide url of the image and get compressed one')

url_image = st.text_input('Enter the URL of the image', '')

if url_image:
    try:
        image = io.imread(url_image)
        
        channels_color = []
        for i in range(3):
            U, sigma_values, V = np.linalg.svd(image[:,:,i])
            sigma = np.zeros((U.shape[0], V.shape[0]))
            np.fill_diagonal(sigma, sigma_values)
            channels_color.append((U, sigma, V))
        
        top_k = st.number_input('Enter Top_k value', 0, min(image.shape[0], image.shape[1]), 1)

        compressed_channels_color = []
        for U, sigma, V in channels_color:
            trunc_U = U[:, :top_k]
            trunc_sigma = sigma[:top_k,:top_k]
            trunc_V = V[:top_k,:]
            compressed_channels_color.append(trunc_U@trunc_sigma@trunc_V)
        
        compressed_image = np.stack(compressed_channels_color, axis=2).astype(np.uint8)
        
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