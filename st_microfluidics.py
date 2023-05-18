# First, import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from VarAE import VAE, encoder, decoder
from plotting import plot_latent_space
st.set_page_config(
    page_title = 'PREMIERE CS3',
    page_icon = ':microscope::droplet:',
    initial_sidebar_state = 'expanded',
    layout = 'wide',
)

# CSS to change the background color
st.markdown("""
    <style>
    .reportview-container {
        background: white
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit code for the app
st.title('Expanding microfluidics design space with generative AI')

# Add a slider and a button on the left column
with st.sidebar:
    input_form = st.form(key='Initial Conditions')
    gt_value = input_form.slider('Number of original experiments', 0, 50, 400)
    synthetic_value = input_form.slider('Number of synthetic experiments to be generated', 0, 100, 10000)
    generate_button = input_form.form_submit_button('Generate synthetic experiments')

    if generate_button:
        with st.spinner('Training...'):

            # Load datasets
            nFeatures = 5
            arrayTraining = scipy.io.loadmat('./data/matTrainingDataSet_{}inputs.mat'.format(nFeatures - 1 ))
            data = np.log(arrayTraining['matTrainingDataSet']['inputs'][0][0]+1)
            data_output = np.squeeze(np.log(arrayTraining['matTrainingDataSet']['output'][0][0] + 1))
            trainingData = np.hstack((data, np.expand_dims(data_output, 1)))[:gt_value, :]

            #Load VAE model
            tf.random.set_seed(42)

            def scaler(x, xmin, xmax, min, max):
                scale = (max - min) / (xmax - xmin)
                xScaled = scale * x + min - xmin * scale
                return xScaled
            min_ls = np.min(trainingData, 0)
            print(min_ls)
            max_ls = np.max(trainingData, 0)
            print(max_ls)
            min = 0
            max = 1
            meanData = np.mean(trainingData, 0)
            stdData = np.std(trainingData, 0)

            data = scaler(trainingData, min_ls, max_ls, min, max)

            print(data.shape)
            vae = VAE(encoder, decoder)
            vae.compile(optimizer=keras.optimizers.Nadam())
            vae.fit(data, epochs=500, batch_size=512)
        with st.spinner('Generating synthetic data...'):
        # Add a scatter plot on the right column
            fig1, fig2, MAPE, time_synth = plot_latent_space(vae, data, synthetic_value, min_ls, max_ls, nFeatures, min=0, max=1)
MAPE = 100*MAPE
st.write(f'MAPE: {MAPE}%')
st.write(f'Time to generate synthetic data: {time_synth} [s]')
st.pyplot(fig1)
st.pyplot(fig2)

# # Call the main function to run the app
# if __name__ == "__main__":
#     main()