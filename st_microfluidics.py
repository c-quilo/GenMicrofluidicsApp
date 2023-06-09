# First, import necessary libraries
import streamlit as st
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from VarAE import VAE, encoder, decoder
from plotting import plot_latent_space

st.set_page_config(
    page_title = 'PREMIERE CS3',
    page_icon = '🔬💧',
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

'''
    [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/c-quilo/GenMicrofluidicsApp) 

'''
st.markdown("<br>",unsafe_allow_html=True)

# Streamlit code for the app
st.title('Expanding microfluidics design space with generative AI')

c1, c2, c3, c4, c5 = st.columns([0.5, 1, 1, 1, 1])
c1.image('./Images/Premiere.jpeg', use_column_width=True)
c2.image('./Images/UKRI.png')
st.write(
    """<style>
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Add a slider and a button on the left column
MAPE = 'Waiting...'
time_synth = 'Waiting...'
fig1 = plt.figure(figsize=(20,10))
fig2 = plt.figure(figsize=(20,10))

with st.sidebar:
    input_form = st.form(key='Initial Conditions for training')
    gt_value = input_form.slider('Number of original experiments', 50, 392, 100)
    batch_size = input_form.slider('Batch size', 16, 512, 128)
    epoch_value = input_form.slider('Number of epochs', 0, 500, 100)

    synthetic_value = input_form.slider('Number of synthetic experiments to be generated', 500, 10000, 1000)
    generate_button = input_form.form_submit_button('Train and generate!')

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
            max_ls = np.max(trainingData, 0)

            min = 0
            max = 1
            meanData = np.mean(trainingData, 0)
            stdData = np.std(trainingData, 0)

            data = scaler(trainingData, min_ls, max_ls, min, max)

            vae = VAE(encoder, decoder)
            
            vae.compile(optimizer=keras.optimizers.Nadam())
            vae.fit(data, epochs=epoch_value, batch_size=batch_size)
            st.write('Training done!')

        with st.spinner('Generating synthetic data...'):
            fig1, fig2, MAPE, time_synth = plot_latent_space(vae, data, synthetic_value, min_ls, max_ls, nFeatures, min=0, max=1)
            #fig1, fig2, MAPE, time_synth = plot_latent_space(st.session_state['vae'], st.session_state['data'], synthetic_value, st.session_state['min_ls'], st.session_state['max_ls'], st.session_state['nFeatures'], min=0, max=1)
            del vae
            tf.keras.backend.clear_session()
            MAPE = 100*MAPE
            
st.write(f'MAPE: {MAPE}%')
st.write(f'Time to generate synthetic data: {time_synth} [s]')
st.pyplot(fig1)
st.pyplot(fig2)

# # Call the main function to run the app
# if __name__ == "__main__":
#     main()