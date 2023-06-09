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
    gt_value = input_form.slider('Number of original experiments', 50, 100, 392)
    batch_size = input_form.slider('Batch size', 16, 128, 512)
    epoch_value = input_form.slider('Number of epochs', 1, 100, 500)
    train_button = input_form.form_submit_button('Train!')
    
    output_form = st.form(key='Data generation')

    synthetic_value = output_form.slider('Number of synthetic experiments to be generated', 500, 1000, 10000)
    generate_button = output_form.form_submit_button('Generate!')

    if train_button:
        with st.spinner('Training...'):

            # Load datasets

            arrayTraining = scipy.io.loadmat('./data/matTrainingDataSet_{}inputs.mat'.format(4))
            data = np.log(arrayTraining['matTrainingDataSet']['inputs'][0][0]+1)
            data_output = np.squeeze(np.log(arrayTraining['matTrainingDataSet']['output'][0][0] + 1))
            trainingData = np.hstack((data, np.expand_dims(data_output, 1)))[:gt_value, :]
            if 'data' not in st.session_state:
                st.session_state['data'] = data
            #Load VAE model
            tf.random.set_seed(42)

            def scaler(x, xmin, xmax, min, max):
                scale = (max - min) / (xmax - xmin)
                xScaled = scale * x + min - xmin * scale
                return xScaled
            min_ls = np.min(trainingData, 0)
            if 'min_ls' not in st.session_state:
                st.session_state['min_ls'] = min_ls
            max_ls = np.max(trainingData, 0)
            if 'max_ls' not in st.session_state:
                st.session_state['max_ls'] = max_ls
            min = 0
            max = 1
            meanData = np.mean(trainingData, 0)
            stdData = np.std(trainingData, 0)

            data = scaler(trainingData, min_ls, max_ls, min, max)

            print(data.shape)
            vae = VAE(encoder, decoder)
            vae.compile(optimizer=keras.optimizers.Nadam())
            vae.fit(data, epochs=epoch_value, batch_size=batch_size)
            st.write('Training done!')
            if 'vae' not in st.session_state:
                st.session_state['vae'] = vae
    if generate_button:
        with st.spinner('Generating synthetic data...'):
            #fig1, fig2, MAPE, time_synth = plot_latent_space(vae, data, synthetic_value, min_ls, max_ls, nFeatures, min=0, max=1)
            fig1, fig2, MAPE, time_synth = plot_latent_space(st.session_state['vae'], st.session_state['data'], synthetic_value, st.session_state['min_ls'], st.session_state['max_ls'], 4, min=0, max=1)

            MAPE = 100*MAPE
st.write(f'MAPE: {MAPE}%')
st.write(f'Time to generate synthetic data: {time_synth} [s]')
st.pyplot(fig1)
st.pyplot(fig2)

# # Call the main function to run the app
# if __name__ == "__main__":
#     main()