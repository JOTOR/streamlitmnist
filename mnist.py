import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow
model = tensorflow.keras.models.load_model('mnist.h5')

st.markdown('# MNIST Digits Classifier')
st.markdown('Developed by: **Jhonnatan Torres**')
st.image('MNIST.png')
st.write('Picture extracted from https://www.researchgate.net/figure/A-subset-of-the-MNIST-database-of-handwritten-digits_fig4_232650721')


st.write("Please draw a digit between Zero (0) and Nine (9)")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # Fixed fill color with some opacity
    stroke_width=35,
    stroke_color='#FFFFFF',
    background_color='#000000',
    background_image=None,
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode='freedraw',
    key="canvas"
)
realtime_update = st.button("Predict")

pred_map = {0: 'Zero (0)',
			1: 'One (1)',
			2: 'Two (2)',
			3: 'Three (3)',
			4: 'Four (4)',
			5: 'Five (5)',
			6: 'Six (6)',
			7: 'Seven (7)',
			8: 'Eight (8)',
			9: 'Nine (9)',
}

if realtime_update == True:
	im = canvas_result.image_data
	imar = im[0:280, 0:280, 0]
	pima = Image.fromarray(np.uint8(imar), 'L')
	pima = pima.resize((28, 28))
	dep = np.array(pima).reshape(1, 28, 28, 1)
	pred = np.argmax(model.predict(dep), axis=1)
	st.write("Predicted Number: ", pred_map[pred[0]])