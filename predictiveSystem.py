# -*- coding: utf-8 -*-


import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler



loaded_model = pickle.load(open('C:/STUDY/ML/part-5/deploying through Streamlit/trained_model.sav' , 'rb'))

input_data  = (1,85,66,29,0,26.6,0.351,31)
# changing the input data into numpy array
input_data_as_numpy_array = np.asarray(input_data)

# rehape the array as we are predicting  for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)



prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print("The person is not  diabetic")

else:
    print("The person is diabetic")


