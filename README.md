# tugaspreprocessing-tugas3

Open In Colab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
     

#pemanggilan data
data = pd.read_csv ('student_performance_updated_1000.csv')
     

data.head()
     
StudentID	Name	Gender	AttendanceRate	StudyHoursPerWeek	PreviousGrade	ExtracurricularActivities	ParentalSupport	FinalGrade	Study Hours	Attendance (%)	Online Classes Taken
0	1.0	John	Male	85.0	15.0	78.0	1.0	High	80.0	4.8	59.0	False
1	2.0	Sarah	Female	90.0	20.0	85.0	2.0	Medium	87.0	2.2	70.0	True
2	3.0	Alex	Male	78.0	10.0	65.0	0.0	Low	68.0	4.6	92.0	False
3	4.0	Michael	Male	92.0	25.0	90.0	3.0	High	92.0	2.9	96.0	False
4	5.0	Emma	Female	NaN	18.0	82.0	2.0	Medium	85.0	4.1	97.0	True

data.tail(5)
     
StudentID	Name	Gender	AttendanceRate	StudyHoursPerWeek	PreviousGrade	ExtracurricularActivities	ParentalSupport	FinalGrade	Study Hours	Attendance (%)	Online Classes Taken
995	NaN	Kenneth Murray	Male	85.0	20.0	NaN	1.0	High	72.0	0.8	80.0	True
996	4497.0	Amy Stout	Female	91.0	NaN	86.0	0.0	High	90.0	3.9	80.0	True
997	1886.0	NaN	Male	85.0	8.0	82.0	2.0	Low	68.0	0.4	54.0	False
998	7636.0	Joseph Sherman	Male	88.0	17.0	60.0	2.0	High	85.0	0.9	53.0	True
999	8021.0	Maria Walls	Female	88.0	10.0	90.0	1.0	Medium	NaN	2.4	94.0	True

#melihat statistika deskriptif
data.describe()
     
StudentID	AttendanceRate	StudyHoursPerWeek	PreviousGrade	ExtracurricularActivities	FinalGrade	Study Hours	Attendance (%)
count	960.000000	960.000000	950.000000	967.000000	957.000000	960.000000	976.000000	959.000000
mean	5416.019792	85.510417	17.630526	77.598759	1.520376	80.030208	2.406967	77.248175
std	2653.748319	7.332125	6.272132	10.006640	1.046439	9.493652	1.620267	19.298148
min	1.000000	70.000000	8.000000	60.000000	0.000000	62.000000	-5.000000	50.000000
25%	3113.500000	82.000000	12.000000	70.000000	1.000000	72.000000	1.200000	63.000000
50%	5396.500000	88.000000	18.000000	78.000000	1.000000	80.000000	2.500000	76.000000
75%	7754.750000	91.000000	22.000000	86.000000	2.000000	88.000000	3.700000	89.000000
max	9998.000000	95.000000	30.000000	90.000000	3.000000	92.000000	5.000000	200.000000

data.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 12 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   StudentID                  960 non-null    float64
 1   Name                       966 non-null    object 
 2   Gender                     952 non-null    object 
 3   AttendanceRate             960 non-null    float64
 4   StudyHoursPerWeek          950 non-null    float64
 5   PreviousGrade              967 non-null    float64
 6   ExtracurricularActivities  957 non-null    float64
 7   ParentalSupport            978 non-null    object 
 8   FinalGrade                 960 non-null    float64
 9   Study Hours                976 non-null    float64
 10  Attendance (%)             959 non-null    float64
 11  Online Classes Taken       975 non-null    object 
dtypes: float64(8), object(4)
memory usage: 93.9+ KB
Berdasarkan output di atas, dapat disimpulkan bahwa dataset tersebut memiliki total 12 kolom, dengan jumlah maksimal baris untuk setiap kolom sebanyak 1000 baris. Akan tetapi, terdapat beberapa kolom yang memiliki jumlah baris kurang dari 1000, sehingga mengindikasikan adanya data yang hilang (missing values). Oleh karena itu, akan dilakukan proses identifikasi lebih lanjut terhadap kolom-kolom tersebut untuk mengetahui pola dan penanganan yang tepat.

Deteksi Missing value

data.isnull()
     
StudentID	Name	Gender	AttendanceRate	StudyHoursPerWeek	PreviousGrade	ExtracurricularActivities	ParentalSupport	FinalGrade	Study Hours	Attendance (%)	Online Classes Taken
0	False	False	False	False	False	False	False	False	False	False	False	False
1	False	False	False	False	False	False	False	False	False	False	False	False
2	False	False	False	False	False	False	False	False	False	False	False	False
3	False	False	False	False	False	False	False	False	False	False	False	False
4	False	False	False	True	False	False	False	False	False	False	False	False
...	...	...	...	...	...	...	...	...	...	...	...	...
995	True	False	False	False	False	True	False	False	False	False	False	False
996	False	False	False	False	True	False	False	False	False	False	False	False
997	False	True	False	False	False	False	False	False	False	False	False	False
998	False	False	False	False	False	False	False	False	False	False	False	False
999	False	False	False	False	False	False	False	False	True	False	False	False
1000 rows × 12 columns


np.sum(data.isnull())
     
/usr/local/lib/python3.12/dist-packages/numpy/_core/fromnumeric.py:84: FutureWarning: The behavior of DataFrame.sum with axis=None is deprecated, in a future version this will reduce over both axes and return a scalar. To retain the old behavior, pass axis=0 (or do not pass axis)
  return reduction(axis=axis, out=out, **passkwargs)
0
StudentID	40
Name	34
Gender	48
AttendanceRate	40
StudyHoursPerWeek	50
PreviousGrade	33
ExtracurricularActivities	43
ParentalSupport	22
FinalGrade	40
Study Hours	24
Attendance (%)	41
Online Classes Taken	25

dtype: int64
Berdasarkan output summarize tersebut, diperoleh bahwa kolom yang mengandung Missing Values (blanks/N/A dalam Python: NaN) adalah kolom StudentID, Name, Gender, AttendanceRate, StudyHoursPerWeek, PreviousGrade, ExtracurricularActivities, ParentalSupport, FinalGrade, Study Hours, Attendance (%), dan Online Classes Taken.

Jumlah data yang hilang pada masing-masing kolom bervariasi, di mana kolom StudyHoursPerWeek memiliki jumlah missing values terbanyak, sedangkan kolom ParentalSupport memiliki jumlah missing values paling sedikit. Hal ini menunjukkan bahwa hampir seluruh kolom dalam dataset masih memiliki data yang tidak lengkap, sehingga perlu dilakukan penanganan lebih lanjut sebelum proses analisis dilakukan.


#mengetahui jumlah missing value di seluruh dataset
data.isnull().sum().sum()
     
np.int64(440)
HANDLING MISSING VALUE
Dalam Machine Learning, missing values merupakan masalah umum yang harus ditangani sebelum model dapat digunakan. Berdasarkan hasil eksplorasi data, diketahui bahwa hampir seluruh kolom dalam dataset memiliki nilai yang hilang (NaN), sehingga diperlukan penanganan yang tepat sesuai dengan jenis datanya.

1. StudentID

#melihat isi dalam kolom
data['StudentID'].unique()
     
array([1.000e+00, 2.000e+00, 3.000e+00, 4.000e+00, 5.000e+00, 6.000e+00,
       7.000e+00, 8.000e+00, 9.000e+00, 1.000e+01, 2.824e+03, 7.912e+03,
       4.611e+03, 6.514e+03,       nan, 2.139e+03, 8.573e+03, 5.386e+03,
       3.340e+03, 6.930e+03, 2.040e+03, 9.797e+03, 9.201e+03, 9.837e+03,
       1.949e+03, 3.705e+03, 8.177e+03, 1.117e+03, 4.510e+03, 6.804e+03,
       4.139e+03, 2.604e+03, 1.960e+03, 3.536e+03, 1.821e+03, 4.853e+03,
       1.152e+03, 2.127e+03, 5.905e+03, 6.617e+03, 1.058e+03, 3.442e+03,
       6.862e+03, 3.900e+03, 7.267e+03, 1.387e+03, 1.452e+03, 8.149e+03,
       6.967e+03, 7.691e+03, 5.930e+03, 8.618e+03, 2.530e+03, 8.461e+03,
       2.746e+03, 3.184e+03, 9.976e+03, 4.919e+03, 3.267e+03, 7.812e+03,
       8.814e+03, 4.595e+03, 9.751e+03, 1.823e+03, 7.906e+03, 1.508e+03,
       8.619e+03, 2.771e+03, 2.669e+03, 9.289e+03, 8.135e+03, 6.280e+03,
       1.470e+03, 1.166e+03, 8.342e+03, 8.753e+03, 2.976e+03, 2.647e+03,
       1.745e+03, 8.216e+03, 3.485e+03, 9.561e+03, 4.404e+03, 8.381e+03,
       4.824e+03, 2.169e+03, 7.594e+03, 5.102e+03, 8.708e+03, 7.690e+03,
       8.451e+03, 7.770e+03, 9.998e+03, 6.091e+03, 9.165e+03, 4.122e+03,
       9.022e+03, 7.325e+03, 9.617e+03, 7.512e+03, 4.846e+03, 1.474e+03,
       3.950e+03, 4.804e+03, 8.237e+03, 2.747e+03, 5.497e+03, 4.115e+03,
       2.476e+03, 3.695e+03, 6.992e+03, 7.242e+03, 2.035e+03, 2.592e+03,
       7.797e+03, 6.942e+03, 2.501e+03, 4.394e+03, 9.850e+03, 1.258e+03,
       6.931e+03, 1.996e+03, 3.430e+03, 8.433e+03, 2.099e+03, 6.400e+03,
       3.535e+03, 8.030e+03, 3.972e+03, 7.163e+03, 1.861e+03, 7.785e+03,
       9.633e+03, 5.186e+03, 9.196e+03, 7.626e+03, 6.928e+03, 6.562e+03,
       6.752e+03, 3.902e+03, 6.179e+03, 6.766e+03, 9.307e+03, 9.214e+03,
       5.059e+03, 7.697e+03, 5.128e+03, 3.112e+03, 9.331e+03, 4.711e+03,
       6.517e+03, 1.715e+03, 8.684e+03, 4.271e+03, 5.670e+03, 6.727e+03,
       4.678e+03, 8.685e+03, 2.821e+03, 2.029e+03, 5.772e+03, 9.576e+03,
       7.338e+03, 6.447e+03, 8.840e+03, 6.170e+03, 4.060e+03, 9.937e+03,
       3.730e+03, 2.229e+03, 3.068e+03, 7.077e+03, 5.089e+03, 8.311e+03,
       8.292e+03, 6.922e+03, 8.657e+03, 2.208e+03, 3.114e+03, 6.669e+03,
       9.234e+03, 1.674e+03, 9.159e+03, 8.202e+03, 6.658e+03, 8.366e+03,
       6.123e+03, 5.681e+03, 7.563e+03, 4.261e+03, 3.440e+03, 9.311e+03,
       1.436e+03, 3.421e+03, 4.340e+03, 8.837e+03, 8.932e+03, 9.011e+03,
       7.464e+03, 8.849e+03, 4.972e+03, 3.252e+03, 5.676e+03, 3.880e+03,
       6.489e+03, 6.417e+03, 3.875e+03, 5.285e+03, 2.923e+03, 7.813e+03,
       2.279e+03, 2.904e+03, 8.484e+03, 5.703e+03, 1.372e+03, 5.516e+03,
       8.314e+03, 3.295e+03, 7.775e+03, 1.768e+03, 4.948e+03, 7.363e+03,
       1.844e+03, 8.854e+03, 4.974e+03, 6.053e+03, 7.073e+03, 5.836e+03,
       5.216e+03, 9.882e+03, 2.113e+03, 8.993e+03, 1.280e+03, 6.807e+03,
       5.914e+03, 3.807e+03, 8.790e+03, 7.288e+03, 7.499e+03, 4.043e+03,
       3.527e+03, 2.674e+03, 2.468e+03, 2.387e+03, 3.390e+03, 5.040e+03,
       7.224e+03, 8.848e+03, 6.339e+03, 1.440e+03, 5.139e+03, 9.907e+03,
       1.725e+03, 8.303e+03, 5.840e+03, 6.706e+03, 1.370e+03, 8.570e+03,
       8.037e+03, 6.993e+03, 7.367e+03, 2.388e+03, 7.852e+03, 3.733e+03,
       8.939e+03, 8.272e+03, 3.755e+03, 9.880e+03, 5.305e+03, 3.719e+03,
       6.302e+03, 1.971e+03, 9.019e+03, 7.600e+03, 8.511e+03, 6.533e+03,
       3.035e+03, 5.489e+03, 4.836e+03, 2.690e+03, 3.688e+03, 3.293e+03,
       5.065e+03, 1.894e+03, 6.185e+03, 1.273e+03, 7.939e+03, 3.545e+03,
       5.005e+03, 2.956e+03, 3.735e+03, 2.262e+03, 2.685e+03, 3.458e+03,
       5.444e+03, 7.568e+03, 5.777e+03, 6.488e+03, 5.584e+03, 2.565e+03,
       5.778e+03, 8.404e+03, 4.522e+03, 3.848e+03, 2.989e+03, 5.240e+03,
       1.699e+03, 3.831e+03, 5.835e+03, 9.614e+03, 1.959e+03, 9.743e+03,
       1.462e+03, 1.621e+03, 3.137e+03, 9.622e+03, 8.157e+03, 6.026e+03,
       4.619e+03, 6.500e+03, 1.020e+03, 2.661e+03, 2.715e+03, 8.909e+03,
       7.705e+03, 1.893e+03, 4.390e+03, 2.493e+03, 5.315e+03, 4.727e+03,
       5.896e+03, 2.970e+03, 4.308e+03, 5.525e+03, 2.730e+03, 1.613e+03,
       1.807e+03, 4.734e+03, 9.303e+03, 5.087e+03, 6.640e+03, 4.626e+03,
       1.138e+03, 4.575e+03, 5.631e+03, 4.086e+03, 9.148e+03, 6.622e+03,
       5.813e+03, 4.659e+03, 1.879e+03, 3.470e+03, 2.324e+03, 3.595e+03,
       1.076e+03, 1.749e+03, 6.615e+03, 1.359e+03, 1.785e+03, 3.508e+03,
       7.862e+03, 6.416e+03, 2.659e+03, 5.760e+03, 6.384e+03, 8.479e+03,
       2.699e+03, 9.598e+03, 6.664e+03, 3.709e+03, 7.607e+03, 2.944e+03,
       9.328e+03, 6.952e+03, 2.741e+03, 7.437e+03, 8.912e+03, 7.859e+03,
       9.975e+03, 3.913e+03, 9.747e+03, 4.275e+03, 6.876e+03, 1.769e+03,
       4.695e+03, 4.063e+03, 5.178e+03, 9.037e+03, 7.814e+03, 9.691e+03,
       1.653e+03, 9.426e+03, 9.814e+03, 2.181e+03, 2.154e+03, 2.938e+03,
       5.748e+03, 3.561e+03, 1.182e+03, 4.441e+03, 2.877e+03, 5.107e+03,
       5.671e+03, 6.971e+03, 5.533e+03, 3.879e+03, 7.637e+03, 7.430e+03,
       4.913e+03, 9.485e+03, 5.314e+03, 9.460e+03, 3.713e+03, 4.773e+03,
       1.367e+03, 1.916e+03, 2.404e+03, 3.979e+03, 2.778e+03, 9.706e+03,
       9.553e+03, 2.600e+03, 5.223e+03, 6.175e+03, 6.990e+03, 5.157e+03,
       2.642e+03, 2.492e+03, 7.296e+03, 4.304e+03, 6.893e+03, 9.482e+03,
       5.530e+03, 2.185e+03, 8.923e+03, 4.610e+03, 5.706e+03, 3.994e+03,
       6.167e+03, 1.887e+03, 8.490e+03, 6.924e+03, 4.013e+03, 4.744e+03,
       8.749e+03, 5.915e+03, 2.577e+03, 1.324e+03, 5.663e+03, 3.745e+03,
       6.191e+03, 7.782e+03, 4.450e+03, 9.584e+03, 8.260e+03, 8.654e+03,
       3.551e+03, 8.816e+03, 8.161e+03, 9.464e+03, 8.460e+03, 4.046e+03,
       9.086e+03, 2.314e+03, 1.206e+03, 9.538e+03, 5.150e+03, 6.316e+03,
       3.539e+03, 1.979e+03, 8.187e+03, 9.892e+03, 2.128e+03, 4.039e+03,
       4.110e+03, 9.572e+03, 7.366e+03, 1.214e+03, 1.955e+03, 4.784e+03,
       2.728e+03, 4.943e+03, 5.660e+03, 5.521e+03, 4.817e+03, 8.905e+03,
       7.266e+03, 3.509e+03, 7.529e+03, 9.589e+03, 8.245e+03, 2.841e+03,
       8.373e+03, 9.991e+03, 5.653e+03, 6.901e+03, 8.940e+03, 6.465e+03,
       9.212e+03, 3.248e+03, 6.395e+03, 6.062e+03, 4.016e+03, 3.489e+03,
       4.014e+03, 5.421e+03, 8.496e+03, 9.530e+03, 7.347e+03, 7.440e+03,
       2.477e+03, 9.383e+03, 3.656e+03, 3.155e+03, 9.084e+03, 5.009e+03,
       3.381e+03, 5.659e+03, 8.001e+03, 4.425e+03, 7.996e+03, 9.069e+03,
       4.396e+03, 5.168e+03, 7.488e+03, 8.888e+03, 9.779e+03, 7.147e+03,
       6.152e+03, 9.192e+03, 4.361e+03, 6.460e+03, 6.257e+03, 3.580e+03,
       5.440e+03, 1.831e+03, 8.304e+03, 2.593e+03, 6.954e+03, 8.842e+03,
       8.903e+03, 3.835e+03, 3.886e+03, 9.098e+03, 8.421e+03, 5.989e+03,
       2.022e+03, 8.073e+03, 3.278e+03, 3.200e+03, 7.303e+03, 9.220e+03,
       2.854e+03, 5.017e+03, 8.681e+03, 3.744e+03, 5.780e+03, 8.317e+03,
       6.653e+03, 7.222e+03, 6.172e+03, 5.183e+03, 9.972e+03, 4.594e+03,
       2.995e+03, 4.018e+03, 3.915e+03, 9.857e+03, 5.105e+03, 7.411e+03,
       8.898e+03, 6.985e+03, 4.163e+03, 1.268e+03, 9.791e+03, 1.750e+03,
       2.742e+03, 6.312e+03, 3.185e+03, 5.648e+03, 3.859e+03, 4.144e+03,
       5.528e+03, 6.966e+03, 9.804e+03, 6.340e+03, 4.276e+03, 8.454e+03,
       6.503e+03, 3.804e+03, 2.416e+03, 5.041e+03, 1.724e+03, 8.412e+03,
       5.846e+03, 8.452e+03, 7.305e+03, 2.326e+03, 6.989e+03, 9.841e+03,
       5.540e+03, 9.714e+03, 5.724e+03, 5.803e+03, 4.097e+03, 8.965e+03,
       7.093e+03, 3.986e+03, 7.056e+03, 9.948e+03, 6.549e+03, 8.306e+03,
       4.206e+03, 6.915e+03, 8.300e+03, 2.449e+03, 4.264e+03, 8.593e+03,
       7.834e+03, 8.872e+03, 1.958e+03, 1.798e+03, 9.908e+03, 3.243e+03,
       3.277e+03, 3.108e+03, 2.273e+03, 4.365e+03, 3.630e+03, 5.617e+03,
       8.203e+03, 3.025e+03, 6.375e+03, 7.672e+03, 2.723e+03, 3.764e+03,
       5.563e+03, 2.770e+03, 2.117e+03, 2.253e+03, 1.517e+03, 5.965e+03,
       6.779e+03, 6.591e+03, 4.844e+03, 1.612e+03, 6.504e+03, 2.234e+03,
       2.552e+03, 5.945e+03, 9.971e+03, 8.133e+03, 7.649e+03, 7.914e+03,
       1.691e+03, 2.479e+03, 9.133e+03, 7.928e+03, 9.034e+03, 6.947e+03,
       5.144e+03, 5.929e+03, 1.555e+03, 5.275e+03, 4.010e+03, 3.585e+03,
       5.270e+03, 4.677e+03, 1.231e+03, 5.717e+03, 8.506e+03, 7.935e+03,
       3.938e+03, 3.323e+03, 5.312e+03, 1.871e+03, 8.150e+03, 3.425e+03,
       1.067e+03, 2.998e+03, 8.439e+03, 4.463e+03, 3.916e+03, 2.975e+03,
       9.021e+03, 4.053e+03, 9.115e+03, 5.073e+03, 7.766e+03, 2.866e+03,
       3.213e+03, 8.139e+03, 8.295e+03, 6.469e+03, 1.606e+03, 1.994e+03,
       2.915e+03, 6.633e+03, 3.148e+03, 8.911e+03, 4.845e+03, 2.043e+03,
       6.820e+03, 2.817e+03, 3.257e+03, 2.396e+03, 8.025e+03, 2.389e+03,
       1.947e+03, 9.647e+03, 7.627e+03, 1.383e+03, 7.110e+03, 8.424e+03,
       7.187e+03, 7.751e+03, 9.502e+03, 3.364e+03, 1.534e+03, 1.659e+03,
       3.328e+03, 2.106e+03, 8.786e+03, 2.463e+03, 2.450e+03, 9.877e+03,
       1.310e+03, 9.052e+03, 2.878e+03, 3.334e+03, 9.076e+03, 7.676e+03,
       5.990e+03, 9.806e+03, 1.985e+03, 8.077e+03, 5.473e+03, 2.270e+03,
       3.044e+03, 4.137e+03, 1.116e+03, 2.070e+03, 1.378e+03, 7.666e+03,
       4.696e+03, 1.456e+03, 1.891e+03, 3.045e+03, 1.897e+03, 7.581e+03,
       8.014e+03, 6.098e+03, 1.939e+03, 2.120e+03, 5.909e+03, 4.772e+03,
       7.558e+03, 8.215e+03, 3.855e+03, 6.209e+03, 5.049e+03, 2.062e+03,
       9.609e+03, 4.325e+03, 7.981e+03, 8.236e+03, 5.096e+03, 4.728e+03,
       6.581e+03, 7.608e+03, 7.247e+03, 5.991e+03, 2.138e+03, 2.126e+03,
       9.615e+03, 6.681e+03, 8.501e+03, 9.810e+03, 9.264e+03, 8.567e+03,
       6.359e+03, 8.713e+03, 3.759e+03, 8.242e+03, 4.305e+03, 4.148e+03,
       4.105e+03, 1.245e+03, 1.719e+03, 5.127e+03, 8.899e+03, 9.248e+03,
       2.300e+03, 4.900e+03, 1.388e+03, 4.644e+03, 4.153e+03, 4.871e+03,
       5.966e+03, 7.188e+03, 7.233e+03, 8.688e+03, 7.113e+03, 3.660e+03,
       7.313e+03, 5.281e+03, 7.297e+03, 9.202e+03, 5.407e+03, 9.600e+03,
       3.981e+03, 5.249e+03, 2.052e+03, 1.685e+03, 8.961e+03, 7.528e+03,
       3.275e+03, 9.436e+03, 8.279e+03, 2.465e+03, 3.016e+03, 6.623e+03,
       2.460e+03, 6.092e+03, 9.515e+03, 9.189e+03, 5.480e+03, 4.687e+03,
       6.528e+03, 9.978e+03, 8.239e+03, 2.670e+03, 2.922e+03, 5.959e+03,
       2.353e+03, 2.549e+03, 7.640e+03, 4.399e+03, 3.110e+03, 6.956e+03,
       3.291e+03, 4.884e+03, 5.544e+03, 8.472e+03, 7.097e+03, 5.425e+03,
       5.476e+03, 3.090e+03, 7.064e+03, 5.519e+03, 2.256e+03, 7.946e+03,
       1.119e+03, 9.577e+03, 7.211e+03, 5.337e+03, 8.999e+03, 4.050e+03,
       7.729e+03, 8.017e+03, 1.095e+03, 6.322e+03, 4.874e+03, 8.027e+03,
       8.031e+03, 4.758e+03, 2.489e+03, 2.005e+03, 5.046e+03, 2.650e+03,
       3.134e+03, 7.292e+03, 4.160e+03, 9.741e+03, 6.996e+03, 9.723e+03,
       4.015e+03, 9.254e+03, 5.781e+03, 1.022e+03, 5.831e+03, 7.258e+03,
       3.550e+03, 3.885e+03, 4.077e+03, 1.193e+03, 5.385e+03, 3.281e+03,
       9.906e+03, 1.019e+03, 6.906e+03, 3.461e+03, 1.035e+03, 5.562e+03,
       7.301e+03, 1.046e+03, 3.283e+03, 2.920e+03, 3.591e+03, 5.698e+03,
       8.443e+03, 8.674e+03, 2.776e+03, 5.410e+03, 6.428e+03, 5.753e+03,
       3.638e+03, 7.350e+03, 1.244e+03, 2.116e+03, 7.701e+03, 3.592e+03,
       2.787e+03, 4.497e+03, 1.886e+03, 7.636e+03, 8.021e+03])

#jumlah missing value
data['StudentID'].isnull().sum()
     
np.int64(40)
Kolom StudentID merupakan identitas unik setiap siswa, sehingga tidak memiliki makna jika diisi dengan rata-rata atau median. Oleh karena itu, penanganan yang paling tepat adalah:

Menghapus baris yang memiliki missing values, atau Menghapus kolom jika tidak digunakan dalam analisis


#mengisi missing value
data = data.dropna(subset=['StudentID'])
     

#jumlah missng value
np.sum(data['StudentID'].isnull())
     
np.int64(0)

#melihat dalam kolom setelah mengisi missing value
data['StudentID'].unique()
     
array([1.000e+00, 2.000e+00, 3.000e+00, 4.000e+00, 5.000e+00, 6.000e+00,
       7.000e+00, 8.000e+00, 9.000e+00, 1.000e+01, 2.824e+03, 7.912e+03,
       4.611e+03, 6.514e+03, 2.139e+03, 8.573e+03, 5.386e+03, 3.340e+03,
       6.930e+03, 2.040e+03, 9.797e+03, 9.201e+03, 9.837e+03, 1.949e+03,
       3.705e+03, 8.177e+03, 1.117e+03, 4.510e+03, 6.804e+03, 4.139e+03,
       2.604e+03, 1.960e+03, 3.536e+03, 1.821e+03, 4.853e+03, 1.152e+03,
       2.127e+03, 5.905e+03, 6.617e+03, 1.058e+03, 3.442e+03, 6.862e+03,
       3.900e+03, 7.267e+03, 1.387e+03, 1.452e+03, 8.149e+03, 6.967e+03,
       7.691e+03, 5.930e+03, 8.618e+03, 2.530e+03, 8.461e+03, 2.746e+03,
       3.184e+03, 9.976e+03, 4.919e+03, 3.267e+03, 7.812e+03, 8.814e+03,
       4.595e+03, 9.751e+03, 1.823e+03, 7.906e+03, 1.508e+03, 8.619e+03,
       2.771e+03, 2.669e+03, 9.289e+03, 8.135e+03, 6.280e+03, 1.470e+03,
       1.166e+03, 8.342e+03, 8.753e+03, 2.976e+03, 2.647e+03, 1.745e+03,
       8.216e+03, 3.485e+03, 9.561e+03, 4.404e+03, 8.381e+03, 4.824e+03,
       2.169e+03, 7.594e+03, 5.102e+03, 8.708e+03, 7.690e+03, 8.451e+03,
       7.770e+03, 9.998e+03, 6.091e+03, 9.165e+03, 4.122e+03, 9.022e+03,
       7.325e+03, 9.617e+03, 7.512e+03, 4.846e+03, 1.474e+03, 3.950e+03,
       4.804e+03, 8.237e+03, 2.747e+03, 5.497e+03, 4.115e+03, 2.476e+03,
       3.695e+03, 6.992e+03, 7.242e+03, 2.035e+03, 2.592e+03, 7.797e+03,
       6.942e+03, 2.501e+03, 4.394e+03, 9.850e+03, 1.258e+03, 6.931e+03,
       1.996e+03, 3.430e+03, 8.433e+03, 2.099e+03, 6.400e+03, 3.535e+03,
       8.030e+03, 3.972e+03, 7.163e+03, 1.861e+03, 7.785e+03, 9.633e+03,
       5.186e+03, 9.196e+03, 7.626e+03, 6.928e+03, 6.562e+03, 6.752e+03,
       3.902e+03, 6.179e+03, 6.766e+03, 9.307e+03, 9.214e+03, 5.059e+03,
       7.697e+03, 5.128e+03, 3.112e+03, 9.331e+03, 4.711e+03, 6.517e+03,
       1.715e+03, 8.684e+03, 4.271e+03, 5.670e+03, 6.727e+03, 4.678e+03,
       8.685e+03, 2.821e+03, 2.029e+03, 5.772e+03, 9.576e+03, 7.338e+03,
       6.447e+03, 8.840e+03, 6.170e+03, 4.060e+03, 9.937e+03, 3.730e+03,
       2.229e+03, 3.068e+03, 7.077e+03, 5.089e+03, 8.311e+03, 8.292e+03,
       6.922e+03, 8.657e+03, 2.208e+03, 3.114e+03, 6.669e+03, 9.234e+03,
       1.674e+03, 9.159e+03, 8.202e+03, 6.658e+03, 8.366e+03, 6.123e+03,
       5.681e+03, 7.563e+03, 4.261e+03, 3.440e+03, 9.311e+03, 1.436e+03,
       3.421e+03, 4.340e+03, 8.837e+03, 8.932e+03, 9.011e+03, 7.464e+03,
       8.849e+03, 4.972e+03, 3.252e+03, 5.676e+03, 3.880e+03, 6.489e+03,
       6.417e+03, 3.875e+03, 5.285e+03, 2.923e+03, 7.813e+03, 2.279e+03,
       2.904e+03, 8.484e+03, 5.703e+03, 1.372e+03, 5.516e+03, 8.314e+03,
       3.295e+03, 7.775e+03, 1.768e+03, 4.948e+03, 7.363e+03, 1.844e+03,
       8.854e+03, 4.974e+03, 6.053e+03, 7.073e+03, 5.836e+03, 5.216e+03,
       9.882e+03, 2.113e+03, 8.993e+03, 1.280e+03, 6.807e+03, 5.914e+03,
       3.807e+03, 8.790e+03, 7.288e+03, 7.499e+03, 4.043e+03, 3.527e+03,
       2.674e+03, 2.468e+03, 2.387e+03, 3.390e+03, 5.040e+03, 7.224e+03,
       8.848e+03, 6.339e+03, 1.440e+03, 5.139e+03, 9.907e+03, 1.725e+03,
       8.303e+03, 5.840e+03, 6.706e+03, 1.370e+03, 8.570e+03, 8.037e+03,
       6.993e+03, 7.367e+03, 2.388e+03, 7.852e+03, 3.733e+03, 8.939e+03,
       8.272e+03, 3.755e+03, 9.880e+03, 5.305e+03, 3.719e+03, 6.302e+03,
       1.971e+03, 9.019e+03, 7.600e+03, 8.511e+03, 6.533e+03, 3.035e+03,
       5.489e+03, 4.836e+03, 2.690e+03, 3.688e+03, 3.293e+03, 5.065e+03,
       1.894e+03, 6.185e+03, 1.273e+03, 7.939e+03, 3.545e+03, 5.005e+03,
       2.956e+03, 3.735e+03, 2.262e+03, 2.685e+03, 3.458e+03, 5.444e+03,
       7.568e+03, 5.777e+03, 6.488e+03, 5.584e+03, 2.565e+03, 5.778e+03,
       8.404e+03, 4.522e+03, 3.848e+03, 2.989e+03, 5.240e+03, 1.699e+03,
       3.831e+03, 5.835e+03, 9.614e+03, 1.959e+03, 9.743e+03, 1.462e+03,
       1.621e+03, 3.137e+03, 9.622e+03, 8.157e+03, 6.026e+03, 4.619e+03,
       6.500e+03, 1.020e+03, 2.661e+03, 2.715e+03, 8.909e+03, 7.705e+03,
       1.893e+03, 4.390e+03, 2.493e+03, 5.315e+03, 4.727e+03, 5.896e+03,
       2.970e+03, 4.308e+03, 5.525e+03, 2.730e+03, 1.613e+03, 1.807e+03,
       4.734e+03, 9.303e+03, 5.087e+03, 6.640e+03, 4.626e+03, 1.138e+03,
       4.575e+03, 5.631e+03, 4.086e+03, 9.148e+03, 6.622e+03, 5.813e+03,
       4.659e+03, 1.879e+03, 3.470e+03, 2.324e+03, 3.595e+03, 1.076e+03,
       1.749e+03, 6.615e+03, 1.359e+03, 1.785e+03, 3.508e+03, 7.862e+03,
       6.416e+03, 2.659e+03, 5.760e+03, 6.384e+03, 8.479e+03, 2.699e+03,
       9.598e+03, 6.664e+03, 3.709e+03, 7.607e+03, 2.944e+03, 9.328e+03,
       6.952e+03, 2.741e+03, 7.437e+03, 8.912e+03, 7.859e+03, 9.975e+03,
       3.913e+03, 9.747e+03, 4.275e+03, 6.876e+03, 1.769e+03, 4.695e+03,
       4.063e+03, 5.178e+03, 9.037e+03, 7.814e+03, 9.691e+03, 1.653e+03,
       9.426e+03, 9.814e+03, 2.181e+03, 2.154e+03, 2.938e+03, 5.748e+03,
       3.561e+03, 1.182e+03, 4.441e+03, 2.877e+03, 5.107e+03, 5.671e+03,
       6.971e+03, 5.533e+03, 3.879e+03, 7.637e+03, 7.430e+03, 4.913e+03,
       9.485e+03, 5.314e+03, 9.460e+03, 3.713e+03, 4.773e+03, 1.367e+03,
       1.916e+03, 2.404e+03, 3.979e+03, 2.778e+03, 9.706e+03, 9.553e+03,
       2.600e+03, 5.223e+03, 6.175e+03, 6.990e+03, 5.157e+03, 2.642e+03,
       2.492e+03, 7.296e+03, 4.304e+03, 6.893e+03, 9.482e+03, 5.530e+03,
       2.185e+03, 8.923e+03, 4.610e+03, 5.706e+03, 3.994e+03, 6.167e+03,
       1.887e+03, 8.490e+03, 6.924e+03, 4.013e+03, 4.744e+03, 8.749e+03,
       5.915e+03, 2.577e+03, 1.324e+03, 5.663e+03, 3.745e+03, 6.191e+03,
       7.782e+03, 4.450e+03, 9.584e+03, 8.260e+03, 8.654e+03, 3.551e+03,
       8.816e+03, 8.161e+03, 9.464e+03, 8.460e+03, 4.046e+03, 9.086e+03,
       2.314e+03, 1.206e+03, 9.538e+03, 5.150e+03, 6.316e+03, 3.539e+03,
       1.979e+03, 8.187e+03, 9.892e+03, 2.128e+03, 4.039e+03, 4.110e+03,
       9.572e+03, 7.366e+03, 1.214e+03, 1.955e+03, 4.784e+03, 2.728e+03,
       4.943e+03, 5.660e+03, 5.521e+03, 4.817e+03, 8.905e+03, 7.266e+03,
       3.509e+03, 7.529e+03, 9.589e+03, 8.245e+03, 2.841e+03, 8.373e+03,
       9.991e+03, 5.653e+03, 6.901e+03, 8.940e+03, 6.465e+03, 9.212e+03,
       3.248e+03, 6.395e+03, 6.062e+03, 4.016e+03, 3.489e+03, 4.014e+03,
       5.421e+03, 8.496e+03, 9.530e+03, 7.347e+03, 7.440e+03, 2.477e+03,
       9.383e+03, 3.656e+03, 3.155e+03, 9.084e+03, 5.009e+03, 3.381e+03,
       5.659e+03, 8.001e+03, 4.425e+03, 7.996e+03, 9.069e+03, 4.396e+03,
       5.168e+03, 7.488e+03, 8.888e+03, 9.779e+03, 7.147e+03, 6.152e+03,
       9.192e+03, 4.361e+03, 6.460e+03, 6.257e+03, 3.580e+03, 5.440e+03,
       1.831e+03, 8.304e+03, 2.593e+03, 6.954e+03, 8.842e+03, 8.903e+03,
       3.835e+03, 3.886e+03, 9.098e+03, 8.421e+03, 5.989e+03, 2.022e+03,
       8.073e+03, 3.278e+03, 3.200e+03, 7.303e+03, 9.220e+03, 2.854e+03,
       5.017e+03, 8.681e+03, 3.744e+03, 5.780e+03, 8.317e+03, 6.653e+03,
       7.222e+03, 6.172e+03, 5.183e+03, 9.972e+03, 4.594e+03, 2.995e+03,
       4.018e+03, 3.915e+03, 9.857e+03, 5.105e+03, 7.411e+03, 8.898e+03,
       6.985e+03, 4.163e+03, 1.268e+03, 9.791e+03, 1.750e+03, 2.742e+03,
       6.312e+03, 3.185e+03, 5.648e+03, 3.859e+03, 4.144e+03, 5.528e+03,
       6.966e+03, 9.804e+03, 6.340e+03, 4.276e+03, 8.454e+03, 6.503e+03,
       3.804e+03, 2.416e+03, 5.041e+03, 1.724e+03, 8.412e+03, 5.846e+03,
       8.452e+03, 7.305e+03, 2.326e+03, 6.989e+03, 9.841e+03, 5.540e+03,
       9.714e+03, 5.724e+03, 5.803e+03, 4.097e+03, 8.965e+03, 7.093e+03,
       3.986e+03, 7.056e+03, 9.948e+03, 6.549e+03, 8.306e+03, 4.206e+03,
       6.915e+03, 8.300e+03, 2.449e+03, 4.264e+03, 8.593e+03, 7.834e+03,
       8.872e+03, 1.958e+03, 1.798e+03, 9.908e+03, 3.243e+03, 3.277e+03,
       3.108e+03, 2.273e+03, 4.365e+03, 3.630e+03, 5.617e+03, 8.203e+03,
       3.025e+03, 6.375e+03, 7.672e+03, 2.723e+03, 3.764e+03, 5.563e+03,
       2.770e+03, 2.117e+03, 2.253e+03, 1.517e+03, 5.965e+03, 6.779e+03,
       6.591e+03, 4.844e+03, 1.612e+03, 6.504e+03, 2.234e+03, 2.552e+03,
       5.945e+03, 9.971e+03, 8.133e+03, 7.649e+03, 7.914e+03, 1.691e+03,
       2.479e+03, 9.133e+03, 7.928e+03, 9.034e+03, 6.947e+03, 5.144e+03,
       5.929e+03, 1.555e+03, 5.275e+03, 4.010e+03, 3.585e+03, 5.270e+03,
       4.677e+03, 1.231e+03, 5.717e+03, 8.506e+03, 7.935e+03, 3.938e+03,
       3.323e+03, 5.312e+03, 1.871e+03, 8.150e+03, 3.425e+03, 1.067e+03,
       2.998e+03, 8.439e+03, 4.463e+03, 3.916e+03, 2.975e+03, 9.021e+03,
       4.053e+03, 9.115e+03, 5.073e+03, 7.766e+03, 2.866e+03, 3.213e+03,
       8.139e+03, 8.295e+03, 6.469e+03, 1.606e+03, 1.994e+03, 2.915e+03,
       6.633e+03, 3.148e+03, 8.911e+03, 4.845e+03, 2.043e+03, 6.820e+03,
       2.817e+03, 3.257e+03, 2.396e+03, 8.025e+03, 2.389e+03, 1.947e+03,
       9.647e+03, 7.627e+03, 1.383e+03, 7.110e+03, 8.424e+03, 7.187e+03,
       7.751e+03, 9.502e+03, 3.364e+03, 1.534e+03, 1.659e+03, 3.328e+03,
       2.106e+03, 8.786e+03, 2.463e+03, 2.450e+03, 9.877e+03, 1.310e+03,
       9.052e+03, 2.878e+03, 3.334e+03, 9.076e+03, 7.676e+03, 5.990e+03,
       9.806e+03, 1.985e+03, 8.077e+03, 5.473e+03, 2.270e+03, 3.044e+03,
       4.137e+03, 1.116e+03, 2.070e+03, 1.378e+03, 7.666e+03, 4.696e+03,
       1.456e+03, 1.891e+03, 3.045e+03, 1.897e+03, 7.581e+03, 8.014e+03,
       6.098e+03, 1.939e+03, 2.120e+03, 5.909e+03, 4.772e+03, 7.558e+03,
       8.215e+03, 3.855e+03, 6.209e+03, 5.049e+03, 2.062e+03, 9.609e+03,
       4.325e+03, 7.981e+03, 8.236e+03, 5.096e+03, 4.728e+03, 6.581e+03,
       7.608e+03, 7.247e+03, 5.991e+03, 2.138e+03, 2.126e+03, 9.615e+03,
       6.681e+03, 8.501e+03, 9.810e+03, 9.264e+03, 8.567e+03, 6.359e+03,
       8.713e+03, 3.759e+03, 8.242e+03, 4.305e+03, 4.148e+03, 4.105e+03,
       1.245e+03, 1.719e+03, 5.127e+03, 8.899e+03, 9.248e+03, 2.300e+03,
       4.900e+03, 1.388e+03, 4.644e+03, 4.153e+03, 4.871e+03, 5.966e+03,
       7.188e+03, 7.233e+03, 8.688e+03, 7.113e+03, 3.660e+03, 7.313e+03,
       5.281e+03, 7.297e+03, 9.202e+03, 5.407e+03, 9.600e+03, 3.981e+03,
       5.249e+03, 2.052e+03, 1.685e+03, 8.961e+03, 7.528e+03, 3.275e+03,
       9.436e+03, 8.279e+03, 2.465e+03, 3.016e+03, 6.623e+03, 2.460e+03,
       6.092e+03, 9.515e+03, 9.189e+03, 5.480e+03, 4.687e+03, 6.528e+03,
       9.978e+03, 8.239e+03, 2.670e+03, 2.922e+03, 5.959e+03, 2.353e+03,
       2.549e+03, 7.640e+03, 4.399e+03, 3.110e+03, 6.956e+03, 3.291e+03,
       4.884e+03, 5.544e+03, 8.472e+03, 7.097e+03, 5.425e+03, 5.476e+03,
       3.090e+03, 7.064e+03, 5.519e+03, 2.256e+03, 7.946e+03, 1.119e+03,
       9.577e+03, 7.211e+03, 5.337e+03, 8.999e+03, 4.050e+03, 7.729e+03,
       8.017e+03, 1.095e+03, 6.322e+03, 4.874e+03, 8.027e+03, 8.031e+03,
       4.758e+03, 2.489e+03, 2.005e+03, 5.046e+03, 2.650e+03, 3.134e+03,
       7.292e+03, 4.160e+03, 9.741e+03, 6.996e+03, 9.723e+03, 4.015e+03,
       9.254e+03, 5.781e+03, 1.022e+03, 5.831e+03, 7.258e+03, 3.550e+03,
       3.885e+03, 4.077e+03, 1.193e+03, 5.385e+03, 3.281e+03, 9.906e+03,
       1.019e+03, 6.906e+03, 3.461e+03, 1.035e+03, 5.562e+03, 7.301e+03,
       1.046e+03, 3.283e+03, 2.920e+03, 3.591e+03, 5.698e+03, 8.443e+03,
       8.674e+03, 2.776e+03, 5.410e+03, 6.428e+03, 5.753e+03, 3.638e+03,
       7.350e+03, 1.244e+03, 2.116e+03, 7.701e+03, 3.592e+03, 2.787e+03,
       4.497e+03, 1.886e+03, 7.636e+03, 8.021e+03])
2. Name

#melihat isi dalam kolom
data['Name'].unique()
     
array(['John', 'Sarah', 'Alex', 'Michael', 'Emma', 'Olivia', 'Daniel',
       'Sophia', 'James', 'Isabella', 'Shannon Harrison', 'Peter Davila',
       'Katherine Gray', 'Megan Miller', 'Derrick Alexander',
       'James Smith', 'James Reynolds', 'Brittany Sutton',
       'Annette Medina', 'James Duncan', 'Donna Taylor',
       'David Robertson', 'Susan Brown', 'Jessica Ortega', 'Jeremy Hall',
       'Peter Jones', 'Michael Sampson', 'Daniel Erickson',
       'David Aguilar', 'Benjamin Vance', 'Yolanda Chen',
       'Kristina Douglas', 'Scott Davidson', 'Alexandra Harrington',
       'Richard Johnson', 'Jerry Browning', 'Carrie Page',
       'Michelle Benjamin', 'James Woods', 'Angela Dunn', 'Nathan Riley',
       'Shelby Mahoney', 'Andrea Frey', 'Justin Martin', 'Mark Hartman',
       'Antonio Hansen', 'Lynn Anderson', 'Michael Cooley',
       'Mitchell Sanders', 'Maria Cross', 'Lisa Castillo',
       'Theresa Rodriguez', 'Kimberly Wood', 'Patricia Cox',
       'Kimberly Blair', 'Jason Ramirez', 'Gail Dudley', 'John West',
       'Patricia Carr', 'David Nunez', 'Collin Peterson', 'Rebecca Smith',
       'Lisa Benjamin', 'Desiree Reyes', 'Megan Diaz', 'Joseph Adams',
       'Sharon Watson', 'Tiffany Smith', 'Alexander Preston',
       'Randall Hickman', 'David Hernandez', 'Michelle Jones',
       'Tonya Gonzalez', 'Sarah Johnson', 'Grace Valencia', 'Sarah Young',
       'Gerald Weber', 'Gregory Turner', 'Laura Cortez', 'Lori Miller',
       'Lori Gomez PhD', 'Dawn Foster', 'Peter Welch', 'Maria Carter',
       'Daniel Hood', 'Amanda Hernandez', 'Alyssa Schmidt',
       'Taylor Merritt DVM', 'Alyssa Smith', 'Nathan Navarro',
       'Lisa Zimmerman', 'Jeffrey Howard', 'Morgan Bell',
       'Angelica Steele', 'Gregory Daniels', 'Eric Davis', 'Ruth Davis',
       'Emily Torres', 'Nancy Davis', 'Amanda Bowers', 'Alison Lang',
       'Matthew Massey', 'James Anderson', 'Robert Howard',
       'Rebecca Brown', 'Kimberly Harrison', 'Derek Parks',
       'Jennifer Baker', 'Diana Schultz', 'Mrs. Jill Long',
       'Heather Johnson', 'Robert Lowery', nan, 'Angela Williams',
       'Brent Burns', 'Melissa Barber', 'John Warner',
       'Christopher Baxter', 'Courtney Haas', 'Sarah Ross',
       'Manuel Baker', 'Jessica Lamb', 'Jaime Nguyen',
       'Christina Thompson', 'Shari Harrell', 'Tracy Gibbs',
       'David Hoffman', 'Robert Davis', 'Patricia Greer', 'James Stewart',
       'Brent Mendoza', 'Dawn Taylor', 'Larry Brown',
       'Miss Teresa Hernandez MD', 'Edward Baker', 'Diana Gould',
       'Amy Rodriguez', 'Paul Hull', 'Allison Hamilton', 'John King',
       'Tristan Avila', 'Connie Cooper', 'Judith Santos',
       'Michaela Powell', 'Lawrence Roberts', 'Nicholas Robinson DDS',
       'Jill Smith', 'Michael Burns', 'Adam Weaver', 'Melvin Mcgee',
       'William Banks', 'Dawn Morton', 'Frank Dawson', 'Gregory Davila',
       'Rebekah Finley', 'Robert Martinez', 'Steven Crawford',
       'Leslie King', 'George Cannon', 'Adam Carlson',
       'Stephanie Lawrence', 'Adam Blackwell', 'Andrew Owens',
       'John Sexton', 'Leon Nguyen', 'John White', 'Justin Martinez',
       'Nathaniel Kemp', 'Jacob Murray', 'Anthony Green',
       'Trevor Freeman', 'Michael Smith', 'William Hogan',
       'Joshua Juarez', 'Keith Wu', 'Ryan Williams', 'Ronald Stewart',
       'Ronald Schmidt', 'Courtney Bradley', 'Deanna Gonzalez',
       'Darrell Adams', 'Michael Mata', 'Anthony Dunn', 'Jill Williams',
       'Julia Russell', 'Tonya Thompson', 'Brandon Williamson',
       'Joseph Davis', 'Elizabeth Andersen', 'Jeffrey Meyers',
       'Christopher Jones', 'Aimee Rogers', 'Alexis Bates',
       'Michael Evans', 'Omar Gonzalez', 'Melissa Perry', 'Dawn Torres',
       'Timothy Dean', 'Anna Rollins MD', 'Timothy Barrera',
       'Susan Morris', 'Gordon Merritt', 'Kristy Rhodes',
       'Ms. Marissa Ramirez', 'Paul Hickman', 'Mrs. Kathy Woodward',
       'Alexis Horn', 'Shannon Harris DDS', 'Alicia Ortiz',
       'Alexis Walton', 'Gina Palmer', 'Jean Moreno', 'David Anderson',
       'Brian Huffman', 'Jose Liu', 'Cathy Pratt', 'Kathy Sandoval',
       'Michael Webb', 'Ronnie Taylor', 'Matthew Miller', 'Shelly Hurley',
       'Mr. Martin Dominguez MD', 'Jennifer Wiggins', 'Anthony Smith',
       'John Lopez', 'Nathan Rodgers', 'Martin Walker', 'Jesus Wood',
       'Kelly Bell', 'Kathryn Randall', 'Sarah Norris', 'Matthew Adams',
       'William Wade', 'Brian Castro', 'Teresa Sanders', 'Richard Oneal',
       'Christine Cooper', 'Michael Foster', 'Loretta Gonzalez',
       'Douglas Ingram', 'Randy Anderson', 'Donna Shaw', 'Joseph Holt',
       'Taylor Acosta', 'Danielle Scott', 'Melissa Aguirre',
       'Cassandra Snyder', 'Kimberly Lambert', 'Rebecca Mendoza DDS',
       'Michael Anderson', 'Hannah Gentry', 'Patrick Reed',
       'Tiffany Huff', 'Thomas Lee', 'Justin Clark', 'Michael Martinez',
       'Vincent Smith', 'Sean Hull', 'Sarah Welch', 'Mr. Stephen Gilmore',
       'Barry Parker', 'Joshua Kane', 'Anthony Wright', 'Nathaniel Lopez',
       'Darlene Fletcher', 'Christopher Moore', 'Benjamin White',
       'Jenny Coleman', 'Michael Gomez', 'Caleb Serrano',
       'William Carlson', 'Timothy Pruitt', 'Mario Johnston',
       'Rebecca Kim', 'Joyce Williams', 'Cynthia Sloan',
       'Patricia Williams', 'Michael Payne', 'Brooke Friedman',
       'Daniel Mathis', 'Lynn Jennings', 'Kimberly Randolph',
       'Jack Simmons', 'Daniel Mendez', 'Travis Lawson',
       'Jason Horne DDS', 'Shawn Galloway', 'Thomas Barnes',
       'Elizabeth Curry', 'Nicole Cunningham', 'Patrick Thompson',
       'Marissa Stewart', 'Melissa Wilson', 'Mark Castillo',
       'Mr. Kent Stanton', 'Tyrone Phillips', 'Christopher Collins',
       'Christopher Mccullough', 'Carla Rodriguez', 'Makayla Howard',
       'Barbara White', 'Sarah Davis', 'Cody Donovan', 'Jessica Jenkins',
       'Benjamin Alexander', 'Isaac Lopez', 'Ariana Morse', 'Brian Avery',
       'Amy Lawson', 'Derek Choi', 'Paul Reese', 'Francisco Chang',
       'Christian Knight', 'Andrew Wilkerson', 'Brandon Alvarez',
       'Gary Young', 'Jessica Wright', 'Timothy Clements',
       'Alexander Gordon DDS', 'James Martinez', 'Francisco Williams',
       'Jordan Brooks', 'Kurt Boyd', 'Bryan Salazar', 'Mrs. Heather Roy',
       'Kathleen Newton', 'Julian Fox', 'Kylie Hancock', 'Matthew Werner',
       'Kelsey Bush', 'Brandon Dickerson', 'Jamie Allen', 'Taylor Smith',
       'Elizabeth Chavez', 'Melissa Morris', 'Donna Hardin',
       'Andrew Cook', 'Samuel Shaw', 'Mike Waters', 'Andrea Newman',
       'Barry Jenkins', 'Laurie Johnson', 'John Coleman', 'John Carter',
       'Mr. Devon Garrett', 'Jill Kelly', 'Richard Gonzalez',
       'Alex White', 'Michael Gonzales', 'John Williams',
       'Daniel Herrera', 'Ian Glass', 'Heidi Simmons', 'Andrea Fuller',
       'David Dunlap', 'Antonio Mitchell', 'Mariah Cole',
       'Raymond Murillo', 'Andrea Bradley', 'Tina Velazquez',
       'Bonnie Callahan', 'Eric Scott', 'Michael Valenzuela',
       'Robin Spears', 'David Owens', 'Elizabeth Clark', 'Beverly Fisher',
       'Sarah Bailey', 'Kendra Ramirez', 'Diane Ross',
       'Stephanie Mcfarland', 'Michelle Woods', 'David Ruiz',
       'Jason Castro', 'Kathleen Montgomery', 'James Bennett',
       'Kelsey Watkins', 'David Park', 'Christopher Poole',
       'Rebecca Baxter', 'Virginia Daniel', 'Melissa Gordon',
       'Brett Barrett', 'Brandon Townsend', 'Haley Robertson',
       'Michelle Gonzalez', 'Gary Holland', 'Felicia Howard',
       'Sean Walker', 'Brianna Mills', 'Michael Case', 'Felicia Houston',
       'Kevin Wilson', 'Mary Vaughn', 'Maurice Martinez',
       'Jonathon Perry', 'Rebecca Evans', 'Paul Williams', 'Tyler Boyer',
       'Kristy Nichols', 'Darren Martin', 'Katelyn Morgan',
       'Jesus Henson', 'Robin Stewart', 'Lawrence Miller', 'Ronnie White',
       'John Fox', 'Daniel Scott', 'Jonathan Mann', 'Bethany Andersen',
       'Tiffany Moore', 'Sylvia Chavez', 'Gerald Reilly', 'Ashlee Wilcox',
       'Lance Johnson', 'Patrick Gonzalez MD', 'Cole Gomez',
       'Alejandro Gutierrez', 'Natalie Long', 'Abigail Ingram',
       'Eileen Lee', 'Nancy Moore', 'Melanie Wilson', 'Gloria Henderson',
       'Christopher Payne', 'John Franklin', 'Angel Stout',
       'Candace Kelly', 'Zachary Daniel', 'Angel Howard', 'Heather Hill',
       'Heather Gamble', 'Sean Levine', 'Brian Diaz', 'Catherine Koch',
       'Anne Martin', 'Mary Warren', 'Amanda Bryant', 'Joel Ford',
       'Mark Miller', 'Sarah Russell', 'Lisa Mitchell', 'William White',
       'Jacqueline Garcia', 'Elizabeth Wright', 'Pamela Woods',
       'Sydney Patterson', 'Lisa Johnson', 'Todd Jackson',
       'Shelby Richards', 'Joseph Roberts', 'Samantha Burton',
       'Cynthia Mcdonald', 'Kimberly Petty', 'Joshua Carpenter',
       'Jessica Fuller', 'Andrea Barber', 'Brian Lyons', 'Tammy Fuentes',
       'Robert Taylor', 'David Gonzalez', 'Steven Myers',
       'Cassandra Perez', 'Jacqueline Eaton', 'Carrie Mccoy',
       'Amy Fisher', 'Stephanie Reid', 'Reginald Hoover',
       'Christina Russo', 'Raymond Howell', 'Angela Perez',
       'Charles Aguilar', 'Caitlyn Green', 'Ronald Davis',
       'Melissa Lucas', 'Heidi Leonard', 'Richard Swanson',
       'Lauren Simpson', 'Marissa Raymond', 'Joyce Bishop',
       'Michael Kelly', 'Peggy Hill', 'Derrick Love', 'Laura Stevens',
       'Terri Potter', 'Veronica Saunders', 'Emily Townsend',
       'James Long', 'Jennifer Dickson', 'Alexa West', 'Phillip Donovan',
       'Monique King', 'Karl Murray', 'Julia Brooks', 'Lauren Montgomery',
       'Nathan Wall', 'Kristin Macias', 'Samuel Smith', 'Bradley Stevens',
       'Morgan Hampton', 'Lori Carrillo', 'Samantha Novak',
       'Audrey Robinson', 'Lindsey Kerr', 'Cynthia Nguyen', 'Ryan Hood',
       'Matthew Stevenson', 'Rhonda White', 'Larry Ramsey',
       'Kimberly Anderson', 'Brian Miles', 'Kelly Baker', 'Wendy Miller',
       'Michael Webster', 'Michelle Ryan', 'Justin Jennings',
       'Norma Ward', 'Vincent Johnson', 'David Stone', 'Amy Evans',
       'Stephen Ramirez', 'Nicholas Baxter', 'Raymond Smith',
       'Roy Johnson', 'Ryan Vasquez', 'Justin Johnson', 'Kaylee Benjamin',
       'Charles Murphy', 'Grant Henderson', 'Michael Woods',
       'Michael Kane', 'Susan Benson', 'Timothy Armstrong',
       'Cindy Harvey', 'Caitlin Carter', 'Ricky Johnson', 'Bryan Hoover',
       'Elijah Wagner', 'Felicia Davis', 'Jonathan Hess', 'Michael Yoder',
       'Samantha York', 'Joanna Martinez', 'Penny Hopkins',
       'Justin Foster', 'Jennifer Morgan', 'Dalton Chen',
       'Mrs. Amy Carney MD', 'Melissa Wang', 'Cameron Bender',
       'Daniel Solomon', 'Deanna Jones', 'Alicia Young', 'Peter Figueroa',
       'Jacob Thompson', 'Jonathon Woodward', 'Brian Hayes',
       'Lisa Carroll', 'Cassandra Ryan', 'Terri Hayden', 'Taylor Burns',
       'Andrew Jimenez', 'Wendy King', 'Luis Carter', 'Nicholas Jones',
       'Tiffany Sanders', 'Tina Knight', 'Brittany Andrews',
       'William Weber', 'Angela Ortiz', 'David Meyer', 'Travis Harrell',
       'Craig Mccann', 'Scott Edwards', 'Kelly Cohen', 'David Diaz',
       'Joseph Green', 'Henry Johnson', 'Keith Wilson', 'Jordan Chase',
       'Kayla Solomon', 'Hayley Johnson', 'John Bender',
       'William Christian', 'Michael Willis', 'James Simmons',
       'Christopher Sullivan', 'Michelle Santos', 'Michael Walters',
       'Jason Anderson', 'James Hoover', 'Julie Garza', 'Lisa Williams',
       'Erica Thomas', 'Calvin Vargas', 'Molly Jones', 'Shannon Hogan',
       'James Fields', 'Arthur Johnson', 'Collin Edwards',
       'Kimberly Smith', 'Mr. Clinton Camacho', 'Taylor Moyer',
       'David Jimenez', 'Kyle Coleman', 'Marvin Cardenas',
       'Eddie Robertson', 'Christopher Taylor', 'Brenda Bell',
       'Ashley Nguyen', 'Michael Bailey', 'Maria Scott',
       'Katherine Mclaughlin', 'Karen Cruz', 'Jodi Taylor',
       'Thomas Mcgrath', 'Kathy Bailey', 'Karen Wolf', 'Mary Jones',
       'Lisa Houston', 'Ashley Jordan', 'Lonnie Williams',
       'Michael Poole', 'David Martinez', 'Eduardo Miller',
       'Kristine Williams', 'Audrey Woods', 'Lauren Brown',
       'Stephanie Merritt', 'Rebecca Jones', 'Isaiah Arnold',
       'Andrew Dudley', 'Kenneth Jones', 'Jane Griffith', 'Jeff Phillips',
       'Dawn Edwards', 'Paige Campbell', 'Belinda Kim',
       'Christopher Cruz', 'Ruben Steele', 'Bradley Martinez',
       'Alison Williams', 'Scott Ferguson', 'David Brown', 'Renee Scott',
       'Aimee Green', 'Ricky Jenkins', 'Kelly Allen MD', 'Sean Smith',
       'Robert Johnson', 'Susan Russell', 'Ryan West',
       'Christina Fitzgerald', 'Susan Roberson', 'Jessica Banks',
       'Jodi Rogers', 'Michelle Clay', 'Jacob Bean', 'Robert Harper',
       'Michelle Graham', 'Seth Johnson', 'Julia Sanchez', 'Justin Sharp',
       'Thomas Yoder', 'Christine Mullins', 'Timothy Mcbride',
       'Deborah Reyes', 'Chad Downs', 'William Hill', 'Makayla Rhodes',
       'Clayton Jones', 'Sean Hernandez', 'Christine Young', 'Brian Hale',
       'Sandra Hart', 'Christine Curtis', 'Tara Johnson',
       'Stephanie Snyder', 'Miguel Leblanc', 'Christina Mcintyre',
       'John Weber', 'Timothy Austin', 'Jeffrey Sullivan',
       'Michael Noble', 'Christine Shah', 'Lisa Maynard', 'Lisa Lynch',
       'Angela Warren', 'Mary Benson', 'Yvonne Krueger', 'Keith Bailey',
       'Barbara Case', 'Jamie Torres', 'Justin Jacobs', 'Kevin Reynolds',
       'Michelle Kelly', 'Terry Hernandez', 'Aaron Duarte',
       'Micheal Perez', 'Troy Bradford', 'Bobby Wilkinson', 'Juan Lopez',
       'Virginia Medina', 'Leah Holmes', 'Todd Brooks', 'Madeline Rush',
       'Nicole Frederick', 'Jason Adams', 'Mallory Fletcher',
       'David Cummings', 'Kelly Suarez', 'Drew Morse', 'Lauren Stephens',
       'Penny Miller', 'Paula Stewart', 'Jonathan Johnson', 'Laura Ross',
       'Brandy Wilson', 'Brittany Briggs', 'Carl Coleman',
       'Crystal Chandler', 'Elizabeth Nichols', 'Jesse Barber',
       'Julie Jackson', 'Cindy Morales', 'Ashley Franco',
       'Kimberly Calderon', 'Lisa Brooks', 'Destiny Todd', 'Dawn Luna',
       'Jacob Yoder', 'Martha Moody', 'James Torres', 'Shelly Stuart',
       'Julie Williams', 'Tamara Jones', 'Todd Orozco', 'Jeffrey Torres',
       'Chad Colon', 'Alan Williams', 'Cameron Austin', 'Dawn Nguyen',
       'Andrea Phillips', 'Robert Bishop', 'Melanie Mckinney',
       'Joseph Ellis', 'Jennifer Smith', 'Vickie Holloway', 'Nancy Munoz',
       'David Taylor', 'Amber Gray', 'Christian Barr', 'Mark Fleming',
       'Felicia Carpenter', 'Rebecca Kelly', 'Aaron Walls', 'Mark King',
       'James Macdonald', 'Alexander Thompson', 'Kaitlin Walton',
       'Ashley Kaufman', 'Megan Bernard', 'Kelly Harper', 'Daniel White',
       'Justin Castro', 'John Odonnell', 'Whitney Tapia', 'Jeff Lawrence',
       'Jeremy Shields', 'Matthew Davies', 'Christine Hamilton',
       'Crystal Proctor', 'Victoria Mcbride', 'Ellen Cross MD',
       'David Stevens', 'Frank Brown DDS', 'Deborah Barajas',
       'Richard Howard Jr.', 'Anita Dougherty', 'Alicia Watson',
       'Deborah Hunter', 'Morgan Gilbert', 'Taylor Crawford',
       'Danny Sanchez', 'Cory Rosario DDS', 'Kyle Perez',
       'Kelly Gibson MD', 'Angela Garcia', 'Dylan Vaughan',
       'Kevin Santiago', 'Terry Rodriguez', 'Chad Mason',
       'Jessica Ferguson', 'Andrew Schroeder', 'Jeffrey Garcia',
       'Jason Hood', 'Sandra Leon', 'Tyler Mccormick', 'Megan Murphy',
       'Charles Vincent', 'Mrs. Cindy Sharp DVM', 'Sarah Dixon',
       'Teresa Rodriguez', 'Shannon Spencer', 'Darlene Tapia',
       'Ryan Ball', 'Anita Berger', 'Tom Bradley', 'Melinda Cummings',
       'Melissa Donaldson', 'Charles Pearson', 'Isaiah Moore',
       'Jonathan Ramirez', 'Mrs. Melanie Mueller', 'Ms. Heather Dean',
       'David Malone', 'Melissa Brown', 'Jason Woods', 'Rachel Diaz',
       'Michael Jones', 'Taylor Mcgee', 'Christopher Ray',
       'Alyssa Berg MD', 'Dr. James Henry DDS', 'James Stevens',
       'Kevin Marquez', 'Colleen Porter', 'Christian Odom',
       'Calvin Macias', 'Kim Fischer', 'Travis Schultz', 'Bonnie Morris',
       'Tiffany Morris', 'Paul Poole', 'Richard Perez',
       'Bernard Whitaker', 'Melinda Miller', 'Christina Gonzales',
       'Maria Gray', 'Mary Bennett', 'Dr. Erik Pineda', 'Sandra Burns',
       'Patricia Hunt', 'Erika Kennedy', 'Timothy Newman', 'Tanya Bowen',
       'Rhonda Davenport', 'Cheryl Brown', 'Angela Bridges',
       'Brandon Rasmussen', 'Ryan Beltran', 'Allison Novak',
       'Kirsten Davis', 'Paige Turner', 'Aimee Christensen',
       'Jessica Howard', 'Lisa Zhang', 'Jonathan Daniel',
       'Cheyenne Oliver', 'Brianna Henderson', 'Mark Sanchez',
       'Melissa Nguyen', 'Jeff Carney', 'Tammy Tanner', 'Donna Robinson',
       'Eric Wilson', 'Linda Armstrong', 'Russell Jones',
       'Darlene Burton', 'Faith Johnson', 'Nathan Miller', 'Donna Fowler',
       'Whitney Jefferson', 'Shane Sanders', 'Felicia Cline',
       'Jennifer Clark', 'Miss Carla Coleman MD', 'William Nelson',
       'Crystal Brown', 'Erica Miller', 'Emily Martin', 'Wesley May',
       'Christopher Atkinson', 'Blake Martinez', 'Joy Henderson',
       'Mary Ramos', 'Colleen Sanchez', 'David Cruz', 'Ralph Becker',
       'Denise Coleman', 'Samantha Mendoza', 'Sean Lyons',
       'Deborah Hernandez', 'Kristi Price', 'Alex Douglas',
       'Ian Mcdonald', 'David Bauer', 'Steven Hill', 'Linda Beasley',
       'Maria Edwards', 'Megan Hutchinson', 'Jaime Simon',
       'Thomas Freeman', 'Holly Leon', 'Travis Pierce',
       'Kristen Galloway', 'Raven Taylor', 'Samuel Robinson',
       'Christina Johnson', 'Holly Spencer', 'Jessica Davies',
       'Daniel Sullivan', 'Richard Franklin', 'Samantha Klein',
       'Becky Perkins', 'Kimberly Pena', 'Rebecca Burnett',
       'Anna Martinez', 'Monica Johnson', 'Shannon Porter', 'Amy Stout',
       'Joseph Sherman', 'Maria Walls'], dtype=object)

#jumlah missing value
data['Name'].isnull().sum()
     
np.int64(34)
Kolom Name bersifat kategorik dan hanya sebagai identitas, sehingga tidak terlalu berpengaruh terhadap model. Penanganan yang dapat dilakukan:


#mengisi missing value
data['Name'] = data['Name'].fillna('Unknown')
     

#jumlah missng value
np.sum(data['Name'].isnull())
     
np.int64(0)

#melihat dalam kolom setelah mengisi missing value
data['Name'].unique()
     
array(['John', 'Sarah', 'Alex', 'Michael', 'Emma', 'Olivia', 'Daniel',
       'Sophia', 'James', 'Isabella', 'Shannon Harrison', 'Peter Davila',
       'Katherine Gray', 'Megan Miller', 'Derrick Alexander',
       'James Smith', 'James Reynolds', 'Brittany Sutton',
       'Annette Medina', 'James Duncan', 'Donna Taylor',
       'David Robertson', 'Susan Brown', 'Jessica Ortega', 'Jeremy Hall',
       'Peter Jones', 'Michael Sampson', 'Daniel Erickson',
       'David Aguilar', 'Benjamin Vance', 'Yolanda Chen',
       'Kristina Douglas', 'Scott Davidson', 'Alexandra Harrington',
       'Richard Johnson', 'Jerry Browning', 'Carrie Page',
       'Michelle Benjamin', 'James Woods', 'Angela Dunn', 'Nathan Riley',
       'Shelby Mahoney', 'Andrea Frey', 'Justin Martin', 'Mark Hartman',
       'Antonio Hansen', 'Lynn Anderson', 'Michael Cooley',
       'Mitchell Sanders', 'Maria Cross', 'Lisa Castillo',
       'Theresa Rodriguez', 'Kimberly Wood', 'Patricia Cox',
       'Kimberly Blair', 'Jason Ramirez', 'Gail Dudley', 'John West',
       'Patricia Carr', 'David Nunez', 'Collin Peterson', 'Rebecca Smith',
       'Lisa Benjamin', 'Desiree Reyes', 'Megan Diaz', 'Joseph Adams',
       'Sharon Watson', 'Tiffany Smith', 'Alexander Preston',
       'Randall Hickman', 'David Hernandez', 'Michelle Jones',
       'Tonya Gonzalez', 'Sarah Johnson', 'Grace Valencia', 'Sarah Young',
       'Gerald Weber', 'Gregory Turner', 'Laura Cortez', 'Lori Miller',
       'Lori Gomez PhD', 'Dawn Foster', 'Peter Welch', 'Maria Carter',
       'Daniel Hood', 'Amanda Hernandez', 'Alyssa Schmidt',
       'Taylor Merritt DVM', 'Alyssa Smith', 'Nathan Navarro',
       'Lisa Zimmerman', 'Jeffrey Howard', 'Morgan Bell',
       'Angelica Steele', 'Gregory Daniels', 'Eric Davis', 'Ruth Davis',
       'Emily Torres', 'Nancy Davis', 'Amanda Bowers', 'Alison Lang',
       'Matthew Massey', 'James Anderson', 'Robert Howard',
       'Rebecca Brown', 'Kimberly Harrison', 'Derek Parks',
       'Jennifer Baker', 'Diana Schultz', 'Mrs. Jill Long',
       'Heather Johnson', 'Robert Lowery', 'Unknown', 'Angela Williams',
       'Brent Burns', 'Melissa Barber', 'John Warner',
       'Christopher Baxter', 'Courtney Haas', 'Sarah Ross',
       'Manuel Baker', 'Jessica Lamb', 'Jaime Nguyen',
       'Christina Thompson', 'Shari Harrell', 'Tracy Gibbs',
       'David Hoffman', 'Robert Davis', 'Patricia Greer', 'James Stewart',
       'Brent Mendoza', 'Dawn Taylor', 'Larry Brown',
       'Miss Teresa Hernandez MD', 'Edward Baker', 'Diana Gould',
       'Amy Rodriguez', 'Paul Hull', 'Allison Hamilton', 'John King',
       'Tristan Avila', 'Connie Cooper', 'Judith Santos',
       'Michaela Powell', 'Lawrence Roberts', 'Nicholas Robinson DDS',
       'Jill Smith', 'Michael Burns', 'Adam Weaver', 'Melvin Mcgee',
       'William Banks', 'Dawn Morton', 'Frank Dawson', 'Gregory Davila',
       'Rebekah Finley', 'Robert Martinez', 'Steven Crawford',
       'Leslie King', 'George Cannon', 'Adam Carlson',
       'Stephanie Lawrence', 'Adam Blackwell', 'Andrew Owens',
       'John Sexton', 'Leon Nguyen', 'John White', 'Justin Martinez',
       'Nathaniel Kemp', 'Jacob Murray', 'Anthony Green',
       'Trevor Freeman', 'Michael Smith', 'William Hogan',
       'Joshua Juarez', 'Keith Wu', 'Ryan Williams', 'Ronald Stewart',
       'Ronald Schmidt', 'Courtney Bradley', 'Deanna Gonzalez',
       'Darrell Adams', 'Michael Mata', 'Anthony Dunn', 'Jill Williams',
       'Julia Russell', 'Tonya Thompson', 'Brandon Williamson',
       'Joseph Davis', 'Elizabeth Andersen', 'Jeffrey Meyers',
       'Christopher Jones', 'Aimee Rogers', 'Alexis Bates',
       'Michael Evans', 'Omar Gonzalez', 'Melissa Perry', 'Dawn Torres',
       'Timothy Dean', 'Anna Rollins MD', 'Timothy Barrera',
       'Susan Morris', 'Gordon Merritt', 'Kristy Rhodes',
       'Ms. Marissa Ramirez', 'Paul Hickman', 'Mrs. Kathy Woodward',
       'Alexis Horn', 'Shannon Harris DDS', 'Alicia Ortiz',
       'Alexis Walton', 'Gina Palmer', 'Jean Moreno', 'David Anderson',
       'Brian Huffman', 'Jose Liu', 'Cathy Pratt', 'Kathy Sandoval',
       'Michael Webb', 'Ronnie Taylor', 'Matthew Miller', 'Shelly Hurley',
       'Mr. Martin Dominguez MD', 'Jennifer Wiggins', 'Anthony Smith',
       'John Lopez', 'Nathan Rodgers', 'Martin Walker', 'Jesus Wood',
       'Kelly Bell', 'Kathryn Randall', 'Sarah Norris', 'Matthew Adams',
       'William Wade', 'Brian Castro', 'Teresa Sanders', 'Richard Oneal',
       'Christine Cooper', 'Michael Foster', 'Loretta Gonzalez',
       'Douglas Ingram', 'Randy Anderson', 'Donna Shaw', 'Joseph Holt',
       'Taylor Acosta', 'Danielle Scott', 'Melissa Aguirre',
       'Cassandra Snyder', 'Kimberly Lambert', 'Rebecca Mendoza DDS',
       'Michael Anderson', 'Hannah Gentry', 'Patrick Reed',
       'Tiffany Huff', 'Thomas Lee', 'Justin Clark', 'Michael Martinez',
       'Vincent Smith', 'Sean Hull', 'Sarah Welch', 'Mr. Stephen Gilmore',
       'Barry Parker', 'Joshua Kane', 'Anthony Wright', 'Nathaniel Lopez',
       'Darlene Fletcher', 'Christopher Moore', 'Benjamin White',
       'Jenny Coleman', 'Michael Gomez', 'Caleb Serrano',
       'William Carlson', 'Timothy Pruitt', 'Mario Johnston',
       'Rebecca Kim', 'Joyce Williams', 'Cynthia Sloan',
       'Patricia Williams', 'Michael Payne', 'Brooke Friedman',
       'Daniel Mathis', 'Lynn Jennings', 'Kimberly Randolph',
       'Jack Simmons', 'Daniel Mendez', 'Travis Lawson',
       'Jason Horne DDS', 'Shawn Galloway', 'Thomas Barnes',
       'Elizabeth Curry', 'Nicole Cunningham', 'Patrick Thompson',
       'Marissa Stewart', 'Melissa Wilson', 'Mark Castillo',
       'Mr. Kent Stanton', 'Tyrone Phillips', 'Christopher Collins',
       'Christopher Mccullough', 'Carla Rodriguez', 'Makayla Howard',
       'Barbara White', 'Sarah Davis', 'Cody Donovan', 'Jessica Jenkins',
       'Benjamin Alexander', 'Isaac Lopez', 'Ariana Morse', 'Brian Avery',
       'Amy Lawson', 'Derek Choi', 'Paul Reese', 'Francisco Chang',
       'Christian Knight', 'Andrew Wilkerson', 'Brandon Alvarez',
       'Gary Young', 'Jessica Wright', 'Timothy Clements',
       'Alexander Gordon DDS', 'James Martinez', 'Francisco Williams',
       'Jordan Brooks', 'Kurt Boyd', 'Bryan Salazar', 'Mrs. Heather Roy',
       'Kathleen Newton', 'Julian Fox', 'Kylie Hancock', 'Matthew Werner',
       'Kelsey Bush', 'Brandon Dickerson', 'Jamie Allen', 'Taylor Smith',
       'Elizabeth Chavez', 'Melissa Morris', 'Donna Hardin',
       'Andrew Cook', 'Samuel Shaw', 'Mike Waters', 'Andrea Newman',
       'Barry Jenkins', 'Laurie Johnson', 'John Coleman', 'John Carter',
       'Mr. Devon Garrett', 'Jill Kelly', 'Richard Gonzalez',
       'Alex White', 'Michael Gonzales', 'John Williams',
       'Daniel Herrera', 'Ian Glass', 'Heidi Simmons', 'Andrea Fuller',
       'David Dunlap', 'Antonio Mitchell', 'Mariah Cole',
       'Raymond Murillo', 'Andrea Bradley', 'Tina Velazquez',
       'Bonnie Callahan', 'Eric Scott', 'Michael Valenzuela',
       'Robin Spears', 'David Owens', 'Elizabeth Clark', 'Beverly Fisher',
       'Sarah Bailey', 'Kendra Ramirez', 'Diane Ross',
       'Stephanie Mcfarland', 'Michelle Woods', 'David Ruiz',
       'Jason Castro', 'Kathleen Montgomery', 'James Bennett',
       'Kelsey Watkins', 'David Park', 'Christopher Poole',
       'Rebecca Baxter', 'Virginia Daniel', 'Melissa Gordon',
       'Brett Barrett', 'Brandon Townsend', 'Haley Robertson',
       'Michelle Gonzalez', 'Gary Holland', 'Felicia Howard',
       'Sean Walker', 'Brianna Mills', 'Michael Case', 'Felicia Houston',
       'Kevin Wilson', 'Mary Vaughn', 'Maurice Martinez',
       'Jonathon Perry', 'Rebecca Evans', 'Paul Williams', 'Tyler Boyer',
       'Kristy Nichols', 'Darren Martin', 'Katelyn Morgan',
       'Jesus Henson', 'Robin Stewart', 'Lawrence Miller', 'Ronnie White',
       'John Fox', 'Daniel Scott', 'Jonathan Mann', 'Bethany Andersen',
       'Tiffany Moore', 'Sylvia Chavez', 'Gerald Reilly', 'Ashlee Wilcox',
       'Lance Johnson', 'Patrick Gonzalez MD', 'Cole Gomez',
       'Alejandro Gutierrez', 'Natalie Long', 'Abigail Ingram',
       'Eileen Lee', 'Nancy Moore', 'Melanie Wilson', 'Gloria Henderson',
       'Christopher Payne', 'John Franklin', 'Angel Stout',
       'Candace Kelly', 'Zachary Daniel', 'Angel Howard', 'Heather Hill',
       'Heather Gamble', 'Sean Levine', 'Brian Diaz', 'Catherine Koch',
       'Anne Martin', 'Mary Warren', 'Amanda Bryant', 'Joel Ford',
       'Mark Miller', 'Sarah Russell', 'Lisa Mitchell', 'William White',
       'Jacqueline Garcia', 'Elizabeth Wright', 'Pamela Woods',
       'Sydney Patterson', 'Lisa Johnson', 'Todd Jackson',
       'Shelby Richards', 'Joseph Roberts', 'Samantha Burton',
       'Cynthia Mcdonald', 'Kimberly Petty', 'Joshua Carpenter',
       'Jessica Fuller', 'Andrea Barber', 'Brian Lyons', 'Tammy Fuentes',
       'Robert Taylor', 'David Gonzalez', 'Steven Myers',
       'Cassandra Perez', 'Jacqueline Eaton', 'Carrie Mccoy',
       'Amy Fisher', 'Stephanie Reid', 'Reginald Hoover',
       'Christina Russo', 'Raymond Howell', 'Angela Perez',
       'Charles Aguilar', 'Caitlyn Green', 'Ronald Davis',
       'Melissa Lucas', 'Heidi Leonard', 'Richard Swanson',
       'Lauren Simpson', 'Marissa Raymond', 'Joyce Bishop',
       'Michael Kelly', 'Peggy Hill', 'Derrick Love', 'Laura Stevens',
       'Terri Potter', 'Veronica Saunders', 'Emily Townsend',
       'James Long', 'Jennifer Dickson', 'Alexa West', 'Phillip Donovan',
       'Monique King', 'Karl Murray', 'Julia Brooks', 'Lauren Montgomery',
       'Nathan Wall', 'Kristin Macias', 'Samuel Smith', 'Bradley Stevens',
       'Morgan Hampton', 'Lori Carrillo', 'Samantha Novak',
       'Audrey Robinson', 'Lindsey Kerr', 'Cynthia Nguyen', 'Ryan Hood',
       'Matthew Stevenson', 'Rhonda White', 'Larry Ramsey',
       'Kimberly Anderson', 'Brian Miles', 'Kelly Baker', 'Wendy Miller',
       'Michael Webster', 'Michelle Ryan', 'Justin Jennings',
       'Norma Ward', 'Vincent Johnson', 'David Stone', 'Amy Evans',
       'Stephen Ramirez', 'Nicholas Baxter', 'Raymond Smith',
       'Roy Johnson', 'Ryan Vasquez', 'Justin Johnson', 'Kaylee Benjamin',
       'Charles Murphy', 'Grant Henderson', 'Michael Woods',
       'Michael Kane', 'Susan Benson', 'Timothy Armstrong',
       'Cindy Harvey', 'Caitlin Carter', 'Ricky Johnson', 'Bryan Hoover',
       'Elijah Wagner', 'Felicia Davis', 'Jonathan Hess', 'Michael Yoder',
       'Samantha York', 'Joanna Martinez', 'Penny Hopkins',
       'Justin Foster', 'Jennifer Morgan', 'Dalton Chen',
       'Mrs. Amy Carney MD', 'Melissa Wang', 'Cameron Bender',
       'Daniel Solomon', 'Deanna Jones', 'Alicia Young', 'Peter Figueroa',
       'Jacob Thompson', 'Jonathon Woodward', 'Brian Hayes',
       'Lisa Carroll', 'Cassandra Ryan', 'Terri Hayden', 'Taylor Burns',
       'Andrew Jimenez', 'Wendy King', 'Luis Carter', 'Nicholas Jones',
       'Tiffany Sanders', 'Tina Knight', 'Brittany Andrews',
       'William Weber', 'Angela Ortiz', 'David Meyer', 'Travis Harrell',
       'Craig Mccann', 'Scott Edwards', 'Kelly Cohen', 'David Diaz',
       'Joseph Green', 'Henry Johnson', 'Keith Wilson', 'Jordan Chase',
       'Kayla Solomon', 'Hayley Johnson', 'John Bender',
       'William Christian', 'Michael Willis', 'James Simmons',
       'Christopher Sullivan', 'Michelle Santos', 'Michael Walters',
       'Jason Anderson', 'James Hoover', 'Julie Garza', 'Lisa Williams',
       'Erica Thomas', 'Calvin Vargas', 'Molly Jones', 'Shannon Hogan',
       'James Fields', 'Arthur Johnson', 'Collin Edwards',
       'Kimberly Smith', 'Mr. Clinton Camacho', 'Taylor Moyer',
       'David Jimenez', 'Kyle Coleman', 'Marvin Cardenas',
       'Eddie Robertson', 'Christopher Taylor', 'Brenda Bell',
       'Ashley Nguyen', 'Michael Bailey', 'Maria Scott',
       'Katherine Mclaughlin', 'Karen Cruz', 'Jodi Taylor',
       'Thomas Mcgrath', 'Kathy Bailey', 'Karen Wolf', 'Mary Jones',
       'Lisa Houston', 'Ashley Jordan', 'Lonnie Williams',
       'Michael Poole', 'David Martinez', 'Eduardo Miller',
       'Kristine Williams', 'Audrey Woods', 'Lauren Brown',
       'Stephanie Merritt', 'Rebecca Jones', 'Isaiah Arnold',
       'Andrew Dudley', 'Kenneth Jones', 'Jane Griffith', 'Jeff Phillips',
       'Dawn Edwards', 'Paige Campbell', 'Belinda Kim',
       'Christopher Cruz', 'Ruben Steele', 'Bradley Martinez',
       'Alison Williams', 'Scott Ferguson', 'David Brown', 'Renee Scott',
       'Aimee Green', 'Ricky Jenkins', 'Kelly Allen MD', 'Sean Smith',
       'Robert Johnson', 'Susan Russell', 'Ryan West',
       'Christina Fitzgerald', 'Susan Roberson', 'Jessica Banks',
       'Jodi Rogers', 'Michelle Clay', 'Jacob Bean', 'Robert Harper',
       'Michelle Graham', 'Seth Johnson', 'Julia Sanchez', 'Justin Sharp',
       'Thomas Yoder', 'Christine Mullins', 'Timothy Mcbride',
       'Deborah Reyes', 'Chad Downs', 'William Hill', 'Makayla Rhodes',
       'Clayton Jones', 'Sean Hernandez', 'Christine Young', 'Brian Hale',
       'Sandra Hart', 'Christine Curtis', 'Tara Johnson',
       'Stephanie Snyder', 'Miguel Leblanc', 'Christina Mcintyre',
       'John Weber', 'Timothy Austin', 'Jeffrey Sullivan',
       'Michael Noble', 'Christine Shah', 'Lisa Maynard', 'Lisa Lynch',
       'Angela Warren', 'Mary Benson', 'Yvonne Krueger', 'Keith Bailey',
       'Barbara Case', 'Jamie Torres', 'Justin Jacobs', 'Kevin Reynolds',
       'Michelle Kelly', 'Terry Hernandez', 'Aaron Duarte',
       'Micheal Perez', 'Troy Bradford', 'Bobby Wilkinson', 'Juan Lopez',
       'Virginia Medina', 'Leah Holmes', 'Todd Brooks', 'Madeline Rush',
       'Nicole Frederick', 'Jason Adams', 'Mallory Fletcher',
       'David Cummings', 'Kelly Suarez', 'Drew Morse', 'Lauren Stephens',
       'Penny Miller', 'Paula Stewart', 'Jonathan Johnson', 'Laura Ross',
       'Brandy Wilson', 'Brittany Briggs', 'Carl Coleman',
       'Crystal Chandler', 'Elizabeth Nichols', 'Jesse Barber',
       'Julie Jackson', 'Cindy Morales', 'Ashley Franco',
       'Kimberly Calderon', 'Lisa Brooks', 'Destiny Todd', 'Dawn Luna',
       'Jacob Yoder', 'Martha Moody', 'James Torres', 'Shelly Stuart',
       'Julie Williams', 'Tamara Jones', 'Todd Orozco', 'Jeffrey Torres',
       'Chad Colon', 'Alan Williams', 'Cameron Austin', 'Dawn Nguyen',
       'Andrea Phillips', 'Robert Bishop', 'Melanie Mckinney',
       'Joseph Ellis', 'Jennifer Smith', 'Vickie Holloway', 'Nancy Munoz',
       'David Taylor', 'Amber Gray', 'Christian Barr', 'Mark Fleming',
       'Felicia Carpenter', 'Rebecca Kelly', 'Aaron Walls', 'Mark King',
       'James Macdonald', 'Alexander Thompson', 'Kaitlin Walton',
       'Ashley Kaufman', 'Megan Bernard', 'Kelly Harper', 'Daniel White',
       'Justin Castro', 'John Odonnell', 'Whitney Tapia', 'Jeff Lawrence',
       'Jeremy Shields', 'Matthew Davies', 'Christine Hamilton',
       'Crystal Proctor', 'Victoria Mcbride', 'Ellen Cross MD',
       'David Stevens', 'Frank Brown DDS', 'Deborah Barajas',
       'Richard Howard Jr.', 'Anita Dougherty', 'Alicia Watson',
       'Deborah Hunter', 'Morgan Gilbert', 'Taylor Crawford',
       'Danny Sanchez', 'Cory Rosario DDS', 'Kyle Perez',
       'Kelly Gibson MD', 'Angela Garcia', 'Dylan Vaughan',
       'Kevin Santiago', 'Terry Rodriguez', 'Chad Mason',
       'Jessica Ferguson', 'Andrew Schroeder', 'Jeffrey Garcia',
       'Jason Hood', 'Sandra Leon', 'Tyler Mccormick', 'Megan Murphy',
       'Charles Vincent', 'Mrs. Cindy Sharp DVM', 'Sarah Dixon',
       'Teresa Rodriguez', 'Shannon Spencer', 'Darlene Tapia',
       'Ryan Ball', 'Anita Berger', 'Tom Bradley', 'Melinda Cummings',
       'Melissa Donaldson', 'Charles Pearson', 'Isaiah Moore',
       'Jonathan Ramirez', 'Mrs. Melanie Mueller', 'Ms. Heather Dean',
       'David Malone', 'Melissa Brown', 'Jason Woods', 'Rachel Diaz',
       'Michael Jones', 'Taylor Mcgee', 'Christopher Ray',
       'Alyssa Berg MD', 'Dr. James Henry DDS', 'James Stevens',
       'Kevin Marquez', 'Colleen Porter', 'Christian Odom',
       'Calvin Macias', 'Kim Fischer', 'Travis Schultz', 'Bonnie Morris',
       'Tiffany Morris', 'Paul Poole', 'Richard Perez',
       'Bernard Whitaker', 'Melinda Miller', 'Christina Gonzales',
       'Maria Gray', 'Mary Bennett', 'Dr. Erik Pineda', 'Sandra Burns',
       'Patricia Hunt', 'Erika Kennedy', 'Timothy Newman', 'Tanya Bowen',
       'Rhonda Davenport', 'Cheryl Brown', 'Angela Bridges',
       'Brandon Rasmussen', 'Ryan Beltran', 'Allison Novak',
       'Kirsten Davis', 'Paige Turner', 'Aimee Christensen',
       'Jessica Howard', 'Lisa Zhang', 'Jonathan Daniel',
       'Cheyenne Oliver', 'Brianna Henderson', 'Mark Sanchez',
       'Melissa Nguyen', 'Jeff Carney', 'Tammy Tanner', 'Donna Robinson',
       'Eric Wilson', 'Linda Armstrong', 'Russell Jones',
       'Darlene Burton', 'Faith Johnson', 'Nathan Miller', 'Donna Fowler',
       'Whitney Jefferson', 'Shane Sanders', 'Felicia Cline',
       'Jennifer Clark', 'Miss Carla Coleman MD', 'William Nelson',
       'Crystal Brown', 'Erica Miller', 'Emily Martin', 'Wesley May',
       'Christopher Atkinson', 'Blake Martinez', 'Joy Henderson',
       'Mary Ramos', 'Colleen Sanchez', 'David Cruz', 'Ralph Becker',
       'Denise Coleman', 'Samantha Mendoza', 'Sean Lyons',
       'Deborah Hernandez', 'Kristi Price', 'Alex Douglas',
       'Ian Mcdonald', 'David Bauer', 'Steven Hill', 'Linda Beasley',
       'Maria Edwards', 'Megan Hutchinson', 'Jaime Simon',
       'Thomas Freeman', 'Holly Leon', 'Travis Pierce',
       'Kristen Galloway', 'Raven Taylor', 'Samuel Robinson',
       'Christina Johnson', 'Holly Spencer', 'Jessica Davies',
       'Daniel Sullivan', 'Richard Franklin', 'Samantha Klein',
       'Becky Perkins', 'Kimberly Pena', 'Rebecca Burnett',
       'Anna Martinez', 'Monica Johnson', 'Shannon Porter', 'Amy Stout',
       'Joseph Sherman', 'Maria Walls'], dtype=object)
3. Gender

#melihat isi dalam kolom
data['Gender'].unique()
     
array(['Male', 'Female', nan], dtype=object)

#jumlah missing value
data['Gender'].isnull().sum()
     
np.int64(45)
Kolom Gender merupakan data kategorik. Oleh karena itu, missing values dapat diisi dengan modus (nilai yang paling sering muncul):


#mengisi missing value
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
     

#jumlah missng value
np.sum(data['Gender'].isnull())
     
np.int64(0)

#melihat dalam kolom setelah mengisi missing value
data['Gender'].unique()
     
array(['Male', 'Female'], dtype=object)
4. AttendanceRate

#melihat isi dalam kolom
data['AttendanceRate'].unique()
     
array([85., 90., 78., 92., nan, 95., 70., 82., 91., 88.])

#jumlah missing value
data['AttendanceRate'].isnull().sum()
     
np.int64(39)
Kolom ini merupakan data numerik (persentase kehadiran). Karena kemungkinan terdapat outlier, maka pendekatan yang lebih aman adalah menggunakan median:


#mengisi missing value
data['AttendanceRate'] = data['AttendanceRate'].fillna(data['AttendanceRate'].median())
     

#jumlah missng value
np.sum(data['AttendanceRate'].isnull())
     
np.int64(0)

#melihat dalam kolom setelah mengisi missing value
data['AttendanceRate'].unique()
     
array([85., 90., 78., 92., 88., 95., 70., 82., 91.])
5. StudyHoursPerWeek

#melihat isi dalam kolom
data['StudyHoursPerWeek'].unique()
     
array([15., 20., 10., 25., 18., 30.,  8., 17., 12., 22., nan])

#jumlah missing value
data['StudyHoursPerWeek'].isnull().sum()
     
np.int64(49)
Data ini bersifat numerik dan berpotensi memiliki outlier. Oleh karena itu:


#mengisi missing value
data['StudyHoursPerWeek'] = data['StudyHoursPerWeek'].fillna(data['StudyHoursPerWeek'].median())
     

#jumlah missng value
data['StudyHoursPerWeek'].isnull().sum()
     
np.int64(0)

#melihat dalam kolom setelah mengisi missing value
data['StudyHoursPerWeek'].unique()
     
array([15., 20., 10., 25., 18., 30.,  8., 17., 12., 22.])
6. PreviousGrade

#melihat isi dalam kolom
data['PreviousGrade'].unique()
     
array([78., 85., 65., 90., 82., 88., 60., 77., 70., 86., nan])

#jumlah missing value
data['PreviousGrade'].isnull().sum()
     
np.int64(32)
Kolom nilai sebelumnya merupakan data numerik, sehingga:


#mengisi missing value
data['PreviousGrade'] = data['PreviousGrade'].fillna(data['PreviousGrade'].mean())
     

#jumlah missng value
data['PreviousGrade'].isnull().sum()
     
np.int64(0)

#melihat dalam kolom setelah mengisi missing value
data['PreviousGrade'].unique()
     
array([78.        , 85.        , 65.        , 90.        , 82.        ,
       88.        , 60.        , 77.        , 70.        , 86.        ,
       77.73599138])
7. ExtracurricularActivities

#melihat isi dalam kolom
data['ExtracurricularActivities'].unique()
     
array([ 1.,  2.,  0.,  3., nan])

#jumlah missing value
data['ExtracurricularActivities'].isnull().sum()
     
np.int64(43)
Kolom ini berupa numerik (jumlah kegiatan), sehingga:


#mengisi missing value
data['ExtracurricularActivities'] = data['ExtracurricularActivities'].fillna(data['ExtracurricularActivities'].median())
     

#jumlah missing value
data['ExtracurricularActivities'].isnull().sum()
     
np.int64(0)

#melihat dalam kolom setelah mengisi missing value
data['ExtracurricularActivities'].unique()
     
array([1., 2., 0., 3.])
8. ParentalSupport

#melihat isi dalam kolom
data['ParentalSupport'].unique()
     
array(['High', 'Medium', 'Low', nan], dtype=object)

#jumlah missing value
data['ParentalSupport'].isnull().sum()
     
np.int64(22)
Kolom ini bersifat kategorik, sehingga:


#mengisi missing value
data['ParentalSupport'] = data['ParentalSupport'].fillna(data['ParentalSupport'].mode()[0])
     

#jumlah missng value
data['ParentalSupport'].isnull().sum()
     
np.int64(0)

#melihat dalam kolom setelah mengisi missing value
data['ParentalSupport'].unique()
     
array(['High', 'Medium', 'Low'], dtype=object)
9. FinalGrade

#melihat isi dalam kolom
data['FinalGrade'].unique()
     
array([80., 87., 68., 92., 85., nan, 62., 78., 72., 88., 90.])

#jumlah missing value
data['FinalGrade'].isnull().sum()
     
np.int64(38)
Kolom ini merupakan target/nilai akhir, sehingga:


#mengisi missing value
data['FinalGrade'] = data['FinalGrade'].fillna(data['FinalGrade'].mean())
     

#jumlah missing value
data['FinalGrade'].isnull().sum()
     
np.int64(0)

#melihat dalam kolom setelah mengisi missing value
data['FinalGrade'].unique()
     
array([80.        , 87.        , 68.        , 92.        , 85.        ,
       80.13015184, 62.        , 78.        , 72.        , 88.        ,
       90.        ])
10. Study Hours

#melihat isi dalam kolom
data['Study Hours'].unique()
     
array([ 4.8,  2.2,  4.6,  2.9,  4.1,  2.8,  4.5,  1. ,  3.6,  3.4,  0.1,
        1.7,  1.3,  1.4,  2. ,  2.1,  0.8,  4.3,  0. ,  4.4,  3.8,  2.6,
        2.3,  2.4,  0.3,  3.1,  2.5,  3.2,  0.2,  3.7,  3. ,  3.3,  3.9,
        1.2,  0.7,  1.6,  4. ,  1.1,  0.4, -5. ,  3.5,  1.5,  0.6,  nan,
        4.7,  1.8,  0.9,  0.5,  2.7,  4.2,  1.9,  5. ,  4.9])

#jumlah missing value
data['Study Hours'].isnull().sum()
     
np.int64(22)
Karena mirip dengan StudyHoursPerWeek, maka:


#mengisi missing value
data['Study Hours'] = data['Study Hours'].fillna(data['Study Hours'].median())
     

#jumlah missing value
data['Study Hours'].isnull().sum()
     
np.int64(0)

#melihat dalam kolom setelah mengisi missing value
data['Study Hours'].unique()
     
array([ 4.8,  2.2,  4.6,  2.9,  4.1,  2.8,  4.5,  1. ,  3.6,  3.4,  0.1,
        1.7,  1.3,  1.4,  2. ,  2.1,  0.8,  4.3,  0. ,  4.4,  3.8,  2.6,
        2.3,  2.4,  0.3,  3.1,  2.5,  3.2,  0.2,  3.7,  3. ,  3.3,  3.9,
        1.2,  0.7,  1.6,  4. ,  1.1,  0.4, -5. ,  3.5,  1.5,  0.6,  4.7,
        1.8,  0.9,  0.5,  2.7,  4.2,  1.9,  5. ,  4.9])
11. Attendance (%)

#melihat isi dalam kolom
data['Attendance (%)'].unique()
     
array([ 59.,  70.,  92.,  96.,  97.,  50.,  84.,  62.,  52.,  90.,  91.,
        87.,  60.,  98.,  69.,  58.,  75.,  64.,  67.,  56.,  54.,  63.,
        76.,  68.,  57.,  73.,  65., 100.,  72.,  74.,  88.,  80.,  94.,
        82.,  83.,  79.,  nan, 200.,  66.,  53.,  77.,  95.,  78.,  89.,
        81.,  55.,  85.,  86.,  93.,  61.,  99.,  71.,  51.])

#jumlah missing value
data['Attendance (%)'].isnull().sum()
     
np.int64(40)
Kolom ini juga numerik:


#mengisi missing value
data['Attendance (%)'] = data['Attendance (%)'].fillna(data['Attendance (%)'].median())
     

#jumlah missing value
data['Attendance (%)'].isnull().sum()
     
np.int64(0)

#melihat dalam kolom setelah mengisi missing value
data['Attendance (%)'].unique()
     
array([ 59.,  70.,  92.,  96.,  97.,  50.,  84.,  62.,  52.,  90.,  91.,
        87.,  60.,  98.,  69.,  58.,  75.,  64.,  67.,  56.,  54.,  63.,
        76.,  68.,  57.,  73.,  65., 100.,  72.,  74.,  88.,  80.,  94.,
        82.,  83.,  79., 200.,  66.,  53.,  77.,  95.,  78.,  89.,  81.,
        55.,  85.,  86.,  93.,  61.,  99.,  71.,  51.])
12. Online Classes Taken

#melihat isi dalam kolom
data['Online Classes Taken'].unique()
     
array([False, True, nan], dtype=object)

#jumlah missing value
data['Online Classes Taken'].isnull().sum()
     
np.int64(25)
Kolom ini kategorik:


#mengisi missing value
data['Online Classes Taken'] = data['Online Classes Taken'].fillna(data['Online Classes Taken'].mode()[0])
     
/tmp/ipykernel_855/2801392645.py:2: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  data['Online Classes Taken'] = data['Online Classes Taken'].fillna(data['Online Classes Taken'].mode()[0])

#jumlah missing value
data['Online Classes Taken'].isnull().sum()
     
np.int64(0)

#melihat dalam kolom setelah mengisi missing value
data['Online Classes Taken'].unique()
     
array([False,  True])
Cek Kembali Missing Values

np.sum(data.isnull())
     
/usr/local/lib/python3.12/dist-packages/numpy/_core/fromnumeric.py:84: FutureWarning: The behavior of DataFrame.sum with axis=None is deprecated, in a future version this will reduce over both axes and return a scalar. To retain the old behavior, pass axis=0 (or do not pass axis)
  return reduction(axis=axis, out=out, **passkwargs)
0
StudentID	0
Name	0
Gender	0
AttendanceRate	0
StudyHoursPerWeek	0
PreviousGrade	0
ExtracurricularActivities	0
ParentalSupport	0
FinalGrade	0
Study Hours	0
Attendance (%)	0
Online Classes Taken	0

dtype: int64

data.info()
     
<class 'pandas.core.frame.DataFrame'>
Index: 960 entries, 0 to 999
Data columns (total 12 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   StudentID                  960 non-null    float64
 1   Name                       960 non-null    object 
 2   Gender                     960 non-null    object 
 3   AttendanceRate             960 non-null    float64
 4   StudyHoursPerWeek          960 non-null    float64
 5   PreviousGrade              960 non-null    float64
 6   ExtracurricularActivities  960 non-null    float64
 7   ParentalSupport            960 non-null    object 
 8   FinalGrade                 960 non-null    float64
 9   Study Hours                960 non-null    float64
 10  Attendance (%)             960 non-null    float64
 11  Online Classes Taken       960 non-null    bool   
dtypes: bool(1), float64(8), object(3)
memory usage: 90.9+ KB
MENGISI Kolom Lain


# Kolom numerik (pakai median)
data['StudentID'] = data['StudentID'].fillna(data['StudentID'].median())
data['AttendanceRate'] = data['AttendanceRate'].fillna(data['AttendanceRate'].median())
data['StudyHoursPerWeek'] = data['StudyHoursPerWeek'].fillna(data['StudyHoursPerWeek'].median())
data['PreviousGrade'] = data['PreviousGrade'].fillna(data['PreviousGrade'].median())
data['ExtracurricularActivities'] = data['ExtracurricularActivities'].fillna(data['ExtracurricularActivities'].median())
data['FinalGrade'] = data['FinalGrade'].fillna(data['FinalGrade'].median())
data['Study Hours'] = data['Study Hours'].fillna(data['Study Hours'].median())
data['Attendance (%)'] = data['Attendance (%)'].fillna(data['Attendance (%)'].median())

# Kolom kategorik (pakai modus)
data['Name'] = data['Name'].fillna(data['Name'].mode()[0])
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['ParentalSupport'] = data['ParentalSupport'].fillna(data['ParentalSupport'].mode()[0])
data['Online Classes Taken'] = data['Online Classes Taken'].fillna(data['Online Classes Taken'].mode()[0])
     

print(data.isnull().sum())
     
StudentID                    0
Name                         0
Gender                       0
AttendanceRate               0
StudyHoursPerWeek            0
PreviousGrade                0
ExtracurricularActivities    0
ParentalSupport              0
FinalGrade                   0
Study Hours                  0
Attendance (%)               0
Online Classes Taken         0
dtype: int64

data.info()
     
<class 'pandas.core.frame.DataFrame'>
Index: 960 entries, 0 to 999
Data columns (total 12 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   StudentID                  960 non-null    float64
 1   Name                       960 non-null    object 
 2   Gender                     960 non-null    object 
 3   AttendanceRate             960 non-null    float64
 4   StudyHoursPerWeek          960 non-null    float64
 5   PreviousGrade              960 non-null    float64
 6   ExtracurricularActivities  960 non-null    float64
 7   ParentalSupport            960 non-null    object 
 8   FinalGrade                 960 non-null    float64
 9   Study Hours                960 non-null    float64
 10  Attendance (%)             960 non-null    float64
 11  Online Classes Taken       960 non-null    bool   
dtypes: bool(1), float64(8), object(3)
memory usage: 90.9+ KB
CEK DAN PENANGANAN OUTLIER
1. Deteksi Outlier dengan Boxplot

import matplotlib.pyplot as plt
import seaborn as sns

# Kolom numerik pada dataset
num_cols = [
    'AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade',
    'ExtracurricularActivities', 'FinalGrade',
    'Study Hours', 'Attendance (%)'
]

plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(y=data[col])
    plt.title(col)

plt.tight_layout()
plt.show()
     

2. Deteksi Outlier dengan Metode IQR

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# Cek jumlah outlier
for col in num_cols:
    outliers = detect_outliers_iqr(data, col)
    print(f"{col}: {len(outliers)} outliers")
     
AttendanceRate: 0 outliers
StudyHoursPerWeek: 0 outliers
PreviousGrade: 0 outliers
ExtracurricularActivities: 0 outliers
FinalGrade: 0 outliers
Study Hours: 10 outliers
Attendance (%): 10 outliers
PENANGANAN OUTLIER
A. Winsorizing

import numpy as np

def winsorize_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])

# Terapkan ke semua kolom numerik
for col in num_cols:
    winsorize_iqr(data, col)
     

#cek dengan boxplot
plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(y=data[col])
    plt.title(col)

plt.tight_layout()
plt.show()
     

B. Menghapus Outlier

for col in num_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
     
ENCODING DATA KATEGORIK
Encoding dilakukan untuk mengubah data kategorik menjadi numerik agar dapat diproses oleh model Machine Learning.

1. Label Encoding

# Melihat isi kategori
print(data['Gender'].unique())
print(data['ParentalSupport'].unique())
print(data['Online Classes Taken'].unique())
     
['Male' 'Female']
['High' 'Medium' 'Low']
[False  True]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['Gender'] = le.fit_transform(data['Gender'])
data['ParentalSupport'] = le.fit_transform(data['ParentalSupport'])
data['Online Classes Taken'] = le.fit_transform(data['Online Classes Taken'])

# Cek hasil
print(data[['Gender', 'ParentalSupport', 'Online Classes Taken']].head())
     
   Gender  ParentalSupport  Online Classes Taken
0       1                0                     0
1       0                2                     1
2       1                1                     0
3       1                0                     0
4       0                2                     1
2. One-Hot Encoding

data = pd.get_dummies(data, columns=[
    'Gender', 'ParentalSupport', 'Online Classes Taken'
], drop_first=True)
     

data.head()
     
StudentID	Name	AttendanceRate	StudyHoursPerWeek	PreviousGrade	ExtracurricularActivities	FinalGrade	Study Hours	Attendance (%)	Gender_1	ParentalSupport_1	ParentalSupport_2	Online Classes Taken_1
0	1.0	John	85.0	15.0	78.0	1.0	80.0	4.8	59.0	True	False	False	False
1	2.0	Sarah	90.0	20.0	85.0	2.0	87.0	2.2	70.0	False	False	True	True
2	3.0	Alex	78.0	10.0	65.0	0.0	68.0	4.6	92.0	True	True	False	False
3	4.0	Michael	92.0	25.0	90.0	3.0	92.0	2.9	96.0	True	False	False	False
4	5.0	Emma	88.0	18.0	82.0	2.0	85.0	4.1	97.0	False	False	True	True

np.sum(data.isnull())
     
/usr/local/lib/python3.12/dist-packages/numpy/_core/fromnumeric.py:84: FutureWarning: The behavior of DataFrame.sum with axis=None is deprecated, in a future version this will reduce over both axes and return a scalar. To retain the old behavior, pass axis=0 (or do not pass axis)
  return reduction(axis=axis, out=out, **passkwargs)
0
StudentID	0
Name	0
AttendanceRate	0
StudyHoursPerWeek	0
PreviousGrade	0
ExtracurricularActivities	0
FinalGrade	0
Study Hours	0
Attendance (%)	0
Gender_1	0
ParentalSupport_1	0
ParentalSupport_2	0
Online Classes Taken_1	0

dtype: int64

data.info()
     
<class 'pandas.core.frame.DataFrame'>
Index: 960 entries, 0 to 999
Data columns (total 13 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   StudentID                  960 non-null    float64
 1   Name                       960 non-null    object 
 2   AttendanceRate             960 non-null    float64
 3   StudyHoursPerWeek          960 non-null    float64
 4   PreviousGrade              960 non-null    float64
 5   ExtracurricularActivities  960 non-null    float64
 6   FinalGrade                 960 non-null    float64
 7   Study Hours                960 non-null    float64
 8   Attendance (%)             960 non-null    float64
 9   Gender_1                   960 non-null    bool   
 10  ParentalSupport_1          960 non-null    bool   
 11  ParentalSupport_2          960 non-null    bool   
 12  Online Classes Taken_1     960 non-null    bool   
dtypes: bool(4), float64(8), object(1)
memory usage: 78.8+ KB
