from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import numpy as np

np.random.seed(7)

# df_all_data = pd.read_csv("../df_master_all.csv",sep=',',header=0)
# df_all_data2 = pd.concat([df_all_data.ix[0:500,1:15],df_all_data.ix[0:500,357]],axis=1)
# len(df_all_data2)
# df_all_data2.shape
# df_all_data2.head(2)
# df_all_data2.to_csv("df_master_all_small.csv",index=False)

df_all_data = pd.read_csv("df_master_all_small.csv",sep=',',header=0)
df_all_data.shape

X = df_all_data.iloc[:,0:14]
Y = df_all_data.iloc[:,14]

# create model
model = Sequential()
model.add(Dense(14, input_dim=14, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X, Y, epochs=10, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)