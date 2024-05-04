import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data=pd.read_csv('home_data.csv')

features = ['sqft_living']  # List of feature column names
X = data[features]  # Select multiple feature columns
y_actual = data['price']           # Select the target column (price)


########################################### this is the regression and the scatter plot##########################################

## Create the linear regression model
model = LinearRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X, y_actual, test_size=0.2, random_state=20)

# # Train the model
model=model.fit(X_train,Y_train )
# the regression result!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
y_pred = model.predict(X_test)
rms = mean_squared_error(Y_test, y_pred, squared=False)
print(rms,"simple")
# print(model.coef_)

# plt.scatter(data['size'],data['price'])
# plt.ticklabel_format(style='plain')
# plt.plot(data['size'], y_pred, color="black", linewidth=3)
# plt.show()

########################################### this is the box plot##########################################
# ax=data.boxplot(column='price', by='zip_code',figsize=(20,5))
# ax.set_ylabel('price')
# plt.show()

############################################regression with more features###############################
model2 = LinearRegression()
advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]  # List of feature column names
X = data[advanced_features]
y_actual= data['price'] 

X_train, X_test, Y_train, Y_test = train_test_split(X, y_actual, test_size=0.2, random_state=20)

# Fit the model to the training data
model2=model2.fit(X_train, Y_train)
y_pred=model2.predict(X_test)
rms2 = mean_squared_error(Y_test, y_pred, squared=False)
print(rms2,"detailed")

test_data=pd.read_csv('home_data.csv')
house=test_data.iloc[5]
print("real price: ",house['price'])
# Assuming 'house' is a Series or DataFrame row containing all features
test_features = house[advanced_features]  # Extract relevant features based on 'features' list
test_features_for_first_model=house[features]
print("model detailed: ",model2.predict([test_features]))
print("model not detailed:",model.predict([test_features_for_first_model]))
