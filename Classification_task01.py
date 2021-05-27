#Import required modules and load data file
%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

fruits = pd.read_table('readonly/fruit_data_with_colors.txt')

fruits.head()

# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_fruit_name = dict(zip(fruits.fruit.label.unique(), fruits.fruit_name.unique())) 
lookup_fruit_name

#The file contains the mass, height, and width of a selection of oranges, lemons and apples. The heights were measured along the core of the fruit. The widths were the widest width perpendicular to the height.



#Examining the data

#ploting a scatter matrix
from matplotlib import cm

X = fruits['height' , 'width' , 'mass' , 'color_score' ]]
Y = fruits['fruit_label']
X_train, X_test, y_train, marker = 'o' , s=40, hist_kwds={'bins':15}, figsize=(9,9) , cmap=cmap)

#Plotting a 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['width'],X_tain['height'], X_tain['color_score'], c=y_train, marker='o' , s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()


#create train-test split

#For this example , we use the mass , width , height features of each fruit instance
X = fruits[['mass','width','height']]
Y = fruits[['fruit_label']]

# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test_split(X, y, random_state=0) 
                                       
