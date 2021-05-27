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
                                       
#Create classifier object

from sklearn.neighbors import KNeighboursClassifier
knn = KNeighborsClassifier (n_neighbors = 5)

#Train the classifier (fit the estimator) using the training data
knn.fit(X_train , y_train)

#Estimate the accuracy of thw classifier on future data using the test data
knn.score(X_test , y_test)

#Use the trained k-NN classifier model to classify new, previously unseen objects
#first example : a small fruit with mass 20g , widhth 4.3cm , height 5.5 cm
fruit_prediction = knn.predict ([[20,4.3,5.5]])
lookup_fruit_name[fruit_predition[0]]

#second example : a larger , alongated fruit with mass 100g, width 6.3cm, height 8.5cm
fruit_prediction = knn.predict([[100,6.3,8.5]])
lookup_fruit_name[fruit_prediction[0]]

#Plot the decision boundaries of the k-NN classifier
from adspy_shared_utilities import plot_fruit_knn

plot_fruit_knn(X_train, y_train, 5, 'uniform') # we choose 5 nearest neighboures

#How sensitive is k-NN classification accuracy to the choice of 'k' parameter?
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors =k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
  
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);

#How sensitive is k-NN classification accuracy to he train/test split proportion ?
t=[0.8,0.7,0.6,0.5,0.4,0.3,0.2]
knn= KNeighborsClassifier(n_neighbors = 5)
plt.figure()
for s in t:
    scores = []
    for i in range(1,1000) :
      X_train , X_test , y_train, y_test = train_test_aplit(X, y , test_size = 1-s)
      knn.fit(X_train, y_train)
      scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores) , 'bo')
    
plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');


