
'''
Classification:
----------------
'''

import numpy as np
import random
from numpy import genfromtxt
import math

data_path= 'D:\Summer-2021\Ai lab\Class3\offiline-1\iris.csv'
my_data= genfromtxt(data_path,delimiter=',')
np.random.shuffle(my_data)

def Euclidean_Distance(x, y):
    distance =0.0
    for i in range(len(x)-1):
         distance +=pow((x[i]-y[i]),2)
    return math.sqrt(distance)

train_set = []
validation_set = []
test_set = []

#dividing the dataset
for i in range(0,my_data.shape[0]):
  e = np.random.random()
  if e>=0 and e<=0.70:
    train_set.append(my_data[i])
  elif e>0.70 and e<=0.85:
    validation_set.append(my_data[i])
  else:
    test_set.append(my_data[i])

train_set = np.array(train_set)
validation_set = np.array(validation_set)
test_set = np.array(test_set)

#slicing
X_train_set = train_set[:,0:4]
y_train_set = train_set[:,4]

X_val_set = validation_set[:,0:4]
y_val_set = validation_set[:,4]

X_test_set = test_set[:,0:4]
y_test_set = test_set[:,4]

k=5

pred_val_set = np.zeros((y_val_set.shape))


for i in range(0, validation_set.shape[0]):
  dist_array = np.zeros((X_train_set.shape[0], 2))
  for j in range(0, train_set.shape[0]):
    #Find Euclidean distance
    dist=Euclidean_Distance(X_val_set[i],X_train_set[j])
    dist_array[j][0] = j
    dist_array[j][1] = dist
      
  #Sorting in ascending order
  dist_array = dist_array[dist_array[:, 1].argsort()]
  temp_voting = np.zeros((k))
  for l in range(0, k):
    closest_index = int(dist_array[l][0])
    temp_voting[l] = y_train_set[closest_index]

  temp_voting = temp_voting.astype('int')
  values, counts = np.unique(temp_voting, return_counts=True)
  ind = np.argmax(counts)
  pred_value = values[ind]
  pred_val_set[i] = pred_value

total_element = y_val_set.shape[0]
correctly_predicted = 0
for i in range(0,y_val_set.shape[0]):
  if y_val_set[i] == pred_val_set[i]:
    correctly_predicted += 1

#Calculate validation accuracy
accuracy = (correctly_predicted/total_element) * 100
print(accuracy)



'''
Result for classification:
-----------
k			accuracy
----------------------------------
1	----->	92.85714285714286
3	----->	89.28571428571429
5	----->	88.0
9	----->	96.55172413793103
'''

'''
Regression:
--------------
'''

import numpy as np
import random
from numpy import genfromtxt
data_path= 'D:\Summer-2021\Ai lab\Class3\offiline-1\diabetes.csv'
my_data= genfromtxt(data_path,delimiter=',')
np.random.shuffle(my_data)

import math
def Euclidean_Distance(x, y):
    distance =0.0
    for i in range(len(x)-1):
         distance +=pow((x[i]-y[i]),2)
    return math.sqrt(distance)

train_set = []
validation_set = []
test_set = []

for i in range(0,my_data.shape[0]):
  e = np.random.random()
  if e>=0 and e<=0.70:
    train_set.append(my_data[i])
  elif e>0.70 and e<=0.85:
    validation_set.append(my_data[i])
  else:
    test_set.append(my_data[i])

train_set = np.array(train_set)
validation_set = np.array(validation_set)
test_set = np.array(test_set)

X_train_set = train_set[:,0:10]
y_train_set = train_set[:,10]

X_val_set = validation_set[:,0:10]
y_val_set = validation_set[:,10]

X_test_set = test_set[:,0:10]
y_test_set = test_set[:,10]

k=9
pred_val_set = np.zeros((y_val_set.shape))
#print(y_val_set.shape)
#print(pred_val_set)


for i in range(0,validation_set.shape[0]):
  dist_array = np.zeros((X_train_set.shape[0],2))
  for j in range(0,train_set.shape[0]):
    dist = np.linalg.norm(X_val_set[i] - X_train_set[j])
    dist_array[j][0] = j
    dist_array[j][1] = dist
  dist_array = dist_array[dist_array[:,1].argsort()]
  temp_voting = np.zeros((k))
  for l in range(0,k):
    closest_index=int(dist_array[l][0])
    temp_voting[l] = y_train_set[closest_index]

  values, counts = np.unique(temp_voting, return_counts=True)
  #ind = np.argmax(counts)
  counts = np.unique(temp_voting, return_counts=True)
  ind = np.average(counts)
  pred_value = ind
  pred_val_set[i] = pred_value

total_element = y_val_set.shape[0]
error = 0
for i in range(0,y_val_set.shape[0]):
  error = error + ((y_val_set[i] - pred_val_set[i]) **2)

Mean_Squared_Error = (error/total_element)
print(Mean_Squared_Error)


'''
Result for Regression:
--------------------------
k		Mean_Squared_Error
----------------------------------
1	----->	10391.279411764706
3	----->	10139.784396701385
5	----->	12122.897033730163
9	----->	8576.436121565705
'''
