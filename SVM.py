import pandas as pd
import os
import skimage
import sklearn
from skimage import transform
from skimage import io
import cv2
#from skimage.transform import resize
#from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
Categories=['Bon','Mauvais']
flat_data_arr=[] #input array
target_arr=[] #output array
datadir="C:/Users/CHEF SIRE YAKRO/Documents/ProjetMetal/C45_Perlit-Ferrit" 
#path which contains all the categories of images
for i in Categories:
    
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        img_resized=transform.resize(img_array,(150,150,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data) #dataframe
df['Target']=target
x=df.iloc[:,:-1] #input data 
y=df.iloc[:,-1] #output data
#construction
from sklearn import svm
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)

#train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
print('Splitted Successfully')
model.fit(x_train,y_train)
print('The Model is trained well with the given images')
# model.best_params_ contains the best parameters obtained from GridSearchCV

#test
y_pred=model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {sklearn.metrics.accuracy_score(y_pred,y_test)*100}% accurate")

#evaluation
#url=input()
img=cv2.imread("C:/Users/CHEF SIRE YAKRO/Documents/ProjetMetal/C45_Perlit-Ferrit/BON/Ref.2_211007_1.Testlauf_C45_Perlit-Ferrit_0_b0t88c0x0-2584y0-2325.jpg")
plt.imshow(img)
plt.show()
img_resize=cv2.resize(img,(150,150))
l=[img_resize.flatten()]
probability=model.predict_proba(l)
for ind,val in enumerate(Categories):
    print(f'{val} = {probability[0][ind]*100}%')
print("The predicted image is : "+Categories[model.predict(l)[0]])