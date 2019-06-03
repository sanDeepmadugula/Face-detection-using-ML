#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import os
os.chdir('C:\\Analytics\\MachineLearning\\face detection')


# In[2]:


import warnings
warnings.filterwarnings('ignore')
print('Warnings Ignored!!')


# In[3]:


data = np.load('olivetti_faces.npy')
target = np.load('olivetti_faces_target.npy')


# In[4]:


print('There are {} images in the dataset'.format(len(data)))
print('There are {} unique target in the dataset'.format(len(np.unique(target))))
print('size of each image is {} x {}'.format(data.shape[1],data.shape[2]))
print('Pixel values were scaled to [0,1] interval. e.g:{}'.format(data[0][0,:4]))


# In[5]:


print('unique target number:',np.unique(target))


# In[6]:


# to show the distinct people in the dataset
def show_40_distinct_people(images,unique_ids):
    fig, axarr = plt.subplots(nrows=4,ncols=10,figsize=(25,12))
    axarr = axarr.flatten()
    
    # iterating over user ids
    for unique_id in unique_ids:
        image_index = unique_id*10
        axarr[unique_id].imshow(images[image_index],cmap='gray')
        axarr[unique_id].set_xticks([])
        axarr[unique_id].set_yticks([])
        axarr[unique_id].set_title('face id:{}'.format(unique_id))
    plt.suptitle('There are 40 distinct prople in the dataset')


# In[7]:


show_40_distinct_people(data,np.unique(target))


# In[10]:


# Show 10 Face Images of selected Target
def show_10_faces_of_n_subject(images,subect_ids):
    cols = 10 # each subject has 10 distinct face images
    rows = (len(subect_ids)*10)/cols
    rows = int(rows)
    fig, axarr = plt.subplots(nrows=rows,ncols=cols,figsize=(25,12))
    for i, subect_id in enumerate(subect_ids):
        for j in range(cols):
            image_index = subect_id*10+j
            axarr[i,j].imshow(images[image_index],cmap='gray')
            axarr[i,j].set_xticks([])
            axarr[i,j].set_yticks([])
            axarr[i,j].set_title('face id:{}'.format(subect_id))


# In[11]:


show_10_faces_of_n_subject(images=data,subect_ids=[0,5,21,24,36])


#  Machine learning Model of Face recognition

# In[12]:


X = data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
print(X.shape)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3,stratify=target,random_state=0)
print('X_train shape:', X_train.shape)
print('y_train shape:{}'.format(y_train.shape))


# In[14]:


y_frame = pd.DataFrame()
y_frame['subject ids'] = y_train
y_frame.groupby(['subject ids']).size().plot.bar(figsize=(25,20),title='Number of Samples for Each Classes')


# Performing PCA

# In[16]:


import mglearn


# In[17]:


mglearn.plots.plot_pca_illustration()


# PCA projection of Defined number of target

# In[18]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)


# In[20]:


number_of_people = 10
index_range = number_of_people * 10
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)
scatter = ax.scatter(X_pca[:index_range,0],
                          X_pca[:index_range,1],
                          c = target[:index_range],
                          s = 10,
                          cmap = plt.get_cmap('jet',number_of_people))

ax.set_xlabel('First Principle Component')
ax.set_ylabel('Second Priciple Component')
ax.set_title('PCA projection of {} people'.format(number_of_people))
fig.colorbar(scatter)


# Finding optimum number of PCA

# In[21]:


pca = PCA()
pca.fit(X)
plt.figure(1,figsize=(12,8))
plt.plot(pca.explained_variance_,linewidth=2)

plt.xlabel('Components')
plt.ylabel('Explained Variances')
plt.show()


# In[22]:


# choosing components as 90
n_components = 90


# In[23]:


pca = PCA(n_components=n_components, whiten=True)
pca.fit(X_train)


# In[24]:


# To show average face
fig,ax = plt.subplots(1,1,figsize=(8,8))
ax.imshow(pca.mean_.reshape((64,64)),cmap='gray')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Average Face')


# Showing the eigen faces

# In[28]:


number_of_eigenfaces=len(pca.components_)
eigen_faces=pca.components_.reshape((number_of_eigenfaces, data.shape[1], data.shape[2]))

cols=10
rows=int(number_of_eigenfaces/cols)
fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))
axarr=axarr.flatten()
for i in range(number_of_eigenfaces):
    axarr[i].imshow(eigen_faces[i],cmap="gray")
    axarr[i].set_xticks([])
    axarr[i].set_yticks([])
    axarr[i].set_title("eigen id:{}".format(i))
plt.suptitle("All Eigen Faces".format(10*"=", 10*"="))


# Performing classification result

# In[29]:


X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# In[30]:


clf = SVC()
clf.fit(X_train_pca,y_train)
y_pred = clf.predict(X_test_pca)
print('accuracy score:{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)))


# In[31]:


import seaborn as sns
plt.figure(1, figsize=(15,9))
sns.heatmap(metrics.confusion_matrix(y_test,y_pred))


# In[32]:


print(metrics.classification_report(y_test,y_pred))


# More Results

# In[33]:


models = []
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("SVM",SVC()))

for name,model in models:
    clf = model
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    print(10*"=","{} Result".format(name).upper(),10*"=")
    print('Accuracy score:{:0.2f}'.format(metrics.accuracy_score(y_test,y_pred)))
    print()


# Validated Results

# In[34]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
pca = PCA(n_components=n_components,whiten=True)
pca.fit(X)
X_pca = pca.transform(X)
for name,model in models:
    kfold = KFold(n_splits=5,shuffle=True,random_state=0)
    cv_scores = cross_val_score(model,X_pca,target,cv=kfold)
    print("{} mean cross validations score:{:.2f}".format(name,cv_scores.mean()))


# In[35]:


lr = LinearDiscriminantAnalysis()
lr.fit(X_train_pca,y_train)
y_pred = lr.predict(X_test_pca)
print('Accuracy sore:{:.2f}'.format(metrics.accuracy_score(y_test,y_pred)))


# In[36]:


cm = metrics.confusion_matrix(y_test,y_pred)

plt.subplots(1, figsize=(12,12))
sns.heatmap(cm)


# In[38]:


print("Classification Results:\n{}".format(metrics.classification_report(y_test, y_pred)))


# Using Leave one out cross validation

# In[39]:


from sklearn.model_selection import LeaveOneOut
loo_cv = LeaveOneOut()
clf = LogisticRegression()
cv_scores = cross_val_score(clf,
                            X_pca,
                            target,
                            cv = loo_cv)

print("{} Leave One Out cross-validation mean accuracy score:{:.2f}".format(clf.__class__.__name__, cv_scores.mean())) 
                                                                            


# In[40]:


from sklearn.model_selection import LeaveOneOut
loo_cv = LeaveOneOut()
clf = LinearDiscriminantAnalysis()
cv_scores = cross_val_score(clf,
                            X_pca,
                            target,
                            cv = loo_cv)
print("{} Leave One Out cross-validation mean accuracy score:{:.2f}".format(clf.__class__.__name__, 
                                                                            cv_scores.mean()))


# In[41]:


from sklearn.model_selection import GridSearchCV


# In[42]:


from sklearn.model_selection import LeaveOneOut
lr=LogisticRegression(C=1.0, penalty="l2")
lr.fit(X_train_pca, y_train)
print("lr score:{:.2f}".format(lr.score(X_test_pca, y_test)))


# Precision- Recall- ROC Curves

# In[44]:


from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
Target = label_binarize(target, classes=range(40))
print(Target.shape)
print(Target[0])
n_classes = Target.shape[1]


# In[45]:


X_train_multiclass, X_test_multiclass,y_train_multiclass,y_test_multiclass = train_test_split(X, Target, test_size=0.3, stratify= Target, random_state=0)


# In[46]:


pca = PCA(n_components=n_components,whiten=True)
pca.fit(X_train_multiclass)
X_train_multiclass_pca = pca.transform(X_train_multiclass)
X_test_multiclass_pca = pca.transform(X_test_multiclass)


# In[48]:


oneRestClassifier=OneVsRestClassifier(lr)

oneRestClassifier.fit(X_train_multiclass_pca, y_train_multiclass)
y_score=oneRestClassifier.decision_function(X_test_multiclass_pca)


# In[49]:


# for each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i],recall[i], _ = metrics.precision_recall_curve(y_test_multiclass[:,i],y_score[:,i])
    precision['micro'], recall['micro'], _ = metrics.precision_recall_curve(y_test_multiclass.ravel(),y_score.ravel())
    
    average_precision['micro'] = metrics.average_precision_score(y_test_multiclass,y_score,average='micro')
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


# In[53]:


from sklearn.utils.fixes import signature

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.figure(1, figsize=(12,8))
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                 **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))


# Machine Learning Automated Workflow: Pipeline

# In[54]:


from sklearn.pipeline import Pipeline


# In[55]:


work_flows_std = list()
work_flows_std.append(('lda', LinearDiscriminantAnalysis(n_components=n_components)))
work_flows_std.append(('logReg', LogisticRegression(C=1.0, penalty="l2")))
model_std = Pipeline(work_flows_std)
model_std.fit(X_train, y_train)
y_pred=model_std.predict(X_test)


# In[56]:


print("Accuracy score:{:.2f}".format(metrics.accuracy_score(y_test, y_pred)))
print("Classification Results:\n{}".format(metrics.classification_report(y_test, y_pred)))


# In[ ]:




