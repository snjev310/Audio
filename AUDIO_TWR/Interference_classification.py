#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_excel('audio_path.xlsx')


# In[4]:


del df['Unnamed: 0']


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


def cal_mfcc(path):
    X,sample_rate = librosa.load(path,res_type = 'kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    feature = mfcc
    return feature


# In[9]:


df['mfcc'] = df['path'].apply(cal_mfcc)


# In[10]:


sns.distplot(df['mfcc'][1],color='red')
plt.title('I=3 Distribution plot')


# In[11]:


sns.distplot(df['mfcc'][27],color='blue')
plt.title('I=4 Distribution plot')


# In[12]:


sns.distplot(df['mfcc'][78],color='green')
plt.title('I=5 Distribution plot')


# In[13]:


plt.figure(figsize=(12,8))
sns.distplot(df['mfcc'][1],color='red',label='I-3')
plt.title('I=3 Distribution plot')
sns.distplot(df['mfcc'][27],color='blue',label='I-4')
plt.title('I=4 Distribution plot')
sns.distplot(df['mfcc'][78],color='green',label='I-5')
plt.title('I=5 Distribution plot')
plt.legend()


# In[14]:


p=0
for i in df['mfcc'][1]:
    if i >-50 and i<50:
        p = p+1
print(p)


# In[15]:


df['mfcc_mean'] = df['mfcc'].apply(lambda mfcc : mfcc.mean() )


# In[16]:


df.head()


# In[17]:


sns.distplot(df['mfcc_mean'])


# In[18]:


df.sort_values(by='mfcc_mean')


# In[ ]:





# In[19]:


from sklearn.preprocessing import LabelEncoder
X = np.array(df['mfcc'].tolist())
y = np.array(df['interference'].tolist())


# In[20]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=123)


# In[21]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()


# In[22]:


clf.fit(X_train,y_train)


# In[23]:


clf.score(X_test,y_test)


# In[24]:


predict = clf.predict(X_test)


# In[25]:


from sklearn.metrics import classification_report,confusion_matrix


# In[26]:


print(confusion_matrix(y_test,predict))


# In[27]:


print(classification_report(y_test,predict))


# In[28]:


from sklearn.neural_network import MLPClassifier


# In[29]:


ml = MLPClassifier()


# In[30]:


ml.fit(X_train,y_train)


# In[31]:


ml.score(X_test,y_test)


# In[32]:


mlp = ml.predict(X_test)


# In[33]:


print(classification_report(mlp,y_test))


# In[34]:


def cal_zcr(path):
    X,sample_rate = librosa.load(path,res_type='kaiser_fast')
    zcr = np.mean(librosa.feature.zero_crossing_rate(X,frame_length=2048,hop_length=512,center=True))
    return zcr


# In[35]:


df['zcr'] = df['path'].apply(cal_zcr)


# In[36]:


df.sort_values(by='zcr')


# In[37]:


def cal_sc(path):
        X,sample_rate = librosa.load(path,res_type='kaiser_fast')
        sc = np.mean(librosa.feature.spectral_centroid(X,sr=sample_rate,n_fft=2048,
                                                       hop_length=512,freq=None,win_length=None,window='hann',
                                                       center=True,pad_mode='reflect'))
        return sc


# In[38]:


df['sc'] = df['path'].apply(cal_sc)


# In[39]:


def cal_rms(path):
    X,sample_rate = librosa.load(path,res_type='kaiser_fast')
    rms = np.mean(librosa.feature.rms(y=X,frame_length=2048,hop_length=512,center=True,pad_mode='reflect'))
    return rms


# In[40]:


df['rms'] = df['path'].apply(cal_rms)


# In[41]:


def cal_melspectrogram(path):
    X,sample_rate = librosa.load(path,res_type='kaiser_fast')
    _melspectrogram = np.mean(librosa.feature.melspectrogram(y=X,sr=sample_rate,n_fft=2048,hop_length=512,
                                                             win_length=None,window='hann',center=True, 
                                                             pad_mode='reflect',power=2.0))
    return _melspectrogram


# In[42]:


df['melspectrogram'] = df['path'].apply(cal_melspectrogram)


# In[43]:


sns.pairplot(df,hue='interference',vars=['mfcc_mean','zcr','sc','rms','melspectrogram'])


# In[44]:


df1 = df[['mfcc_mean','zcr','sc','rms','melspectrogram']]
sns.heatmap(df1.corr(),annot=True)


# In[45]:


df.head()


# In[46]:


df1.corr()


# In[47]:


X1 = df[['mfcc_mean','zcr','sc','rms','melspectrogram']]
y1 = df['interference']


# In[48]:


X1_train,X1_test,y1_train,y1_test = train_test_split(X1,y1,test_size=0.33,random_state=234)


# In[49]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()


# In[50]:


dtc.fit(X1_train,y1_train)


# In[51]:


dtc_pred = dtc.predict(X1_test)


# In[52]:


print(confusion_matrix(y1_test,dtc_pred))


# In[53]:


print(classification_report(y1_test,dtc_pred))


# # AVERAGE

# In[54]:


Xa = np.array(df['mfcc'].tolist())
ya = np.array(df['interference'].tolist())


# In[55]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(Xa)
scaled_feature = scalar.transform(Xa)


# In[56]:


Xaf = scaled_feature


# In[57]:


from sklearn.tree import DecisionTreeClassifier
lst = []
for i in range(100,150):
    Xa_train,Xa_test,ya_train,ya_test = train_test_split(Xaf,ya,test_size=0.3,random_state=i)
    clf = DecisionTreeClassifier()
    clf.fit(Xa_train,ya_train)
    
    lst.append(clf.score(Xa_test,ya_test))


# In[58]:


sum(lst)/len(lst)


# In[59]:


Xa_train,Xa_test,ya_train,ya_test = train_test_split(Xaf,ya,test_size=0.3,random_state=123)
clf = DecisionTreeClassifier()
clf.fit(Xa_train,ya_train)
print(clf.score(Xa_test,ya_test))


# In[60]:


ya_test-clf.predict(Xa_test)


# In[61]:


df['interference'].value_counts()


# In[62]:


df.head()


# In[63]:


writer = pd.ExcelWriter('interference_feature.xlsx',engine='xlsxwriter')
df.to_excel(writer,sheet_name='Sheet1')
writer.save()


# In[ ]:




