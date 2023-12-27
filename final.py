
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns                 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


data = pd.read_csv('heart_failure_clinical_records_dataset.csv') 


# In[5]:


data.describe() 


# In[6]:


data.isnull().sum()  


# In[7]:


data.corr()['DEATH_EVENT'].sort_values()


# In[8]:


plt.figure(figsize=(12, 6))


# In[9]:


sns.heatmap(data.corr(), annot=True)


# In[10]:


X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=29)


# In[12]:


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


# In[13]:


# Train the classifier using cross-validation
cross_val_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5, scoring='accuracy')
mean_cross_val_accuracy = cross_val_scores.mean()


# In[14]:


# Train the classifier on the full training set
rf_classifier.fit(X_train, y_train)


# In[15]:


# Make predictions on the test set
predictions = rf_classifier.predict(X_test)


# In[17]:


# Evaluate and print the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy (Before Cross-Validation): {accuracy:.4f}")


# In[18]:


# Display Cross-Validation Mean Accuracy
print(f"Cross-Validation Mean Accuracy: {mean_cross_val_accuracy:.4f}")


# In[19]:


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)


# In[20]:


# AUC-ROC Curve
y_proba = rf_classifier.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)


# In[21]:


plt.figure(figsize=(8, 6))


# In[22]:


plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))


# In[23]:


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')


# In[24]:


plt.xlabel('False Positive Rate')


# In[25]:


plt.ylabel('True Positive Rate')


# In[26]:


plt.title('Receiver Operating Characteristic (ROC) Curve')


# In[29]:


plt.legend(loc='lower right')
plt.show()


# In[32]:


from sklearn.metrics import log_loss


# In[33]:


# Accuracy and Log Loss Curves during Training
train_acc = []
test_acc = []
train_logloss = []
test_logloss = []


# In[34]:


for n_estimators in range(1, 101):
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Training Accuracy and Log Loss
    train_preds = rf_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_preds)
    train_log_loss = log_loss(y_train, rf_classifier.predict_proba(X_train))

    # Test Accuracy and Log Loss
    test_preds = rf_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_log_loss = log_loss(y_test, rf_classifier.predict_proba(X_test))

    # Append values
    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)
    train_logloss.append(train_log_loss)
    test_logloss.append(test_log_loss)


# In[35]:


# Plotting Accuracy Curves
plt.figure(figsize=(12, 4))


# In[36]:


plt.subplot(1, 2, 1)


# In[37]:


plt.plot(range(1, 101), train_acc, label='Training Accuracy')


# In[38]:


plt.plot(range(1, 101), test_acc, label='Testing Accuracy')


# In[39]:


plt.xlabel('Number of Trees (n_estimators)')


# In[40]:


plt.ylabel('Accuracy')


# In[41]:


plt.legend()


# In[42]:


# Plotting Log Loss Curves
plt.subplot(1, 2, 2)


# In[43]:


plt.plot(range(1, 101), train_logloss, label='Training Log Loss')


# In[44]:


plt.plot(range(1, 101), test_logloss, label='Testing Log Loss')


# In[45]:


plt.xlabel('Number of Trees (n_estimators)')


# In[46]:


plt.ylabel('Log Loss')


# In[52]:


sns.heatmap(data.corr(), annot=True)


# In[53]:


accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy (Before Cross-Validation): {accuracy:.4f}")


# In[54]:


# Display Cross-Validation Mean Accuracy
print(f"Cross-Validation Mean Accuracy: {mean_cross_val_accuracy:.4f}")

