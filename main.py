# importing library
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import string

from feature_calculation.feature import Features

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

# reading dataset
df = pd.read_csv(r'C:\Users\akank\Dropbox\My PC (LAPTOP-NQ9H8NTJ)\Documents\Sem 8\Project\datasets\brown\brown.csv')
print(df.head())

##################################################################
#####                  1. DATA PREPARATION                   #####
##################################################################

plt.figure(figsize=(10, 3))
df.groupby([df['label']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Brown')
plt.xlabel('Number of texts')
plt.ylabel('Types of texts')
plt.show()

# dropping labels other than defined Fiction and Non-Fiction labels
index = df[(df.label == 'religion') | (df.label == 'lore') | (df.label == 'editorial') | 
(df.label == 'humor') | (df.label == 'belles_lettres')].index
df.drop(index, inplace = True)
print(df.head())
# changing labels to fiction_genre and non_fiction_genre
df["label"] = np.where(df["label"] == ('fiction' or 'mystery' or 'romance' or 'adventure' or 'science_fiction'), 
                        "fiction_genre", "non_fiction_genre")
print(df.head())

# dropping paragraphs with sentences less than 5 or 6 (to deal with data imbalance)
sent_count = df.groupby(['filename', 'para_id'], as_index =  False).size()
print(sent_count)

for i in range(len(sent_count)):
    size = sent_count['size'].iloc[i]
    if((size < 5) or (size > 6)):
        doc = sent_count['filename'].iloc[i]
        para = sent_count['para_id'].iloc[i]
        index = df[(df['filename'] == doc) & (df['para_id'] == para)].index
        df.drop(index, inplace = True)
print(df.head())

# storing labels of each paragraph
df_label = {k: f.groupby('para_id')['label'].apply(list).to_dict()
     for k, f in df.groupby('filename')}

class_label = []
for file_id, filename in df_label.items():
    for para_id, label in filename.items():
            class_label.append(label[0])
print(class_label)

pd.DataFrame(class_label, columns= ['label']).to_csv("labels.csv")

##################################################################
#####                  2. DATA PRE_PROCESSING                #####
##################################################################

# removing punctuation
def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct

df['tokenized_text_wo_punct'] =  df['tokenized_text'].apply(lambda x: remove_punctuation(x))
print(df.head())

# POS Tagging, Dependency and Constituency Parsing is done before Feature Extraction in next step

##################################################################
#####                  3. FEATURE EXTRACTION                 #####
##################################################################

# calling Features class from feature_extraction.py
features = Features()

# creating sentence list for each paragraph
df_dict = {k: f.groupby('para_id')['tokenized_text_wo_punct'].apply(list).to_dict()
     for k, f in df.groupby('filename')}

# getting all features for each paragraph
feature_list = []
for file_id, filename in df_dict.items():
    for para_id, sent_list in filename.items():
            feature_list.append(features.get_all_features(sent_list))

# saving the features in a new dataframe
features = pd.DataFrame(feature_list)
features.to_csv("extracted_features.csv")

# calling the dataframe which will be used for training
X = pd.read_csv(r'C:\Users\akank\Dropbox\My PC (LAPTOP-NQ9H8NTJ)\Documents\Sem 8\Project\code\extracted_features.csv')
y = pd.read_csv(r'C:\Users\akank\Dropbox\My PC (LAPTOP-NQ9H8NTJ)\Documents\Sem 8\Project\code\labels.csv')
X = X.fillna(0)
print(X.head())

##################################################################
#####                  4. SUPERVISED LEARNING                #####
##################################################################


# scaling data
scaler = StandardScaler()
scaled_data = scaler.fit(X)
scaled_data = scaler.transform(X)
print(scaled_data)

# defining classifier
clf = LogisticRegression(penalty = 'l1', solver= 'saga', max_iter = 1000)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=10, 
    scoring= "accuracy",
    n_jobs=2,
)
rfecv.fit(scaled_data, y.label)

# printing results
print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Ranking of features (Selected features are given rank 1): {rfecv.ranking_}")

# visualising the result
print(X.columns)

optimal_features = []
for i in range(len(X.columns)):
    if((rfecv.ranking_[i])==1):
        optimal_features.append(X.columns[i])
print(optimal_features)