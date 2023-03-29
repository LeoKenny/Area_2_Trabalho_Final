# %%
import os
from re import search
import pandas as pd
import numpy as np
import scipy.io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

import seaborn as sns
import matplotlib.pyplot as plt

SAMPLE_FREQUENCY = 48_000

if not os.path.exists("images"):
    os.mkdir("images")

path = '/home/kenny/Area_2_Trabalho_Final/48k_DE/'

size_sample = 2048
df_original = pd.DataFrame()

for entry in os.scandir(path):
    if entry.is_file():
        mat = scipy.io.loadmat(path+entry.name)
        for i in mat.keys():
            if search('DE',i):
                key = i

        raw_data = [item for sublist in mat[key] for item in sublist]

        samples = [raw_data[i*size_sample:(i+1)*size_sample] for i in range(int(len(raw_data)/2048))]
        df_original_raw = pd.DataFrame(zip(samples),columns=['Samples'])
        df_original_raw['Fault'] = entry.name.split('.')[0].split('_')[0]
        df_original = pd.concat([df_original,df_original_raw])

df = df_original.copy(deep=True)
df = df.reset_index(drop=True)

import plotly.express as px

sample = 100
y = df.iloc[sample]['Samples']
x = np.array(range(len(y)))*1/SAMPLE_FREQUENCY

fig = px.line(x=x, y=y, title=f'Sample {sample}',
              labels={'y':'Acceleration [g]',
                      'x':'Time [s]'
              })

fig.write_image(f"images/Amostra_{sample}.png")

df['Max'] = df.apply(lambda x: np.array(x[0]).max(),axis=1)
df['Min'] = df.apply(lambda x: np.array(x[0]).min(),axis=1)
df['Mean'] = df.apply(lambda x: np.array(x[0]).mean(),axis=1)
df['RMS'] = df.apply(lambda x: np.sqrt(np.mean(np.array(x[0])**2)),axis=1)
df['Var'] = df.apply(lambda x: np.var(np.array(x[0])),axis=1)
df['Crest'] = df.apply(lambda x: (np.array(x[0]).max())/(np.sqrt(np.mean(np.array(x[0])**2))),axis=1)
df['Form'] = df.apply(lambda x: np.sqrt(np.mean(np.array(x[0])**2))/np.abs(np.array(x[0])).mean(),axis=1)
df['Impu'] = df.apply(lambda x: np.array(x[0]).max()/np.sqrt(np.mean(np.abs(np.array(x[0])))),axis=1)
df['Clear'] = df.apply(lambda x: np.array(x[0]).max()/np.mean(np.sqrt(np.abs(np.array(x[0])))),axis=1)

df = df.drop(columns=['Samples'])
df_full = df.copy(deep=True)

fig = plt.figure(1,figsize=(8,6))
df = df_full.drop(columns=['Fault'])
corr_ticks = df.columns
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot= True,
            xticklabels=corr_ticks, yticklabels=corr_ticks,
            cmap = "Blues", cbar = False)
plt.title('Matriz de Correlação')
fig.savefig("images/base_corr.png")

plt.clf()
df = df_full.drop(columns=['Fault','Impu','Min'])
corr_ticks = df.columns
corr_matrix = df.corr()
plot2 = sns.heatmap(corr_matrix, annot= True,
            xticklabels=corr_ticks, yticklabels=corr_ticks,
            cmap = "Blues", cbar = False)
plt.title('Matriz de Correlação')
fig = plot2.get_figure()
fig.savefig("images/base_corr_drop.png")

# df = df_full.drop(columns=['Fault'])
scaler = StandardScaler()
data_time_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(data_time_scaled, columns=df.columns)

X_train, X_test, y_train, y_test = train_test_split(df_scaled,df_full['Fault'], test_size = 0.3, stratify = df_full['Fault'], random_state = 1)

# %%
parameters = [{"C":[1, 10, 45, 47,49, 50, 51, 55, 100, 300, 500, 1000],
               'kernel':["linear"]},
              {"C":[1, 10, 45, 47,49, 50, 51, 55, 100, 300, 500, 1000],
               'gamma':['auto',0.001, 0.01, 0.05, 0.1, 0.5, 1, 5],
               'kernel':["rbf"]},
              {"C":[1, 10, 45, 47,49, 50, 51, 55, 100, 300, 500, 1000],
               'gamma':['auto',0.001, 0.01, 0.1, 1],
               'kernel':["poly"]},
              {"C":[1, 10, 45, 47,49, 50, 51, 55, 100, 300, 500, 1000],
               'gamma':['auto',0.001, 0.01, 0.05, 0.1, 0.5, 1, 5],
               'kernel':["sigmoid"]},
             ]

tuned_svm_clf = GridSearchCV(SVC(),parameters,n_jobs = -1, cv= 10, verbose=1)
tuned_svm_clf.fit(X_train, y_train)

train_predictions_best = tuned_svm_clf.best_estimator_.predict(X_train)
test_predictions_best = tuned_svm_clf.best_estimator_.predict(X_test)

train_confu_matrix_best = confusion_matrix(y_train, train_predictions_best)
test_confu_matrix_best = confusion_matrix(y_test, test_predictions_best)

fault_type = df_full.Fault.unique()

fig = plt.figure(1,figsize=(8,6))
sns.heatmap(train_confu_matrix_best, annot= True,fmt = "d",
            xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
plt.title('Matrix de Confusão da base de treino')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
fig.savefig('images/output_SVM_training.png')

fig.clf()
sns.heatmap(test_confu_matrix_best, annot = True,fmt = "d",
            xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
plt.title('Matrix de Confusão da base de teste')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')

fig.savefig('images/output_SVM_test.png')

# Classification report (test set)
class_report = classification_report(y_pred = test_predictions_best, y_true = y_test)

with open("results/SVM.txt") as f:
    f.write(f"Best Params: {tuned_svm_clf.best_params_}")
    f.write(f"Best Estimator: {tuned_svm_clf.best_estimator_}")
    f.write("\n")
    f.write(class_report)

# %%
parameters = {"multi_class":['multinomial','auto'],
              "solver":["newton-cg", "sag", "saga", "lbfgs", "liblinear"],
              "max_iter":[10000],
              "penalty":[None,"l1","l2","elasticnet"],
              "tol":[0.1,0.01,0.001]}

tuned_logis_clf = GridSearchCV(LogisticRegression(),parameters,n_jobs = -1, cv= 10, verbose=1)
tuned_logis_clf.fit(X_train, y_train)

print(tuned_logis_clf.best_params_)
print(tuned_logis_clf.best_estimator_)

train_predictions_best = tuned_logis_clf.best_estimator_.predict(X_train)
test_predictions_best = tuned_logis_clf.best_estimator_.predict(X_test)

train_confu_matrix_best = confusion_matrix(y_train, train_predictions_best)
test_confu_matrix_best = confusion_matrix(y_test, test_predictions_best)

fig = plt.figure(1,figsize=(8,6))
sns.heatmap(train_confu_matrix_best, annot= True,fmt = "d",
xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
plt.title('Matrix de Confusão da base de treino')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
fig.savefig('images/output_log_training.png')

fig.clf()
sns.heatmap(test_confu_matrix_best, annot = True,fmt = "d",
xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
plt.title('Matrix de Confusão da base de teste')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')

fig.savefig('images/output_log_test.png')

class_report = classification_report(y_pred = test_predictions_best, y_true = y_test)

with open("results/Logis.txt") as f:
    f.write(f"Best Params: {tuned_logis_clf.best_params_}")
    f.write(f"Best Estimator: {tuned_logis_clf.best_estimator_}")
    f.write("\n")
    f.write(class_report)

# %%
from sklearn.preprocessing import OneHotEncoder

# Create an instance of the OneHotEncoder class
encoder = OneHotEncoder(categories=[df_full['Fault'].unique()])

# Reshape the vector into a 2D array
vector_train = y_train
vector_train = [[label] for label in vector_train]
vector_test = y_test
vector_test = [[label] for label in vector_test]

# Fit and transform the vector using the encoder
one_hot_encoded_train = encoder.fit_transform(vector_train)
one_hot_encoded_test = encoder.fit_transform(vector_test)

# Convert the sparse matrix to a dense array
y_train_transformed = one_hot_encoded_train.toarray()
y_test_transformed = one_hot_encoded_test.toarray()

# %%
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, mode='auto', restore_best_weights=True)
baseline_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=30, mode='auto', baseline=0.7, restore_best_weights=True)

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

from scikeras.wrappers import KerasClassifier

# Define the model
def create_model(optimizer='adam', lr=0.01, hiddenA=10, hiddenB=10, loss='categorical_crossentropy'):
    if optimizer == 'sgd':
        optimizer = optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=lr)
    elif optimizer == 'adagrad':
        optimizer = optimizers.Adagrad(learning_rate=lr)
    elif optimizer == 'adamax':
        optimizer = optimizers.Adamax(learning_rate=lr)
    else:
        optimizer = optimizers.Adam(learning_rate=lr)

    model = Sequential([
        Dense(hiddenA, activation='relu', input_shape=(len(X_train.columns),)),
        Dense(hiddenB, activation='relu'),
        Dense(len(df_full['Fault'].unique()), activation='softmax')
    ])
    model.compile(loss=loss, optimizer=optimizer, metrics=['categorical_crossentropy','categorical_accuracy'])
    return model

# Define the hyperparameters to tune
param_grid = {'optimizer': ['sgd', 'rmsprop', 'adagrad', 'adamax', 'adam'],
              'lr': [0.1,0.01, 0.001],
              'hiddenA': [5,10,20,30],
              'hiddenB': [5,10,20,30],
              'loss': ['categorical_crossentropy'],
              'batch_size': [32]}

# Create the grid search object
clf = KerasClassifier(model=create_model,optimizer='adam', lr=0.01, hiddenA=10, hiddenB=10, loss='categorical_crossentropy', callbacks=[early_stop,baseline_stop])
grid_search = GridSearchCV(clf,
                           param_grid=param_grid, cv=3, n_jobs=1,
                           verbose=1)

#%%
# Fit the grid search to the data
grid_search.fit(X_train, y_train_transformed, epochs=200, validation_data=(X_test, y_test_transformed), verbose=1, callbacks=[early_stop,baseline_stop])

# Print the best hyperparameters and their corresponding score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
# %%
train_predictions_best = encoder.inverse_transform(clf.predict(X_train))
test_predictions_best = encoder.inverse_transform(clf.predict(X_test))

train_confu_matrix_best = confusion_matrix(y_train, train_predictions_best)
test_confu_matrix_best = confusion_matrix(y_test, test_predictions_best)

fig = plt.figure(1,figsize=(8,6))
sns.heatmap(train_confu_matrix_best, annot= True,fmt = "d",
xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
plt.title('Matrix de Confusão da base de treino')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
fig.savefig('images/output_mlp_training.png')

fig.clf()
sns.heatmap(test_confu_matrix_best, annot = True,fmt = "d",
xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
plt.title('Matrix de Confusão da base de teste')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')

fig.savefig('images/output_mlp_test.png')

class_report = classification_report(y_pred = test_predictions_best, y_true = y_test)
print(class_report)

with open("results/MLP.txt") as f:
    f.write(f"Best Params: {grid_search.best_params_}")
    f.write(f"Best Estimator: {grid_search.score_}")
    f.write("\n")
    f.write(class_report)
