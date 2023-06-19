import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn.model_selection import learning_curve, GridSearchCV, RandomizedSearchCV # RandomizedCV quand ya bcp de combinaisons
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA



data = pd.read_excel('dataset.xlsx')
df = data.copy()

## Elimination des NaN et variables inutiles
missing_rate = df.isna().sum()/df.shape[0]

blood_columns = list(df.columns[(missing_rate < 0.9) & (missing_rate >0.88)])
viral_columns = list(df.columns[(missing_rate < 0.80) & (missing_rate > 0.75)])
key_columns = ['Patient age quantile','SARS-Cov-2 exam result']

df = df[key_columns + blood_columns + viral_columns]

## TrainTest - Nettoyage - Encodage
trainset, testset = train_test_split(df,test_size = 0.2, random_state = 0)

def encodage(df):
    code = {'positive' : 1,
            'negative' : 0,
            'detected' : 1,
            'not_detected' : 0}
    
    for col in df.select_dtypes('object'):
        df[col] = df[col].map(code)
    
    return df

def feature_engineering(df):
    df['est malade'] = df[viral_columns].sum(axis=1) >= 1
    df = df.drop(viral_columns , axis =1)
    return df

def imputation(df):
    # df['is na'] = (df['Parainfluenza 3'].isna()) | (df['Leukocytes'].isna())
    # df = df.fillna(-999)  
    df = df.dropna(axis = 0)
    return df

def preprocessing(df):
    df = encodage(df)
    df = feature_engineering(df)
    df = imputation(df)
    
    X = df.drop('SARS-Cov-2 exam result',axis = 1)
    y = df['SARS-Cov-2 exam result']
    
    print(y.value_counts())
    return X,y

X_train,y_train = preprocessing(trainset)
X_test,y_test = preprocessing(testset)

print('X_train shape : ',X_train.shape)
print('X_test shape : ',X_test.shape)

## Modelisation (Tester sur decisionTreeClassifier solo puis RandomTReeClassifier et ajouter une pipeline avec de la sélection et polynomialFeatures en last)
model_1 = RandomForestClassifier(random_state= 0)


preprocessor = make_pipeline(PolynomialFeatures(2, include_bias= False),
                      SelectKBest(f_classif, k=10))             # k meilleurs variables en test de ANOVA
 
RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state = 0))
AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state = 0))
SVM = make_pipeline(preprocessor,StandardScaler(), SVC(random_state= 0))
KNN = make_pipeline(preprocessor, StandardScaler(),KNeighborsClassifier())
# N.B : Arbres de décision non pas besoin d'avoir des données normalisées

dict_of_models = {'RandomForest' : RandomForest, 
                  'AdaBoost' : AdaBoost, 
                  'SVM' : SVM,
                  'KNN' : KNN }

## Procédure d'évaluation
def evaluation(model,name):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred)) # --> voir les performances du modèle
    
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv = 4, # --> tracer les courbes de train et de val pour voir si le modele est en overfitting (train_score >>) ou underfitting (val_score >>)
                                               scoring = 'f1',                  # counter l'ovrfitting : fournir plus de données ou selection de variables
                                               train_sizes = np.linspace(0.1,1,10))
    plt.figure(figsize= (12,8))
    plt.plot(N,train_score.mean(axis = 1) , label = 'train_score')
    plt.plot(N,val_score.mean(axis = 1) , label = 'validation_score')
    plt.title(name)
    plt.legend()
    plt.show()
    
for name,model in dict_of_models.items():
    print(name)
    evaluation(model,name)
    
# --> SVM ou AdaBoost les plus prometteurs

    # Variables les plus importantes pour l'arbre de décision (pour la sélection pour éviter l'overfitting)
#pd.DataFrame(model.feature_importances_, index = X_train.columns).plot.bar(figsize = (12,8))


# Optimisation des hyper-paramètres SVC
hyper_params = {'svc__gamma' : [1e-3,1e-4],
                'svc__C' : [1, 10, 100, 1000],
                'pipeline__polynomialfeatures__degree' : [2, 3, 4],
                'pipeline__selectkbest__k' : range(40,60)}

grid = RandomizedSearchCV(SVM, hyper_params,cv = 4, scoring = 'recall', n_iter = 40)
grid.fit(X_train,y_train)

print(grid.best_params_)

y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

evaluation(grid.best_estimator_,'SVC')

# Pecision Recall Curve (pour une classification binaire)
precision, recall ,threshold = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))

plt.plot(threshold, precision[:-1], label = 'precision')
plt.plot(threshold, recall[:-1], label = 'recall')
plt.xlabel('Threshold')
plt.legend()

def model_final(model,X, threshold = 0):
    return model.decision_function(X) > threshold

y_pred = model_final(grid.best_estimator_, X_test, threshold = -1)

print('Score f1 : ',f1_score(y_test,y_pred))
print('Recall : ',recall_score(y_test,y_pred))

# Choisir le threshold pour avoir un bon compromis entre le recall (test qqun de négatif alors qu'il est réellement positif)
# et la precision generale de la prédicition


# Optimisation des hyper-paramètres AdaBoost
hyper_params = {'adaboostclassifier__n_estimators' : range(10,100),
                'pipeline__polynomialfeatures__degree' : [2, 3, 4],
                'pipeline__selectkbest__k' : range(4,100)}

grid = RandomizedSearchCV(AdaBoost, hyper_params,cv = 4, scoring = 'recall', n_iter = 40)
grid.fit(X_train,y_train)

print(grid.best_params_)
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

evaluation(grid.best_estimator_,'AdaBoost')

# Pecision Recall Curve (pour une classification binaire)
precision, recall ,threshold = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))

plt.plot(threshold, precision[:-1], label = 'precision')
plt.plot(threshold, recall[:-1], label = 'recall')
plt.xlabel('Threshold')
plt.legend()

def model_final(model,X, threshold = 0):
    return model.decision_function(X) > threshold

y_pred = model_final(grid.best_estimator_, X_test, threshold = -0.3)

print('Score f1 : ',f1_score(y_test,y_pred))
print('Recall : ',recall_score(y_test,y_pred))
