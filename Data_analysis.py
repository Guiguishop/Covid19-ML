import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

data = pd.read_excel('dataset.xlsx')
df = data.copy()

## Analyse de la forme
print("(echantillons, variables) : ",df.shape)
print(df.dtypes.value_counts())

plt.figure(figsize=(20,10))
sns.heatmap(df.isna(),cbar = False)  # voir les NaN dans tout le dataset
plt.show()

data_count_nan = (df.isna().sum()/df.shape[0]).sort_values(ascending = True)


    # Elimination des colonnes où ya plus de 90% de NaN
df = df[df.columns[df.isna().sum()/df.shape[0] <0.9]]  

plt.figure(figsize=(20,10))
sns.heatmap(df.isna(),cbar = False)  # voir les NaN dans tout le dataset
plt.show()

df = df.drop('Patient ID',axis = 1)  # colonne useless


## Analyse du fond
    # Visualisation de la target
print(df['SARS-Cov-2 exam result'].value_counts(normalize = True))

    # Distribution de chaque variable float
cnt =1
plt.figure(figsize=(20,14))
for col in df.select_dtypes('float'):
    plt.subplot(4,4,cnt)
    cnt+=1
    sns.distplot(df[col])
plt.show()
    
    # Distribution de l'age
sns.displot(df['Patient age quantile'])
plt.show()
    
    # Distribution de chaque variable category
for col in df.select_dtypes('object'):      # Affiche les valeurs dans les variables
    print(f'{col :-<50}{df[col].unique()}') 

cnt =1
plt.figure(figsize=(15,8))
for col in df.select_dtypes('object'):      # Compte les valeurs pour chaques variables
    plt.subplot(5,4,cnt)
    cnt +=1
    df[col].value_counts().plot.pie()
    plt.title(col)
plt.show()
    
### Relation target / variables
    
    # création de sous-ensembles postifs et négatifs
positive_df = df[df['SARS-Cov-2 exam result'] == 'positive']
negative_df = df[df['SARS-Cov-2 exam result'] == 'negative']


    # création des ensembles Blood et viral
missing_rate = df.isna().sum()/df.shape[0]

blood_columns = df.columns[(missing_rate < 0.9) & (missing_rate >0.88)]
viral_columns = df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)]

    # Relation Target/ Blood (variable continue)
cnt =1
plt.figure(figsize=(20,15))
for col in blood_columns:
    plt.subplot(4,4,cnt)
    cnt+=1
    sns.distplot(positive_df[col], label='positive')
    sns.distplot(negative_df[col], label='negative')
    plt.legend()
plt.show()
    
    # Relation Target/age
sns.countplot(x ='Patient age quantile', hue ='SARS-Cov-2 exam result',data = df  )

    # Relation Target/ Viral (variable categorielle)
cnt =1
plt.figure(figsize = (23,15))
for col in viral_columns:
    plt.subplot(5,4,cnt)
    cnt+=1
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'],df[col]),annot = True, fmt = 'd')
plt.show()


### Relation variables / variables (corrélation)

    # Relation Blood/Blood
# sns.pairplot(df[blood_columns])
# plt.show()

sns.heatmap(df[blood_columns].corr())
plt.show()

sns.clustermap(df[blood_columns].corr())
plt.show()

    # Relation Blood/Age
plt.figure(figsize=(20,15))
for col in blood_columns:
    plt.figure()
    sns.lmplot(x = 'Patient age quantile', y=col, hue = 'SARS-Cov-2 exam result',data = df)
    plt.show()
    
#print(df.corr().loc['Patient age quantile'].sort_values()) 
# --> age n'influe pas sur les taux sanguins

    # Relation entre Influenza et rapid test
print(pd.crosstab(df['Influenza A'],df['Influenza A, rapid test']))
print(pd.crosstab(df['Influenza B'],df['Influenza B, rapid test']))

        # --> Influenza rapid test très peu fiable
    
    # Relation Viral/sanguin

# création d'une nouvelle variable "est malade"
df['est malade'] = np.sum(df[viral_columns[:-2]] == 'detected',axis =1)>=1 # on enleve les 2 derniers colonnes pas fiables
malade_df = df[df['est malade'] == True]
non_malade_df = df[df['est malade'] == False]

cnt =1
plt.figure(figsize=(20,15))
for col in blood_columns:
    plt.subplot(4,4,cnt)
    cnt+=1
    sns.distplot(malade_df[col], label='malade')
    sns.distplot(non_malade_df[col], label='non malade')
    plt.legend()
plt.show()

    # Relation Blood/ soins

def hospitalisation(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'surveillance'
    elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'soins semi-intensives'
    elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'soins intensifs'
    else:
        return 'inconnu'

df['status'] = df.apply(hospitalisation,axis = 1)

cnt =1
plt.figure(figsize=(20,15))
for col in blood_columns:
    plt.subplot(4,4,cnt)
    cnt+=1
    for cat in df['status'].unique():
        sns.distplot(df[df['status']==cat][col], label = cat)
    plt.legend()
plt.show()
    
### Impact des NaN sur Target
df1 = df[viral_columns[:-2]]
df1['covid'] = df['SARS-Cov-2 exam result']
print(df1.dropna()['covid'].value_counts(normalize = True)) # on vire tous les NaN du dataset viral puis on regarde le % de négatif ou positif au covid

df2 = df[blood_columns]
df2['covid'] = df['SARS-Cov-2 exam result']
print(df2.dropna()['covid'].value_counts(normalize = True))



### Tests d'hypothèses
 # Test de Student (les proportions des classes doivent être équilibrées)
balanced_neg = negative_df.sample(positive_df.shape[0]) # sous- échantillonne

def t_test(col):
    alpha = 0.02
    stat, p = ttest_ind(balanced_neg[col].dropna(), positive_df[col].dropna())
    if p < alpha:
        return 'H0 Rejetée'
    else :
        return 0

for col in blood_columns:
    print(f'{col :-<50} {t_test(col)}')



