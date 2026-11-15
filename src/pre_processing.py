import os
os.chdir(r"C:\\Users\\mirko\\Desktop\\cross-validation-research")
from src.functions.imports import * 

df = pd.read_csv("data\\Utrecht2019_dataset_plos_one_0222717_s001.csv")

# caratteristiche dataset  
df.info()
df.describe()      # vi sono alcuni valori NA 

# sistemo le classi 
df.dtypes
df["SVZ_contact"].unique()

df["alive0_deceased1"] = df["alive0_deceased1"].astype("category")
df["biopsy0_resection1"] = df["biopsy0_resection1"].astype("category")
df["KPS_greater_or_equal70"] = df["KPS_greater_or_equal70"].astype("category")
df["SVZ_contact"] = df["KPS_greater_or_equal70"].astype("category")
df["adjuvant_treatment"] = df["adjuvant_treatment"].astype("category")

# correlazione tra variabili
df.corr()

# valori mancanti
df.isna().sum() / len(df)
# come random forest gestisce nativamente valori NA in X 
# nativamente va a testare se è meglio andare a classificare osservazioni con NA su ramo dx o ramo sx nel training, in base al criterio di impurità o informazione
# non sta andando ad imputare valore, sta scegliendo quello che porta a classificazione migliore
# considero eliminazione valori mancancti per analisi descrittive 
df_nona = df.dropna()
# verificare con dicotomizzazione valori na verificando se na dipendono da altre variabili

# verifico con stat descrittive relazione variabili 
# x quantitativa - y binaria
y = df.iloc[:,-1]

plt.figure()
sns.boxplot(data = df, x = y, y = df.loc[:,"age"])
plt.show()

# tabella contingenza x binaria - y binaria 
pd.crosstab(
    y,
    df["KPS_greater_or_equal70"]
)

# distanza mahalanobis per verificati valori anomali NELLE VARIABILI QUANTITATIVE 
X = df_nona.loc[:,["age","survival_days"]].to_numpy()

n = len(X)
one = np.ones((n,1))
x_avg = np.array(X.T@one/n)

X_tilde = X - one @ x_avg.T
S = X_tilde.T @ X_tilde / n 

mahal = np.sum((X_tilde @ np.linalg.inv(S)) * X_tilde, axis=1)

# se osservazioni seguono distribuzione normale multivariata => mahal ~ chi^2 p 
p = X.shape[1]
alpha = 0.01
from scipy.stats import chi2
threshold = chi2.ppf(1 - alpha, df=p)

outliers = np.where(np.array(mahal) > threshold)[0]
X_outliers = X[outliers, :]



# feauture selection con variance impurity
model = RandomForestClassifier(
    n_estimators= 100 ,
    criterion = "gini",
    random_state = 11
)

# split dei dati 
X = df.iloc[:,:-1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

# mean impurity
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

# importanza relativa 
importance_relative = importance / importance.max()
importance_relative = importance_relative.sort_values()

plt.figure()
importance_relative.plot.barh()
plt.show()
