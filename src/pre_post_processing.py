import os
os.chdir(r"C:\\Users\\mirko\\Desktop\\cross-validation")
from src.functions.imports import *

# !!! WORKING IN PROGRESS !!!

df = pd.read_csv("data\\\Utrecht2019_dataset_plos_one_0222717_s001.csv")

# dataset characteristics
df.info()
df.describe()      # there are some NA values

# fix the classes
df.dtypes
df["SVZ_contact"].unique()

df["alive0_deceased1"] = df["alive0_deceased1"].astype("category")
df["biopsy0_resection1"] = df["biopsy0_resection1"].astype("category")
df["KPS_greater_or_equal70"] = df["KPS_greater_or_equal70"].astype("category")
df["SVZ_contact"] = df["KPS_greater_or_equal70"].astype("category")
df["adjuvant_treatment"] = df["adjuvant_treatment"].astype("category")

# correlation between variables
df.corr()

# missing values
df.isna().sum() / len(df)
# how Random Forest natively handles NA values in X
# it natively tests whether it is better to classify observations with NA values into the right or left branch during training, based on the impurity or information criterion
# it does not impute values; it selects the option that leads to better classification
# I consider removing missing values for descriptive analyses
df_nona = df.dropna()
# check through dichotomization whether missing values depend on other variables

# examine relationships between variables using descriptive statistics
# quantitative x - binary y
y = df.iloc[:,-1]

plt.figure()
sns.boxplot(data = df, x = y, y = df.loc[:,"age"])
plt.show()

# contingency table: binary x - binary y
pd.crosstab(
    y,
    df["KPS_greater_or_equal70"]
)

# Mahalanobis distance to detect outliers IN THE QUANTITATIVE VARIABLES
X = df_nona.loc[:,["age","survival_days"]].to_numpy()

n = len(X)
one = np.ones((n,1))
x_avg = np.array(X.T@one/n)

X_tilde = X - one @ x_avg.T
S = X_tilde.T @ X_tilde / n

mahal = np.sum((X_tilde @ np.linalg.inv(**S**)) * X_tilde, axis=1)

# if observations follow a multivariate normal distribution => mahal ~ chi^2 p
p = X.shape[1]
alpha = 0.01
from scipy.stats import chi2
threshold = chi2.ppf(1 - alpha, df=p)

outliers = np.where(np.array(mahal) > threshold)[0]
X_outliers =X[outliers, :]


# feature selection using impurity variance
model = RandomForestClassifier(
    n_estimators= 100 ,
    criterion = "gini",
    random_state = 11
)

# data splitting
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

# relative importance
importance_relative = importance / importance.max()
importance_relative = importance_relative.sort_values()

plt.figure()
importance_relative.plot.barh()
plt.show()
