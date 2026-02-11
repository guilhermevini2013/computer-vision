import pandas as pd
import sklearn.preprocessing as sk
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

best_k = 4

def test_kmeans(df):
    for i in range(2, 15):
        model = KMeans(n_clusters=i, random_state=69)
        model.fit(df)
        print(f" k = {i} - {silhouette_score(df, model.predict(df))}  -  {model.inertia_}")

def train_kmeans(df):
    model = KMeans(n_clusters=best_k, random_state=69)
    model.fit(df)
    return model

df = pd.read_csv("customer.csv")
df = df.drop("customer_id", axis = 1)

"""
 normalization data
"""
df_normalized = DataFrame(data=sk.MinMaxScaler().fit_transform(df), columns=df.columns)

#test_kmeans(df_normalized)
model = train_kmeans(df_normalized)

result = model.predict(df_normalized)

df_with_kmeans = df_normalized.copy()
df_with_kmeans["k"] = result

cluster_profile = df_with_kmeans.groupby("k").mean().round(2)
print(cluster_profile)




