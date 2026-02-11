import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score


dt_mall = pd.read_csv("./mall_customers.csv")
dt_mall = dt_mall.drop("customer_id", axis=1)

le = LabelEncoder()
dt_mall['gender'] = le.fit_transform(dt_mall['gender'])

"""
for i in range(2,10):
    cluster = KMeans(n_clusters=i, random_state=44,n_init='auto')
    cluster.fit(dt_mall_scaled)
    print(i)
    print(cluster.inertia_)
    print(f'k={i} - ' + str(silhouette_score(dt_mall_scaled, cluster.predict(dt_mall_scaled))))
"""

scaler = StandardScaler()
dt_mall_scaled = scaler.fit_transform(dt_mall)

cluster = KMeans(n_clusters=9, random_state=44,n_init='auto')
cluster.fit(dt_mall_scaled)



novos_clientes = pd.DataFrame({
    'gender': [1, 0],        # Male=1, Female=0
    'age': [21, 40],
    'annual_income': [180, 70],
    'spending_score': [100, 30]
})

novos_clientes_scaled = scaler.fit_transform(novos_clientes)

previsao = cluster.predict(novos_clientes_scaled)

print(previsao)