import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split

uri = "https://gist.githubusercontent.com/guilhermesilveira/b9dd8e4b62b9e22ebcb9c8e89c271de4/raw/c69ec4b708fba03c445397b6a361db4345c83d7a/tracking.csv"

cvs = pd.read_csv(uri)

x = cvs[["inicial", "palestras", "contato",	"patrocinio"]]
y = cvs["comprou"]

X_train, X_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.2, random_state=53)

model = sk.svm.LinearSVC()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

acurrency = sk.metrics.accuracy_score(y_test, prediction)
print(acurrency*100)