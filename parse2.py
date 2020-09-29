from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

f = open("./data/reviews_sentiment.csv", encoding='utf-8')
line = f.readline()
line = f.readline()
X = []
Y = []
while line != '\n' and line != '':
    values = line.split(';')
    reg = []
    reg.append(int(values[2]))
    
    if values[3] == "negative":
        reg.append(-1)
    elif values[3] == "positive":
        reg.append(1)
    else:
        reg.append(0)
        
    if values[4] == "negative":
        reg.append(-1)
    elif values[4] == "positive":
        reg.append(1)
    else:
        reg.append(0)

    reg.append(float(values[6]))
    Y.append(int(values[5]))

    X.append(reg)
    line = f.readline()

X = np.array(X)
#scaler = StandardScaler()
#scaled_data = scaler.fit_transform(X)
#imp = SimpleImputer(strategy="mean")
#imputed_data = imp.fit_transform(scaled_data)
#print(scaled_data)

outF = open("reviews_sentiment_fixed.csv", "w")
idx = 0
for reg in X:
    for value in reg:
        outF.write(str(value))
        outF.write(";")
    outF.write(str(Y[idx]))
    outF.write("\n")
    idx += 1
outF.close()
