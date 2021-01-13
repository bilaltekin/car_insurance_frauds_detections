import pandas as pd
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

dataFrame = pd.read_csv('datasource/Dataset.csv')

fraud = dataFrame['FraudFound_P']

print("Data info :{}".format(dataFrame.info()))

dataFrame.drop(['Month', 'WeekOfMonth', 'DayOfWeek', 'MonthClaimed', 'WeekOfMonthClaimed', 'DayOfWeekClaimed', 'FraudFound_P','Deductible'], axis=1,
               inplace=True)

dataFrame['fraud'] = fraud


print("dataFrame data :{}".format(dataFrame))
print("dataFrame shape :{}".format(dataFrame.columns))
print("dataFrame shape :{}".format(dataFrame.shape))



dataFrame.dropna(inplace=True)
dataFrame.drop(dataFrame[dataFrame['Age']==0].index,axis=0,inplace=True)

print("dataFrame shape :{}".format(dataFrame.shape))
dataFrame.to_csv('datasource/Dataset1.csv')

print(dataFrame.corr())


#Data encoder
#Dataları sayısal değere çevirme
label = LabelEncoder()
dataFrame['Make'] = label.fit_transform(dataFrame['Make'])
dataFrame['AccidentArea'] = label.fit_transform(dataFrame['AccidentArea'])
dataFrame['Sex'] = label.fit_transform(dataFrame['Sex'])
dataFrame['MaritalStatus'] = label.fit_transform(dataFrame['MaritalStatus'])
dataFrame['Fault'] = label.fit_transform(dataFrame['Fault'])
dataFrame['PolicyType'] = label.fit_transform(dataFrame['PolicyType'])
dataFrame['VehicleCategory'] = label.fit_transform(dataFrame['VehicleCategory'])
dataFrame['VehiclePrice'] = label.fit_transform(dataFrame['VehiclePrice'])
dataFrame['Days_Policy_Accident'] = label.fit_transform(dataFrame['Days_Policy_Accident'])
dataFrame['Days_Policy_Claim'] = label.fit_transform(dataFrame['Days_Policy_Claim'])
dataFrame['PastNumberOfClaims'] = label.fit_transform(dataFrame['PastNumberOfClaims'])
dataFrame['AgeOfVehicle'] = label.fit_transform(dataFrame['AgeOfVehicle'])
dataFrame['AgeOfPolicyHolder'] = label.fit_transform(dataFrame['AgeOfPolicyHolder'])
dataFrame['PoliceReportFiled'] = label.fit_transform(dataFrame['NumberOfCars'])
dataFrame['WitnessPresent'] = label.fit_transform(dataFrame['BasePolicy'])
dataFrame['AgentType'] = label.fit_transform(dataFrame['AgentType'])
dataFrame['NumberOfSuppliments'] = label.fit_transform(dataFrame['NumberOfSuppliments'])
dataFrame['AddressChange_Claim'] = label.fit_transform(dataFrame['AddressChange_Claim'])
dataFrame['NumberOfCars'] = label.fit_transform(dataFrame['NumberOfCars'])
dataFrame['BasePolicy'] = label.fit_transform(dataFrame['BasePolicy'])

print( dataFrame.corr())

sns.heatmap(dataFrame.corr(),annot=True,lw =2,cbar=False)


Data_train, Data_test, Target_train, Target_test = train_test_split(dataFrame.iloc[:,:-1], dataFrame.iloc[:,-1], test_size=0.65)

error = []
# Will take some time
from sklearn import metrics

for i in range(1, 40):
    neigh = KNeighborsClassifier(n_neighbors=i).fit(Data_train, Target_train)
    yhat = neigh.predict(Data_test)
    error.append(np.mean(Target_test != yhat))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

plt.show()



knn = KNeighborsClassifier(5)

knn.fit(Data_train, Target_train)
knn_predict = knn.predict(Data_test)
knn_acuracy = metrics.accuracy_score(knn_predict, Target_test)
print("KNN Score : {}".format(round( knn_acuracy,3)))
knn_cnf_matrix = confusion_matrix(knn_predict, Target_test)
print(knn_cnf_matrix)

knn_disp = plot_roc_curve(knn, Data_test, Target_test)
plt.show()

sns.heatmap(confusion_matrix(Target_test, knn_predict),annot=True,lw =2,cbar=False,fmt="d")
plt.ylabel("True Values")
plt.xlabel("Predicted Values")
plt.title("KNN CONFUSION MATRIX VISUALIZATION")
plt.show()



randomForestClassifier = RandomForestClassifier(n_estimators=200, n_jobs=-1)
randomForestClassifier.fit(Data_train, Target_train)
rf_predict = randomForestClassifier.predict(Data_test)
rf_acuracy = metrics.accuracy_score(rf_predict, Target_test)
print("Random Forest Score : {}".format(round( rf_acuracy,3)))
rf_cnf_matrix = confusion_matrix(rf_predict, Target_test)
print(rf_cnf_matrix)

rf_disp = plot_roc_curve(randomForestClassifier, Data_test, Target_test)
plt.show()

sns.heatmap(confusion_matrix(Target_test, rf_predict),annot=True,lw =2,cbar=False,fmt="d")
plt.ylabel("True Values")
plt.xlabel("Predicted Values")
plt.title("Random Forest CONFUSION MATRIX VISUALIZATION")
plt.show()


svmClasifier=svm.SVC()
svmClasifier.fit(Data_train, Target_train)
svm_predict = svmClasifier.predict(Data_test)
svm_acuracy = metrics.accuracy_score(svm_predict, Target_test)
print("SVM Score : {}".format(round( svm_acuracy,3)))
svm_cnf_matrix = confusion_matrix(svm_predict, Target_test)
print(svm_cnf_matrix)

svm_disp = plot_roc_curve(svmClasifier, Data_test, Target_test)
plt.show()


sns.heatmap(confusion_matrix(svm_predict, Target_test),annot=True,lw =2,cbar=False,fmt="d")
plt.ylabel("True Values")
plt.xlabel("Predicted Values")
plt.title("SVM Forest CONFUSION MATRIX VISUALIZATION")
plt.show()




print("Finished")


