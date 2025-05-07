from sklearn.svm import SVC
from machine_learning.scikit_learn.prepared_data import prepared_data

x_train, x_train_std, x_test, x_test_std, y_train, y_test = prepared_data()

svc = SVC(C=1.0, kernel='linear', random_state=1)
svc.fit(x_train_std, y_train)
print(svc.predict(x_test_std))

svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)

