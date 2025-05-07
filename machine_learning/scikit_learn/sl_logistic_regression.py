from sklearn.linear_model import LogisticRegression

from machine_learning.scikit_learn.prepared_data import prepared_data

x_train, x_train_std, x_test, x_test_std, y_train, y_test = prepared_data()

# Логистическая регрессия
lg = LogisticRegression(C=100, solver='lbfgs')  # С - регуляризация.
lg.fit(x_train_std, y_train)
print(lg.predict(x_test))  # Покажет принадлежность к классу.
# Покажет % принадлежности к определенному классу.
for i, prob in enumerate(lg.predict_proba(x_test)):
    print(f"Sample {i + 1}: Class 0 → {prob[0]:.2%}, Class 1 → {prob[1]:.2%}, Class 2 → {prob[2]:.2%}")
