from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from machine_learning.scikit_learn.prepared_data import get_wine_data
from machine_learning.scikit_learn.visualisation import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    df, x_train, x_train_std, x_test, x_test_std, y_train, y_test = get_wine_data()
    lda = LDA(n_components=2)
    lr_model = LogisticRegression(random_state=1, solver='lbfgs', C=0.01)

    # Получаем новое k-мерное пространство признаков с 2 линейными дискриминантами.
    x_train_std_lda = lda.fit_transform(x_train_std, y_train)
    x_test_std_lda = lda.transform(x_test_std)

    # Обучаем модель на основе новых признаков.
    lr_model.fit(x_train_std_lda, y_train)

    # Визуализация границ решений классификатора.
    x_comb = np.vstack([x_train_std_lda, x_test_std_lda])
    y_comb = np.hstack([y_train, y_test])
    plot_decision_regions(x_comb, y_comb, lr_model, range(105, 150))
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.legend(loc='lower right')
    plt.show()
    plt.show()



if __name__ == '__main__':
    main()