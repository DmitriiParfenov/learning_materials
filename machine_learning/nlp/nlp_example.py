from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from machine_learning.nlp.utils import (
    get_df_from_file,
    TextCleaner,
    simple_tokenizer,
    porter_tokenizer,
)

if __name__ == '__main__':
    cleaner = TextCleaner()
    stop_words = stopwords.words('english')

    # Загрузили все данные в dataframe.
    df = get_df_from_file(basedir=r'C:\Users\GTA\Desktop\aclImdb', permute=True)
    df['review'] = df['review'].apply(cleaner.clean_text)
    x_train = df.loc[:3999, 'review'] .values
    x_test = df.loc[3999:, 'review'].values
    y_train = df.loc[:3999, 'sentiment'] .values
    y_test = df.loc[3999:, 'sentiment'].values

    # Оптимизация параметров.
    # 1 - оптимизируем tf-idf, 2 - tf. Сделали так, чтобы уменьшить комбинаторное поле.
    param_grid = [
        {
            'tfidfvectorizer__ngram_range': [(1, 1)],
            'tfidfvectorizer__stop_words': [None],
            'tfidfvectorizer__tokenizer': [simple_tokenizer, porter_tokenizer, word_tokenize],
            'tfidfvectorizer__token_pattern': [None],
            'logisticregression__penalty': ['l2'],
            'logisticregression__C': [0.1, 1.0, 10.0]
        },
        {
            'tfidfvectorizer__ngram_range': [(1, 1)],
            'tfidfvectorizer__stop_words': [stop_words, None],
            'tfidfvectorizer__use_idf': [False],
            'tfidfvectorizer__smooth_idf': [False],
            'tfidfvectorizer__norm': [None],
            'logisticregression__penalty': ['l2'],
            'logisticregression__C': [0.1, 1.0, 10.0]
        },
    ]

    lg_pipeline = make_pipeline(
        # Объявляем преобразователь текста в числовой вектор.
        TfidfVectorizer(
            strip_accents=None,
            lowercase=False,
            preprocessor=None  # Устанавливаем None для оптимизации
        ),
        LogisticRegression(random_state=1, solver='liblinear')
    )

    gs = GridSearchCV(
        estimator=lg_pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=-1,
        cv=5,
        verbose=2,
        refit=True
    )
    gs.fit(x_train, y_train)

    # Результаты.
    print(f'Лучшие параметры: {gs.best_params_}')
    print(f'Лучшая точность при оптимизации: {gs.best_score_:.2f}')
    best_estimator = gs.best_estimator_
    print(f'Точность на тестовых данных: {best_estimator.score(x_test, y_test):.2f}')

    new_x = ['Terrible! No story, awful CGI. Don’t watch.']
    print(best_estimator.predict(new_x))