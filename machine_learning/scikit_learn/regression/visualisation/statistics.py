import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper


def anova_results(model: RegressionResultsWrapper) -> pd.DataFrame:
    """
    Метод возвращает dataframe с информацией о том, статистически значима модель на основе регрессии или нет.
    Значимость определяется через F-статистику (95%).
    """
    return pd.DataFrame(
        {
            # Регрессия, Остатки, Общее
            'df': [float(model.df_model), float(model.df_resid), float(model.nobs - 1)],
            'ss': [model.ess.round(2), model.ssr.round(2), (model.ess + model.ssr).round(2)],
            'ms (дисперсия)': [model.mse_model.round(1), model.mse_resid.round(2), '--'],
            'F': [model.fvalue.round(3), '--', '--'],
            'P': [f"{model.f_pvalue:.2e}", '--', '--']
        }
    )


def get_statistics_result(model: RegressionResultsWrapper) -> pd.DataFrame:
    """
    Метод возвращает dataframe с информацией о том, статистически значимы модель коэффициенты регрессии или нет.
    Значимость определяется через t-статистику (95%).
    """
    df = pd.DataFrame({
        'Coeff. SC': model.params.round(2),
        'Std. Err.': model.bse.round(2),
        't': model.tvalues.round(2),
        'p': model.pvalues.round(4)
    })
    conf_int = model.conf_int(alpha=0.05)
    df['DI (+/-)'] = round(abs(df['Coeff. SC'] - conf_int[0]), 4)
    return df


def get_accuracy_scores(model: RegressionResultsWrapper) -> dict[str, int | float]:
    """
    Метод возвращает показатели качества модели на основе регрессии: R2, R2 adj, Cond. No., RSD.
    """
    return {
        'R2': model.rsquared.round(2),
        'R2_adj': model.rsquared_adj.round(2),
        'N': model.nobs,
        'Cond. No.': f'{model.condition_number:.2e}',
        'RSD': np.sqrt(model.mse_resid).round(2)
    }
