import numpy as np
import pandas as pd
import sys
from sklearn.impute import SimpleImputer, KNNImputer
from src.logger import setup_logging  # Humne path update kiya hai

# Logger setup
logger = setup_logging('missing_value_techniques')

class MISSING_VALUE_TECHNIQUES:
    
    @staticmethod
    def mean_imputation(X_train, X_test):
        try:
            logger.info(f'Mean imputation for missing values started')
            X_tr = X_train.copy()
            X_te = X_test.copy()

            cols = X_tr.select_dtypes(exclude='object').columns
            imp = SimpleImputer(strategy='mean')

            X_tr[cols] = imp.fit_transform(X_tr[cols])
            X_te[cols] = imp.transform(X_te[cols])

            logger.info(f'Mean imputation completed successfully')
            return X_tr, X_te

        except Exception as e:
            _, _, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {str(e)}')
            return X_train, X_test

    @staticmethod
    def median_imputation(X_train, X_test):
        try:
            logger.info(f'Median imputation started')
            X_tr = X_train.copy()
            X_te = X_test.copy()

            cols = X_tr.select_dtypes(exclude='object').columns
            imp = SimpleImputer(strategy='median')

            X_tr[cols] = imp.fit_transform(X_tr[cols])
            X_te[cols] = imp.transform(X_te[cols])

            logger.info(f'Median imputation completed')
            return X_tr, X_te
        except Exception as e:
            _, _, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {str(e)}')
            return X_train, X_test

    @staticmethod
    def mode_imputation(X_train, X_test):
        try:
            logger.info(f'Mode imputation started')
            X_tr = X_train.copy()
            X_te = X_test.copy()

            imp = SimpleImputer(strategy='most_frequent')
            X_tr[:] = imp.fit_transform(X_tr)
            X_te[:] = imp.transform(X_te)

            logger.info(f'Mode imputation completed')
            return X_tr, X_te
        except Exception as e:
            _, _, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {str(e)}')
            return X_train, X_test

    @staticmethod
    def knn_imputation(X_train, X_test):
        try:
            logger.info(f'KNN imputation started')
            X_tr = X_train.copy()
            X_te = X_test.copy()

            cols = X_tr.select_dtypes(exclude='object').columns
            imp = KNNImputer(n_neighbors=5)

            X_tr[cols] = imp.fit_transform(X_tr[cols])
            X_te[cols] = imp.transform(X_te[cols])

            logger.info(f'KNN imputation completed')
            return X_tr, X_te
        except Exception as e:
            _, _, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {str(e)}')
            return X_train, X_test

    @staticmethod
    def forward_fill(X_train, X_test):
        try:
            logger.info(f'Forward fill started')
            X_tr = X_train.copy().ffill()
            X_te = X_test.copy().ffill()
            logger.info(f'Forward fill completed')
            return X_tr, X_te
        except Exception as e:
            _, _, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {str(e)}')
            return X_train, X_test

    @staticmethod
    def backward_fill(X_train, X_test):
        try:
            logger.info(f'Backward fill started')
            X_tr = X_train.copy().bfill()
            X_te = X_test.copy().bfill()
            logger.info(f'Backward fill completed')
            return X_tr, X_te
        except Exception as e:
            _, _, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {str(e)}')
            return X_train, X_test

    @staticmethod
    def random_sample_imputation(X_train, X_test):
        try:
            logger.info('Random Sample Imputation started')
            X_tr = X_train.copy()
            X_te = X_test.copy()

            for col in X_tr.columns:
                if X_tr[col].isnull().sum() > 0:
                    random_samples = X_tr[col].dropna()
                    if len(random_samples) == 0:
                        continue
                    X_tr[col] = X_tr[col].apply(lambda x: np.random.choice(random_samples) if pd.isnull(x) else x)
                    X_te[col] = X_te[col].apply(lambda x: np.random.choice(random_samples) if pd.isnull(x) else x)

            logger.info('Random Sample Imputation Completed')
            return X_tr, X_te
        except Exception as e:
            _, _, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {str(e)}')
            return X_train, X_test