import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats import chi2_contingency


class DataAnalyzer:
    def __init__(self, file_path=None, dataframe=None):
        """
        :param file_path: file_path to analyze
        :type file_path: str
        :param dataframe: dataframe to analyze
        :type dataframe: dataframe
        """
        self.file_path = file_path
        if not file_path:
            self.dataframe = dataframe
        else:
            self._load_data()
        self.no_of_rows = self.dataframe.shape[0]
        self.no_of_columns = self.dataframe.shape[1]
        self.target = self.dataframe.columns[-1]
        self.features = self.dataframe.columns[:-1]
        self.numerical_dataframe = None
        self.categorical_dataframe = None
        self.numerical_variables = []
        self.categorical_variables = []
        self.analysis = None

    def _load_data(self):
        """
        loads the dataframe from the given csv file.

        """
        self.dataframe = pd.read_csv(self.file_path, encoding='utf-8-sig')

    def _detect_column_type(self, row):
        """
        returns whether the column of the dataframe is numerical or categorical.

        :param row: row of the transformed dataframe
        :type row: series

        """
        if row['Columns'] in self.numerical_variables:
            return "Numerical"
        return "Categorical"

    def _set_column_type(self):
        """
        returns a list of data type of the columns of the dataframe.
        """
        self.analysis['column_type'] = self.analysis.apply(lambda row: self._detect_column_type(row), axis=1)

    def _set_numerical_categorical_columns(self):
        """
        Updates numerical and categorical details
        """

        numeric_var = [key for key in dict(self.dataframe.dtypes)
                       if dict(self.dataframe.dtypes)[key]
                       in ['float64', 'float32', 'int32', 'int64']]  # Numeric Variable

        cat_var = [key for key in dict(self.dataframe.dtypes)
                   if dict(self.dataframe.dtypes)[key] in ['object']]  # Categorical Variable
        self.numerical_dataframe = self.dataframe[numeric_var]
        self.categorical_dataframe = self.dataframe[cat_var]
        self.numerical_variables = numeric_var
        self.categorical_variables = cat_var

    def _get_mean_of_columns(self):
        """
        calculates mean of all columns in the dataframe.
        """

        return round(self.numerical_dataframe[list(self.numerical_dataframe.columns)].mean())

    def _get_min_of_columns(self):
        """
        calculates minimum of all columns in the dataframe.
        """

        return round(self.numerical_dataframe[list(self.numerical_dataframe.columns)].min(), 2)

    def _get_max_of_columns(self):
        """
        calculates maximum of all columns in the dataframe.
        """

        return round(self.numerical_dataframe[list(self.numerical_dataframe.columns)].max(), 2)

    def _get_var_of_columns(self):
        """
        calculates variance of all columns in the dataframe.
        """

        return round(self.numerical_dataframe[list(self.numerical_dataframe.columns)].var(), 2)

    def _get_std_of_columns(self):
        """
        calculates standard deviation of all columns in the dataframe.
        """

        return round(self.numerical_dataframe[list(self.numerical_dataframe.columns)].std(), 2)

    def _get_median_of_columns(self):
        """
        calculates median of all columns in the dataframe.
        """

        return round(self.numerical_dataframe[list(self.numerical_dataframe.columns)].median(), 2)

    def _get_mode_of_columns(self):
        """
        calculates mode of all columns in the dataframe.
        """
        return self.dataframe.mode().iloc[0]

    def _get_missing_values(self):
        """
        calculates missing values in all columns of the dataframe.
        """
        missing_values = self.dataframe[list(self.dataframe.columns)].isna().sum()
        missing_values_percentage = self.dataframe[list(self.dataframe.columns)].isna().sum().div(self.no_of_rows)
        return missing_values, missing_values_percentage

    @staticmethod
    def _detect_unique_vals_of_columns(col):
        """
        calculates unique values in all columns of the dataframe.
        :param col: column of the dataframe
        :type col: series
        """
        return col.nunique()

    def _get_unique_values(self):
        """
        gives a list of unique values of each column in the dataframe.
        """

        unique_values_list = self.dataframe.apply(lambda col: DataAnalyzer._detect_unique_vals_of_columns(col), axis=0)
        return unique_values_list

    @staticmethod
    def _detect_skewness(col):
        """
        calculates skewness type of all columns in the dataframe.
        :param col: column of the dataframe
        :type col: series
        """

        skewness = skew(col)
        return float("{:.2f}".format(skewness))

    def _get_skewness(self):
        """
        gives a list of skewness values of all columns in the dataframe.
        """
        skewness_values = self.numerical_dataframe.apply(lambda col: DataAnalyzer._detect_skewness(col), axis=0)
        return skewness_values

    @staticmethod
    def _detect_outliers(col, total_rows):
        """
        detects outliers of all columns in the dataframe.
        :param col: column of the dataframe
        :type col: series
        """

        quartile1, quartile3 = np.percentile(col, [25, 75])
        iqr = quartile3 - quartile1
        lower_bound = quartile1 - (1.5 * iqr)
        upper_bound = quartile3 + (1.5 * iqr)
        outliers = ((col < lower_bound) | (col > upper_bound)).sum()
        outlier_percentage = (outliers / total_rows) * 100
        return pd.Series((outliers, iqr, outlier_percentage))

    def _get_outliers(self):
        """
        gives a list of outliers of all columns in the dataframe.
        """

        series_res = self.numerical_dataframe.apply(lambda col: DataAnalyzer._detect_outliers(col, self.no_of_rows),
                                                    axis=0, result_type='expand')
        return series_res.iloc[0], series_res.iloc[1], series_res.iloc[2]

    def chi_square_test(self, col):
        alpha = 0.1
        dependent_cat = ''
        for i in range(len(self.categorical_variables)):
            myfield1 = col
            myfield2 = self.dataframe[self.categorical_variables[i]]
            mycrosstable = pd.crosstab(myfield1, myfield2)
            stat, p, dof, expected = chi2_contingency(mycrosstable)
            if p > (1 - alpha) and p != 1:
                dependent_cat = dependent_cat + " " + str(self.categorical_variables[i])

        return (dependent_cat)

    def _get_correlation_categoriacal(self):
        categorical_correlated = self.categorical_dataframe.apply(lambda col: DataAnalyzer.chi_square_test(self, col),
                                                                  axis=0)
        return categorical_correlated

    def correlation_matrix(self):
        return self.numerical_dataframe.corr(method='pearson')

    def numerical_correlation(self, col):
        correlated = ''
        for i in range(len(self.numerical_variables)):
            correlation = col.corr(self.dataframe[self.numerical_variables[i]])
            if (correlation > 0.8 or correlation < (-0.8)) and col.name != self.numerical_variables[i]:
                correlated = correlated + " " + str(self.numerical_variables[i])
        return correlated

    def _get_correlation_numerical(self):
        numerical_correlated = self.numerical_dataframe.apply(lambda col: DataAnalyzer.numerical_correlation(self, col),
                                                              axis=0)
        return numerical_correlated

    def _get_correlation_with_target(self):
        """
        computes information gain of each feature
        :return: series
        """
        item_values = []
        for feature in self.features:
            try:
                ig = 1  # info_gain(self.dataframe, feature, self.target) # implement information gain
                item_values.append(ig)
            except Exception as ex:
                pass
                # self._log.exception(f"Error calculating correlation for {feature}")
        # Return the information gain values for the features in the dataset
        return pd.Series(item_values, index=self.features)

    def _run_analysis(self):
        """ gets all statistics of all columns from loaded dataframe and creates a new dataframe"""
        self._set_numerical_categorical_columns()
        unique_values = self._get_unique_values()
        maximum_values = self._get_max_of_columns()
        mean_values = self._get_mean_of_columns()
        minimum_values = self._get_min_of_columns()
        variance_values = self._get_var_of_columns()
        std_values = self._get_std_of_columns()
        median_values = self._get_median_of_columns()
        categorical_correlation = self._get_correlation_categoriacal()
        numerical_correlation = self._get_correlation_numerical()
        skewness_values = self._get_skewness()
        outlier_values, iqr_values, outlier_percentage = self._get_outliers()
        mode_values = self._get_mode_of_columns()
        missing_values, missing_values_percentage = self._get_missing_values()

        # TODO: Re-enable Correlation when we have a faster calculation.
        # Currently the Information Gain based method takes 10 - 20 mins.
        # Once we have a better solution we'll update it.
        correlation = self._get_correlation_with_target()
        data_analysed = {
            'unique': unique_values,
            'max': maximum_values,
            'mean': mean_values,
            'min': minimum_values,
            'var': variance_values,
            'std_deviation': std_values,
            'median': median_values,
            'skew': skewness_values,
            'mode': mode_values,
            'iqr_value': iqr_values,
            'no_of_outliers': outlier_values,
            'outlier_percentage': outlier_percentage,
            'no_of_null_values': missing_values,
            'null_values_percentage': missing_values_percentage,
            'categorical_variable_correlation': categorical_correlation,
            'numerical_variable_correlation': numerical_correlation
            # 'correlation': correlation
        }
        self.analysis = pd.DataFrame(data_analysed).fillna(-1)
        self.analysis['total_instances'] = self.no_of_rows

    def get_analysis(self):
        """
        gets all statistics of all columns from loaded dataframe and creates a new dataframe
        :return dataframe
        """
        self._run_analysis()
        self.analysis.reset_index(level=0, inplace=True)
        self.analysis.rename(columns={'index': 'columns'}, inplace=True)
        return self.analysis


#da = DataAnalyzer(file_path="./test.csv")
#res = da.get_analysis()
