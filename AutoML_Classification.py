import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit
import matplotlib as mpl
import shap
from scipy.stats import chi2_contingency
# from pycaret.regression import *
from pycaret.classification import *
# from pycaret.regression import *
# from pycaret.clustering import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -*- coding: utf-8-sig -*-

# 폰트 설정
# font_path = '/Users/yerin/Library/Fonts/NanumBarunGothic.ttf'
# font_name = plt.matplotlib.font_manager.FontProperties(fname=font_path).get_name()
# plt.rcParams['font.family'] = font_name
# # font = fm.FontProperties(fname=fontpath, size = 24)

def cramers_v(x, y):
    """Cramér's V를 계산합니다."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

class Classification:
    def __init__(self, data_path, target_col):
        import pycaret.classification as clf_module
        self.clf_module = clf_module
        self.data_path = data_path
        self.data = None
        self.target_col = target_col
        self.setup_data = None
        self.tuned_models = []
        self.best_models = []
        self.best_model = None
        self.blended_model = None
        self.best_final_model = None
        self.models_dict = {}  # 모델 이름과 모델 객체를 매핑하는 딕셔너리
        
        # self.optimization_completed = False  # 상태 추적 변수 추가


    # 데이터 불러오기
    def load_data(self):
        """데이터를 불러옵니다."""
        if self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path, encoding='utf-8-sig')
        # 추가적인 데이터 형식에 대한 처리 (예: .xlsx)는 필요에 따라 확장 가능

    def load_data(self, dataframe=None):
        """데이터를 불러옵니다. 파일 경로 또는 데이터프레임을 사용합니다."""
        if dataframe is not None:
            self.data = dataframe
        elif self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path, encoding='utf-8-sig')
        elif self.data_path.endswith('.xlsx'):
            self.data = pd.read_excel(self.data_path)
        
    def load_uploaded_file(uploaded_file):
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)

    # 데이터 탐색
    def explore_data(self):
        """데이터의 형태와 컬럼을 확인합니다."""
        print(f'데이터 행 수: {self.data.shape[0]}')
        print(f'데이터 열 수: {self.data.shape[1]}')
        print(f'데이터 컬럼: {self.data.columns}')
        data_description = self.data.describe()
        return data_description
        
    # 타입 별 변수 구분
    def feature_type(self):
        """데이터의 변수 타입을 구분합니다."""
        categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
        numerical_features = self.data.select_dtypes(exclude=['object']).columns.tolist()
        print(f'Categorical Features: {categorical_features}')
        print(f'Numerical Features: {numerical_features}')
        return categorical_features, numerical_features

    # 수치형 변수 시각화
    def visualize_numerical_distribution(self):
        """수치형 변수의 분포를 시각화합니다."""
        
        # 수치형 변수 추출
        num_cols = self.data.select_dtypes(exclude=['object']).columns.tolist()
        
        # 그래프 스타일 및 팔레트 설정
        sns.set_style("whitegrid")
        sns.set_palette("pastel")

        # 그래프의 행과 열 수 계산 (2개의 열)
        rows = len(num_cols) // 2
        if len(num_cols) % 2:
            rows += 1

        # 그래프 크기와 간격 조정
        fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
        for i, column in enumerate(num_cols):
            ax = axes[i // 2, i % 2] if rows > 1 else axes[i % 2]
            sns.histplot(self.data[column], kde=True, bins=30, ax=ax)
            ax.set_title(f'Distribution of {column}', fontsize=15)
            ax.set_ylabel('Frequency')

        # 불필요한 빈 서브플롯 제거
        if len(num_cols) % 2:
            if rows > 1:
                axes[-1, -1].axis('off')
            else:
                axes[-1].axis('off')

        # # 수치형 변수의 분포 시각화
        # plt.figure(figsize=(14, rows * 5))
        # for i, column in enumerate(num_cols, 1):
        #     plt.subplot(rows, 2, i)
        #     sns.histplot(self.data[column], kde=True, bins=30)
        #     plt.title(f'Distribution of {column}', fontsize=15)
        #     plt.ylabel('Frequency')
        #     plt.tight_layout()

        # plt.show()
        # 간격 조정
        plt.tight_layout()
        return plt.gcf()

    # 범주형 변수 시각화
    def visualize_categorical_distribution(self):
        """범주형 변수의 분포를 시각화합니다."""
        # 범주형 변수 추출
        cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        # 범주형 변수가 없으면 None 반환
        if not cat_cols:
            print('범주형 변수가 없습니다.')
            return None

        # 그래프 그리기 설정
        rows = len(cat_cols)

        # 시각화 설정
        sns.set_style("whitegrid")
        palette = sns.color_palette("pastel")

        # 범주형 변수의 분포와 빈도를 시각화
        fig, axes = plt.subplots(rows, 1, figsize=(10, 5 * rows), squeeze=False)
        for i, column in enumerate(cat_cols):
            sns.countplot(y=self.data[column], ax=axes[i, 0], palette=palette, order=self.data[column].value_counts().index)
            axes[i, 0].set_title(f'Distribution of {column}')
            axes[i, 0].set_xlabel('Count')

        plt.tight_layout()
        return fig
        

    # 결측치 시각화
    def visualize_missing_distribution(self):
        """결측치 분포를 시각화합니다."""
        
        # 결측치 비율 계산
        missing_ratio = self.data.isnull().mean() * 100
        missing_count = self.data.isnull().sum()

        # 결측치 건수 및 비율에 대한 데이터프레임
        missing_df = pd.DataFrame({'Missing Count': missing_count, 'Missing Ratio (%)': missing_ratio})

        # 결측치 비율을 시각화
        plt.figure(figsize=(16, 8))
        sns.barplot(x=missing_ratio.index, y=missing_ratio, palette=sns.color_palette("pastel"))
        plt.axhline(30, color='red', linestyle='--')  # 30% 초과를 나타내는 빨간색 점선 추가
        plt.xticks(rotation=45)
        plt.title('Percentage of Missing Values by Columns')
        plt.ylabel('Missing Value Percentage (%)')
        plt.tight_layout()

        # plt.show()
        return missing_df, plt.gcf()

    # 결측치 처리
    def handle_and_visualize_missing(self, threshold=30):
        """결측치 처리 후 데이터를 확인하고 시각화합니다."""
        
        # 1. 결측치 비율 계산
        missing_ratio = self.data.isnull().mean() * 100
        
        # 2. 결측치 비율이 threshold(기본값: 30%)가 넘는 변수들 추출
        columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

        # 3. 해당 변수들 제거
        self.data.drop(columns=columns_to_drop, inplace=True)

        # 4. 결측치 비율 재확인
        missing_ratio_cleaned = self.data.isnull().mean() * 100
        missing_count_cleaned = self.data.isnull().sum()

        # 결측치 건수 및 비율에 대한 데이터프레임
        missing_df_cleaned = pd.DataFrame({'Missing Count': missing_count_cleaned, 'Missing Ratio (%)': missing_ratio_cleaned})

        # 시각화 그래프
        plt.figure(figsize=(16, 8))
        sns.barplot(x=missing_ratio_cleaned.index, y=missing_ratio_cleaned, palette=sns.color_palette("pastel"))
        plt.ylim(0, 100) # y축의 범위를 0부터 100까지로 설정
        plt.xticks(rotation=45)
        plt.title('Percentage of Missing Values by Columns (After Cleaning)')
        plt.ylabel('Missing Value Percentage (%)')
        plt.tight_layout()

        # plt.show()
        return missing_df_cleaned, plt.gcf()

    # 수치형 상관관계 분석
    def numerical_correlation(self):
        """수치형 변수들 간의 상관관계를 분석합니다."""
        corr_matrix = self.data.corr()

        # 상단 삼각형 마스크
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # 파스텔 톤 색상 팔레트
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        plt.figure(figsize=(20, 12))
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    fmt=".2f", 
                    cmap=cmap, 
                    mask=mask,
                    linewidths=0.5,
                    cbar_kws={"shrink": .8})
        plt.title("Numerical Features Correlation Matrix", fontsize=16)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        # plt.show()
        return plt.gcf()

    # def numerical_correlation(self):
    #     """수치형 변수들 간의 상관관계를 분석합니다."""
    #     corr_matrix = self.data.corr()

    #     # 파스텔 톤 색상 팔레트
    #     cmap = sns.diverging_palette(230, 20, as_cmap=True)

    #     plt.figure(figsize=(14, 12))
    #     sns.heatmap(corr_matrix, 
    #                 annot=True, 
    #                 fmt=".2f", 
    #                 cmap=cmap, 
    #                 linewidths=0.5,
    #                 cbar_kws={"shrink": .8})
    #     plt.title("Numerical Features Correlation Matrix", fontsize=16)
    #     plt.xticks(fontsize=10)
    #     plt.yticks(fontsize=10)
    #     plt.show()

    # def categorical_correlation(self):
    #     """범주형 변수들 간의 상관관계를 분석합니다."""
    #     columns = self.data.select_dtypes(include=['object', 'category']).columns
    #     corr_matrix = pd.DataFrame(index=columns, columns=columns)

    #     for i in columns:
    #         for j in columns:
    #             corr_matrix.loc[i, j] = cramers_v(self.data[i], self.data[j])

    #     corr_matrix = corr_matrix.astype(float)

    #     # 파스텔 톤 색상 팔레트
    #     cmap = sns.diverging_palette(230, 20, as_cmap=True)

    #     plt.figure(figsize=(12, 10))
    #     sns.heatmap(corr_matrix, 
    #                 annot=True, 
    #                 fmt=".2f", 
    #                 cmap=cmap)
    #     plt.title("Categorical Features Correlation Matrix", fontsize=16)
    #     plt.xticks(fontsize=10)
    #     plt.yticks(fontsize=10)
    #     plt.show()
    
    def categorical_correlation(self):
        """범주형 변수들 간의 상관관계를 분석합니다."""
        try:
            columns = self.data.select_dtypes(include=['object', 'category']).columns
            corr_matrix = pd.DataFrame(index=columns, columns=columns)

            for i in columns:
                for j in columns:
                    corr_matrix.loc[i, j] = cramers_v(self.data[i], self.data[j])

            corr_matrix = corr_matrix.astype(float)

            # 파스텔 톤 색상 팔레트
            cmap = sns.diverging_palette(230, 20, as_cmap=True)

            plt.figure(figsize=(20, 12))
            sns.heatmap(corr_matrix, 
                        annot=True, 
                        fmt=".2f", 
                        cmap=cmap)
            plt.title("Categorical Features Correlation Matrix", fontsize=16)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            # plt.show()
        
        except: 
            print('범주형 변수가 없습니다.')

        return plt.gcf()
        
    # # 결측치 확인
    # def check_missing_values(self):
    #     """데이터 내의 결측치를 확인합니다."""
    #     missing_data = self.data.isnull().sum()
    #     print(missing_data[missing_data > 0])
        
    # # 결측치 처리 (평균값으로 대체하는 경우)
    # def handle_missing_values(self, strategy="mean"):
    #     """결측치를 처리합니다."""
    #     if strategy == "mean":
    #         self.data.fillna(self.data.mean(), inplace=True)
    #     # Other strategies can be added if needed
        
    # 옵션 설정
    def setup(self, fix_imbalance=False, fix_imbalance_method = 'SMOTE', remove_outliers=False, remove_multicollinearity=True,
                      multicollinearity_threshold=0.9, train_size=0.7, fold_strategy='stratifiedkfold',
                      fold=5, profile=True, session_id=786, verbose=False):
        """옵션을 설정하고 데이터를 준비합니다."""
        # from pycaret.classification import setup
        self.setup_data = setup(data=self.data, target=self.target_col,
                                fix_imbalance=fix_imbalance,
                                fix_imbalance_method=fix_imbalance_method,
                                remove_outliers=remove_outliers,
                                remove_multicollinearity=remove_multicollinearity,
                                multicollinearity_threshold=multicollinearity_threshold,
                                train_size=train_size,
                                fold_strategy=fold_strategy,
                                fold=fold,
                                profile=profile,
                                session_id=session_id, 
                                # silent=silent, 
                                verbose=verbose)
        
        result = pull()
        result = result.iloc[:-5, :]

        self.feature_names = self.data.columns.drop(self.target_col) # 타겟 열을 제외한 모든 열 이름 저장

        return self.setup_data, result
    
    # # 모델 비교 및 생성/최적화
    # def compare_and_optimize_models(self, n_select=3, n_iter=50):
    #     """모델을 비교하고 상위 n_select개의 모델을 생성한 후 최적화합니다."""
        
    #     best = compare_models(n_select=n_select,
    #                           include=['lr','knn', 'dt', 'svm', 'mlp', 'rf', 'ada', 'gbc', 'et', 'xgboost', 'lightgbm', 'catboost'])
    #     best_df = pull()
        
    #     self.best_models = []
    #     self.tuned_models = []
        
    #     for i in range(n_select):
    #         best_model_name = str(best_df.index[i])
    #         model = create_model(best_model_name)
    #         tuned_model = tune_model(model, n_iter=n_iter)
            
    #         self.best_models.append(model)
    #         self.tuned_models.append(tuned_model)
        
    #     return self.tuned_models

    # 모델 비교 및 생성/최적화
    def compare_and_optimize_models(self, n_select=3, n_iter=50):
        best = compare_models(n_select=n_select,
                              include=['lr','knn', 'dt', 'svm', 'mlp', 'rf', 'ada', 'gbc', 'et', 'xgboost', 'lightgbm', 'catboost'])
        best_df = pull()

        self.best_models = []
        self.tuned_models = []
        optimization_results = []  # 결과를 저장할 리스트

        for i in range(n_select):
            best_model_name = str(best_df.index[i])
            model = create_model(best_model_name)
            tuned_model = tune_model(model, n_iter=n_iter)
            
            self.best_models.append(model)
            self.tuned_models.append(tuned_model)
            self.models_dict[f"모델 {i+1}"] = tuned_model  # 각 단계의 결과를 딕셔너리에 추가
            # self.models_dict[f"모델 {i+1}"] = {"model": tuned_model, "features": self.feature_names}

            # 각 단계의 결과를 리스트에 추가
            result_df = pull()  # 각 모델의 최적화 결과를 가져옵니다.
            optimization_results.append(result_df)

        # self.optimization_completed = True  # 최적화가 완료되었음을 상태 추적 변수에 저장

        return self.models_dict, self.tuned_models, best_df, optimization_results
    
    def create_ensemble_model(self, optimize='Recall'):
        """최적화된 모델들을 사용하여 앙상블 모델을 생성합니다."""
        # if not hasattr(self, 'tuned_models') or not self.tuned_models:
        #     raise AttributeError("최적화된 모델이 정의되지 않았습니다. 먼저 모델 비교 및 최적화 설정 단계를 실행하세요.")
        
        # self.blended_model = blend_models(estimator_list=self.tuned_models, optimize=optimize)

        if not self.tuned_models:
            raise AttributeError("최적화된 모델이 정의되지 않았습니다. 먼저 모델 비교 및 최적화 설정 단계를 실행하세요.")
        
        self.blended_model = blend_models(estimator_list=self.tuned_models, optimize=optimize)
        result_df = pull()

        result_df = pull()

        return self.blended_model, result_df

    def select_best_model(self, optimize='F1'):
        """최고 성능의 모델을 선정합니다."""
        self.best_final_model = automl(optimize=optimize)
        self.models_dict["최고 성능 모델"] = self.best_final_model 
        print(self.best_final_model)
        return self.best_final_model
    
    def save_model(self, model_name, save_directory):
        """모델을 지정된 디렉토리에 저장합니다."""
        save_path = os.path.join(save_directory, model_name)
        save_model(self.best_final_model, save_path)
        
    # 모델 시각화
    def visualize_model(self, model, plot_type):
        """
        선택된 모델의 성능을 시각화합니다.
        plot_type: ‘auc’, ‘threshold’, ‘pr’, ‘error’, ‘class_report’,
        ‘boundary’, ‘rfe’, ‘learning’, ‘manifold’, ‘calibration’, ‘vc’,
        ‘dimension’, ‘feature’, ‘feature_all’, ‘parameter’, ‘lift’, ‘gain’,
        ‘tree’, ‘ks’, ‘confusion_matrix’
        """
        plot_result = plot_model(model, plot=plot_type, display_format='streamlit', plot_kwargs={"fontsize":40})
        return plot_result
        
        
    # 모델 해석
    def interpret_model(self, model, plot_type, **kwargs):
        """모델을 해석하고 SHAP 값을 시각화합니다."""
        interpret_result = interpret_model(model, plot=plot_type, **kwargs)
        return interpret_result

    @classmethod
    def predict_data(cls, model, data):
        """모델을 사용하여 데이터를 예측합니다."""
        predictions = predict_model(model, data=data, raw_score = True)
        return predictions
        

# class Regression:
#     def __init__(self, data_path, target_col):
#         import pycaret.regression as reg_module
#         self.reg_module = reg_module
#         self.data_path = data_path
#         self.data = None
#         self.target_col = target_col
#         self.setup_data = None
#         self.tuned_models = []
#         self.best_models = []
#         self.best_model = None
#         self.blended_model = None
#         self.best_final_model = None
#         self.models_dict = {}  # 모델 이름과 모델 객체를 매핑하는 딕셔너리

#     # 데이터 불러오기
#     def load_data(self):
#         """데이터를 불러옵니다."""
#         if self.data_path.endswith('.csv'):
#             self.data = pd.read_csv(self.data_path, encoding='utf-8-sig')
#         elif self.data_path.endswith('.xlsx'):
#             self.data = pd.read_excel(self.data_path)
#         # 추가적인 데이터 형식에 대한 처리 (예: .xlsx)는 필요에 따라 확장 가능

#     def load_data(self, dataframe=None):
#         """데이터를 불러옵니다. 파일 경로 또는 데이터프레임을 사용합니다."""
#         if dataframe is not None:
#             self.data = dataframe
#         elif self.data_path.endswith('.csv'):
#             self.data = pd.read_csv(self.data_path, encoding='utf-8-sig')
#         elif self.data_path.endswith('.xlsx'):
#             self.data = pd.read_excel(self.data_path)
        
#     def load_uploaded_file(uploaded_file):
#         if uploaded_file.name.endswith('.csv'):
#             return pd.read_csv(uploaded_file)
#         elif uploaded_file.name.endswith('.xlsx'):
#             return pd.read_excel(uploaded_file)

#     # 데이터 탐색
#     def explore_data(self):
#         """데이터의 형태와 컬럼을 확인합니다."""
#         print(f'데이터 행 수: {self.data.shape[0]}')
#         print(f'데이터 열 수: {self.data.shape[1]}')
#         print(f'데이터 컬럼: {self.data.columns}')
#         data_description = self.data.describe()
#         return data_description
        
#     # 타입 별 변수 구분
#     def feature_type(self):
#         """데이터의 변수 타입을 구분합니다."""
#         categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
#         numerical_features = self.data.select_dtypes(exclude=['object']).columns.tolist()
#         print(f'Categorical Features: {categorical_features}')
#         print(f'Numerical Features: {numerical_features}')
#         return categorical_features, numerical_features

#     # 수치형 변수 시각화
#     def visualize_numerical_distribution(self):
#         """수치형 변수의 분포를 시각화합니다."""
        
#         # 수치형 변수 추출
#         num_cols = self.data.select_dtypes(exclude=['object']).columns.tolist()
        
#         # 그래프 스타일 및 팔레트 설정
#         sns.set_style("whitegrid")
#         sns.set_palette("pastel")

#         # 그래프의 행과 열 수 계산 (2개의 열)
#         rows = len(num_cols) // 2
#         if len(num_cols) % 2:
#             rows += 1

#         # 그래프 크기와 간격 조정
#         fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
#         for i, column in enumerate(num_cols):
#             ax = axes[i // 2, i % 2] if rows > 1 else axes[i % 2]
#             sns.histplot(self.data[column], kde=True, bins=30, ax=ax)
#             ax.set_title(f'Distribution of {column}', fontsize=15)
#             ax.set_ylabel('Frequency')

#         # 불필요한 빈 서브플롯 제거
#         if len(num_cols) % 2:
#             if rows > 1:
#                 axes[-1, -1].axis('off')
#             else:
#                 axes[-1].axis('off')

#         # # 수치형 변수의 분포 시각화
#         # plt.figure(figsize=(14, rows * 5))
#         # for i, column in enumerate(num_cols, 1):
#         #     plt.subplot(rows, 2, i)
#         #     sns.histplot(self.data[column], kde=True, bins=30)
#         #     plt.title(f'Distribution of {column}', fontsize=15)
#         #     plt.ylabel('Frequency')
#         #     plt.tight_layout()

#         # plt.show()
#         # 간격 조정
#         plt.tight_layout()
#         return plt.gcf()

#     # 범주형 변수 시각화
#     def visualize_categorical_distribution(self):
#         """범주형 변수의 분포를 시각화합니다."""
#         # 범주형 변수 추출
#         cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
#         # 범주형 변수가 없으면 None 반환
#         if not cat_cols:
#             print('범주형 변수가 없습니다.')
#             return None

#         # 그래프 그리기 설정
#         rows = len(cat_cols)

#         # 시각화 설정
#         sns.set_style("whitegrid")
#         palette = sns.color_palette("pastel")

#         # 범주형 변수의 분포와 빈도를 시각화
#         fig, axes = plt.subplots(rows, 1, figsize=(10, 5 * rows), squeeze=False)
#         for i, column in enumerate(cat_cols):
#             sns.countplot(y=self.data[column], ax=axes[i, 0], palette=palette, order=self.data[column].value_counts().index)
#             axes[i, 0].set_title(f'Distribution of {column}')
#             axes[i, 0].set_xlabel('Count')

#         plt.tight_layout()
#         return fig
        

#     # 결측치 시각화
#     def visualize_missing_distribution(self):
#         """결측치 분포를 시각화합니다."""
        
#         # 결측치 비율 계산
#         missing_ratio = self.data.isnull().mean() * 100
#         missing_count = self.data.isnull().sum()

#         # 결측치 건수 및 비율에 대한 데이터프레임
#         missing_df = pd.DataFrame({'Missing Count': missing_count, 'Missing Ratio (%)': missing_ratio})

#         # 결측치 비율을 시각화
#         plt.figure(figsize=(16, 8))
#         sns.barplot(x=missing_ratio.index, y=missing_ratio, palette=sns.color_palette("pastel"))
#         plt.axhline(30, color='red', linestyle='--')  # 30% 초과를 나타내는 빨간색 점선 추가
#         plt.xticks(rotation=45)
#         plt.title('Percentage of Missing Values by Columns')
#         plt.ylabel('Missing Value Percentage (%)')
#         plt.tight_layout()

#         # plt.show()
#         return missing_df, plt.gcf()

#     # 결측치 처리
#     def handle_and_visualize_missing(self, threshold=30):
#         """결측치 처리 후 데이터를 확인하고 시각화합니다."""
        
#         # 1. 결측치 비율 계산
#         missing_ratio = self.data.isnull().mean() * 100
        
#         # 2. 결측치 비율이 threshold(기본값: 30%)가 넘는 변수들 추출
#         columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

#         # 3. 해당 변수들 제거
#         self.data.drop(columns=columns_to_drop, inplace=True)

#         # 4. 결측치 비율 재확인
#         missing_ratio_cleaned = self.data.isnull().mean() * 100
#         missing_count_cleaned = self.data.isnull().sum()

#         # 결측치 건수 및 비율에 대한 데이터프레임
#         missing_df_cleaned = pd.DataFrame({'Missing Count': missing_count_cleaned, 'Missing Ratio (%)': missing_ratio_cleaned})

#         # 시각화 그래프
#         plt.figure(figsize=(16, 8))
#         sns.barplot(x=missing_ratio_cleaned.index, y=missing_ratio_cleaned, palette=sns.color_palette("pastel"))
#         plt.ylim(0, 100) # y축의 범위를 0부터 100까지로 설정
#         plt.xticks(rotation=45)
#         plt.title('Percentage of Missing Values by Columns (After Cleaning)')
#         plt.ylabel('Missing Value Percentage (%)')
#         plt.tight_layout()

#         # plt.show()
#         return missing_df_cleaned, plt.gcf()

#     # 수치형 상관관계 분석
#     def numerical_correlation(self):
#         """수치형 변수들 간의 상관관계를 분석합니다."""
#         corr_matrix = self.data.corr()

#         # 상단 삼각형 마스크
#         mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

#         # 파스텔 톤 색상 팔레트
#         cmap = sns.diverging_palette(230, 20, as_cmap=True)

#         plt.figure(figsize=(20, 12))
#         sns.heatmap(corr_matrix, 
#                     annot=True, 
#                     fmt=".2f", 
#                     cmap=cmap, 
#                     mask=mask,
#                     linewidths=0.5,
#                     cbar_kws={"shrink": .8})
#         plt.title("Numerical Features Correlation Matrix", fontsize=16)
#         plt.xticks(fontsize=10)
#         plt.yticks(fontsize=10)
#         # plt.show()
#         return plt.gcf()

#     # def numerical_correlation(self):
#     #     """수치형 변수들 간의 상관관계를 분석합니다."""
#     #     corr_matrix = self.data.corr()

#     #     # 파스텔 톤 색상 팔레트
#     #     cmap = sns.diverging_palette(230, 20, as_cmap=True)

#     #     plt.figure(figsize=(14, 12))
#     #     sns.heatmap(corr_matrix, 
#     #                 annot=True, 
#     #                 fmt=".2f", 
#     #                 cmap=cmap, 
#     #                 linewidths=0.5,
#     #                 cbar_kws={"shrink": .8})
#     #     plt.title("Numerical Features Correlation Matrix", fontsize=16)
#     #     plt.xticks(fontsize=10)
#     #     plt.yticks(fontsize=10)
#     #     plt.show()

#     # def categorical_correlation(self):
#     #     """범주형 변수들 간의 상관관계를 분석합니다."""
#     #     columns = self.data.select_dtypes(include=['object', 'category']).columns
#     #     corr_matrix = pd.DataFrame(index=columns, columns=columns)

#     #     for i in columns:
#     #         for j in columns:
#     #             corr_matrix.loc[i, j] = cramers_v(self.data[i], self.data[j])

#     #     corr_matrix = corr_matrix.astype(float)

#     #     # 파스텔 톤 색상 팔레트
#     #     cmap = sns.diverging_palette(230, 20, as_cmap=True)

#     #     plt.figure(figsize=(12, 10))
#     #     sns.heatmap(corr_matrix, 
#     #                 annot=True, 
#     #                 fmt=".2f", 
#     #                 cmap=cmap)
#     #     plt.title("Categorical Features Correlation Matrix", fontsize=16)
#     #     plt.xticks(fontsize=10)
#     #     plt.yticks(fontsize=10)
#     #     plt.show()
    
#     def categorical_correlation(self):
#         """범주형 변수들 간의 상관관계를 분석합니다."""
#         try:
#             columns = self.data.select_dtypes(include=['object', 'category']).columns
#             corr_matrix = pd.DataFrame(index=columns, columns=columns)

#             for i in columns:
#                 for j in columns:
#                     corr_matrix.loc[i, j] = cramers_v(self.data[i], self.data[j])

#             corr_matrix = corr_matrix.astype(float)

#             # 파스텔 톤 색상 팔레트
#             cmap = sns.diverging_palette(230, 20, as_cmap=True)

#             plt.figure(figsize=(20, 12))
#             sns.heatmap(corr_matrix, 
#                         annot=True, 
#                         fmt=".2f", 
#                         cmap=cmap)
#             plt.title("Categorical Features Correlation Matrix", fontsize=16)
#             plt.xticks(fontsize=10)
#             plt.yticks(fontsize=10)
#             # plt.show()
        
#         except: 
#             print('범주형 변수가 없습니다.')
#         return plt.gcf()

#     def setup(self, remove_outliers=True, remove_multicollinearity=True, multicollinearity_threshold=0.9,
#               train_size=0.7, fold_strategy='kfold', fold=5, feature_selection=False, feature_selection_method='classic',
#               feature_selection_estimator='lr', profile=True, session_id=786, verbose=False):
#         """예측 모델을 위한 옵션을 설정하고 데이터를 준비합니다."""
#         from pycaret.regression import setup
#         self.setup_data = setup(data=self.data, target=self.target_col,
#                                 remove_outliers=remove_outliers,
#                                 remove_multicollinearity=remove_multicollinearity,
#                                 multicollinearity_threshold=multicollinearity_threshold,
#                                 train_size=train_size,
#                                 fold_strategy=fold_strategy,
#                                 fold=fold,
#                                 feature_selection=feature_selection,
#                                 feature_selection_method=feature_selection_method,
#                                 feature_selection_estimator=feature_selection_estimator,
#                                 profile=profile,
#                                 session_id=session_id,
#                                 verbose=verbose)
#         # self.feature_names = self.data.columns.drop(self.target_col) # 타겟 열을 제외한 모든 열 이름 저장

#         result = pull()
#         result = result.iloc[:-5, :]

#         return self.setup_data, result

#     def compare_and_optimize_models(self, n_select=3, n_iter=50):
#         """모델을 비교하고 최적화합니다."""
#         best = compare_models(n_select=n_select,
#                               include=['lr', 'lasso', 'ridge', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 'mlp', 'xgboost', 'lightgbm', 'catboost'])
#         best_df = pull()

#         self.best_models = []
#         self.tuned_models = []
#         optimization_results = []  # 결과를 저장할 리스트

#         for i in range(n_select):
#             best_model_name = str(best_df.index[i])
#             model = create_model(best_model_name)
#             tuned_model = tune_model(model, n_iter=n_iter)
            
#             self.best_models.append(model)
#             self.tuned_models.append(tuned_model)
#             self.models_dict[f"모델 {i+1}"] = tuned_model  # 각 단계의 결과를 딕셔너리에 추가

#             # 각 단계의 결과를 리스트에 추가
#             result_df = pull()  # 각 모델의 최적화 결과를 가져옵니다.
#             optimization_results.append(result_df)

#         return self.models_dict, self.tuned_models, best_df, optimization_results

#     def create_ensemble_model(self, optimize='MAE'):
#         """최적화된 모델들을 사용하여 앙상블 모델을 생성합니다."""
#         if not self.tuned_models:
#             raise AttributeError("최적화된 모델이 정의되지 않았습니다. 먼저 모델 비교 및 최적화 설정 단계를 실행하세요.")
        
#         self.blended_model = blend_models(estimator_list=self.tuned_models, optimize=optimize)
#         result_df = pull()

#         return self.blended_model, result_df

#     def select_best_model(self, optimize='MAE'):
#         """최고 성능의 모델을 선정합니다."""
#         self.best_final_model = automl(optimize=optimize)
#         self.models_dict["최고 성능 모델"] = self.best_final_model 
#         print(self.best_final_model)
#         return self.best_final_model

#     def save_model(self, model_name, save_directory):
#         """모델을 지정된 디렉토리에 저장합니다."""
#         save_path = os.path.join(save_directory, model_name)
#         save_model(self.best_final_model, save_path)

#     def visualize_model(self, model, plot_type):
#         """
#         선택된 모델의 성능을 시각화합니다.
#         plot_type: 'residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature',
#                    'feature_all', 'parameter', 'tree'
#         """
#         plt.figure()
#         plot_result = plot_model(model, plot=plot_type, display_format='streamlit')
        
#         return plot_result

#     def interpret_model(self, model, plot_type, **kwargs):
#         """모델을 해석하고 SHAP 값을 시각화합니다."""
#         interpret_result = interpret_model(model, plot=plot_type, **kwargs)
#         return interpret_result

#     @classmethod
#     def predict_data(cls, model, data):
#         """모델을 사용하여 데이터를 예측합니다."""
#         predictions = predict_model(model, data=data)
#         return predictions


# class Clustering: ############################################################################################
#     def __init__(self, data, session_id=786):
#         """클러스터링 클래스의 생성자입니다."""
#         self.data = data
#         self.session_id = session_id
#         # self.models = {}
#         self.clustering_model=None
#         self.clustering_setup = None
#         self.clustered_data = None
#         self.models_dict = {}  # 모델 이름과 모델 객체를 매핑하는 딕셔너리

#     # 데이터 불러오기
#     def load_data(self):
#         """데이터를 불러옵니다."""
#         if self.data_path.endswith('.csv'):
#             self.data = pd.read_csv(self.data_path, encoding='utf-8-sig')
#         # 추가적인 데이터 형식에 대한 처리 (예: .xlsx)는 필요에 따라 확장 가능

#     def load_data(self, dataframe=None):
#         """데이터를 불러옵니다. 파일 경로 또는 데이터프레임을 사용합니다."""
#         if dataframe is not None:
#             self.data = dataframe
#         elif self.data_path.endswith('.csv'):
#             self.data = pd.read_csv(self.data_path, encoding='utf-8-sig')
#         elif self.data_path.endswith('.xlsx'):
#             self.data = pd.read_excel(self.data_path)
        
#     def load_uploaded_file(uploaded_file):
#         if uploaded_file.name.endswith('.csv'):
#             return pd.read_csv(uploaded_file)
#         elif uploaded_file.name.endswith('.xlsx'):
#             return pd.read_excel(uploaded_file)

#     # 데이터 탐색
#     def explore_data(self):
#         """데이터의 형태와 컬럼을 확인합니다."""
#         print(f'데이터 행 수: {self.data.shape[0]}')
#         print(f'데이터 열 수: {self.data.shape[1]}')
#         print(f'데이터 컬럼: {self.data.columns}')
#         data_description = self.data.describe()
#         return data_description
        
#     # 타입 별 변수 구분
#     def feature_type(self):
#         """데이터의 변수 타입을 구분합니다."""
#         categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
#         numerical_features = self.data.select_dtypes(exclude=['object']).columns.tolist()
#         print(f'Categorical Features: {categorical_features}')
#         print(f'Numerical Features: {numerical_features}')
#         return categorical_features, numerical_features

#     # 수치형 변수 시각화
#     def visualize_numerical_distribution(self):
#         """수치형 변수의 분포를 시각화합니다."""
        
#         # 수치형 변수 추출
#         num_cols = self.data.select_dtypes(exclude=['object']).columns.tolist()
        
#         # 그래프 스타일 및 팔레트 설정
#         sns.set_style("whitegrid")
#         sns.set_palette("pastel")

#         # 그래프의 행과 열 수 계산 (2개의 열)
#         rows = len(num_cols) // 2
#         if len(num_cols) % 2:
#             rows += 1

#         # 그래프 크기와 간격 조정
#         fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
#         for i, column in enumerate(num_cols):
#             ax = axes[i // 2, i % 2] if rows > 1 else axes[i % 2]
#             sns.histplot(self.data[column], kde=True, bins=30, ax=ax)
#             ax.set_title(f'Distribution of {column}', fontsize=15)
#             ax.set_ylabel('Frequency')

#         # 불필요한 빈 서브플롯 제거
#         if len(num_cols) % 2:
#             if rows > 1:
#                 axes[-1, -1].axis('off')
#             else:
#                 axes[-1].axis('off')

        
#         plt.tight_layout()
#         return plt.gcf()

#     # 범주형 변수 시각화
#     def visualize_categorical_distribution(self):
#         """범주형 변수의 분포를 시각화합니다."""
#         # 범주형 변수 추출
#         cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
#         # 범주형 변수가 없으면 None 반환
#         if not cat_cols:
#             print('범주형 변수가 없습니다.')
#             return None

#         # 그래프 그리기 설정
#         rows = len(cat_cols)

#         # 시각화 설정
#         sns.set_style("whitegrid")
#         palette = sns.color_palette("pastel")

#         # 범주형 변수의 분포와 빈도를 시각화
#         fig, axes = plt.subplots(rows, 1, figsize=(10, 5 * rows), squeeze=False)
#         for i, column in enumerate(cat_cols):
#             sns.countplot(y=self.data[column], ax=axes[i, 0], palette=palette, order=self.data[column].value_counts().index)
#             axes[i, 0].set_title(f'Distribution of {column}')
#             axes[i, 0].set_xlabel('Count')

#         plt.tight_layout()
#         return fig
        

#     # 결측치 시각화
#     def visualize_missing_distribution(self):
#         """결측치 분포를 시각화합니다."""
        
#         # 결측치 비율 계산
#         missing_ratio = self.data.isnull().mean() * 100
#         missing_count = self.data.isnull().sum()

#         # 결측치 건수 및 비율에 대한 데이터프레임
#         missing_df = pd.DataFrame({'Missing Count': missing_count, 'Missing Ratio (%)': missing_ratio})

#         # 결측치 비율을 시각화
#         plt.figure(figsize=(16, 8))
#         sns.barplot(x=missing_ratio.index, y=missing_ratio, palette=sns.color_palette("pastel"))
#         plt.axhline(30, color='red', linestyle='--')  # 30% 초과를 나타내는 빨간색 점선 추가
#         plt.xticks(rotation=45)
#         plt.title('Percentage of Missing Values by Columns')
#         plt.ylabel('Missing Value Percentage (%)')
#         plt.tight_layout()

#         # plt.show()
#         return missing_df, plt.gcf()

#     # 결측치 처리
#     def handle_and_visualize_missing(self, threshold=30):
#         """결측치 처리 후 데이터를 확인하고 시각화합니다."""
        
#         # 1. 결측치 비율 계산
#         missing_ratio = self.data.isnull().mean() * 100
        
#         # 2. 결측치 비율이 threshold(기본값: 30%)가 넘는 변수들 추출
#         columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

#         # 3. 해당 변수들 제거
#         self.data.drop(columns=columns_to_drop, inplace=True)

#         # 4. 결측치 비율 재확인
#         missing_ratio_cleaned = self.data.isnull().mean() * 100
#         missing_count_cleaned = self.data.isnull().sum()

#         # 결측치 건수 및 비율에 대한 데이터프레임
#         missing_df_cleaned = pd.DataFrame({'Missing Count': missing_count_cleaned, 'Missing Ratio (%)': missing_ratio_cleaned})

#         # 시각화 그래프
#         plt.figure(figsize=(16, 8))
#         sns.barplot(x=missing_ratio_cleaned.index, y=missing_ratio_cleaned, palette=sns.color_palette("pastel"))
#         plt.ylim(0, 100) # y축의 범위를 0부터 100까지로 설정
#         plt.xticks(rotation=45)
#         plt.title('Percentage of Missing Values by Columns (After Cleaning)')
#         plt.ylabel('Missing Value Percentage (%)')
#         plt.tight_layout()

#         # plt.show()
#         return missing_df_cleaned, plt.gcf()

#     # 수치형 상관관계 분석
#     def numerical_correlation(self):
#         """수치형 변수들 간의 상관관계를 분석합니다."""
#         corr_matrix = self.data.corr()

#         # 상단 삼각형 마스크
#         mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

#         # 파스텔 톤 색상 팔레트
#         cmap = sns.diverging_palette(230, 20, as_cmap=True)

#         plt.figure(figsize=(20, 12))
#         sns.heatmap(corr_matrix, 
#                     annot=True, 
#                     fmt=".2f", 
#                     cmap=cmap, 
#                     mask=mask,
#                     linewidths=0.5,
#                     cbar_kws={"shrink": .8})
#         plt.title("Numerical Features Correlation Matrix", fontsize=16)
#         plt.xticks(fontsize=10)
#         plt.yticks(fontsize=10)
#         # plt.show()
#         return plt.gcf()

#     def categorical_correlation(self):
#         """범주형 변수들 간의 상관관계를 분석합니다."""
#         try:
#             columns = self.data.select_dtypes(include=['object', 'category']).columns
#             corr_matrix = pd.DataFrame(index=columns, columns=columns)

#             for i in columns:
#                 for j in columns:
#                     corr_matrix.loc[i, j] = cramers_v(self.data[i], self.data[j])

#             corr_matrix = corr_matrix.astype(float)

#             # 파스텔 톤 색상 팔레트
#             cmap = sns.diverging_palette(230, 20, as_cmap=True)

#             plt.figure(figsize=(20, 12))
#             sns.heatmap(corr_matrix, 
#                         annot=True, 
#                         fmt=".2f", 
#                         cmap=cmap)
#             plt.title("Categorical Features Correlation Matrix", fontsize=16)
#             plt.xticks(fontsize=10)
#             plt.yticks(fontsize=10)
#             # plt.show()
        
#         except: 
#             print('범주형 변수가 없습니다.')

#         return plt.gcf()
    
#     def calculate_wcss(self, range_n_clusters):
#         wcss = []
#         for n_clusters in range_n_clusters:
#             clusterer = KMeans(n_clusters=n_clusters, random_state=42)
#             clusterer.fit(self.data)
#             wcss.append(clusterer.inertia_)
#         return wcss

#     def calculate_silhouette_scores(self, range_n_clusters):
#         sil_scores = []
#         for n_clusters in range_n_clusters:
#             clusterer = KMeans(n_clusters=n_clusters, random_state=42)
#             cluster_labels = clusterer.fit_predict(self.data)
#             silhouette_avg = silhouette_score(self.data, cluster_labels)
#             sil_scores.append(silhouette_avg)
#         return sil_scores

#     def plot_elbow_curve(self, range_n_clusters):
#         wcss = self.calculate_wcss(range_n_clusters)
#         plt.figure()
#         plt.plot(range_n_clusters, wcss, 'bo-', markerfacecolor='red', markersize=5)
#         plt.title('Elbow Curve')
#         plt.xlabel('Number of clusters')
#         plt.ylabel('WCSS')
#         plt.grid(True)
#         return plt.gcf()  # Return the figure object

#     def plot_silhouette_scores(self, range_n_clusters):
#         sil_scores = self.calculate_silhouette_scores(range_n_clusters)
#         plt.figure()
#         plt.plot(range_n_clusters, sil_scores, 'go-', markerfacecolor='red', markersize=5)
#         plt.title('Silhouette Scores')
#         plt.xlabel('Number of clusters')
#         plt.ylabel('Silhouette Score')
#         plt.grid(True)
#         return plt.gcf()  # Return the figure object

#     # def setup(self, profile=True, session_id = 786, verbose=False):
#     #     """클러스터링 모델을 설정합니다."""
#     #     self.clustering_setup = setup(data=self.data, 
#     #                                   profile=profile,
#     #                                   verbose=verbose,
#     #                                   session_id=session_id)
#     #     return self.clustering_setup

#     # def create_model(self, model_name, num_clusters=None):
#     #     """클러스터링 모델을 생성하고 결과를 반환합니다."""
#     #     if num_clusters is not None:
#     #         model = create_model(model_name, num_clusters=num_clusters)
#     #     else:
#     #         model = create_model(model_name)
#     #     self.models[model_name] = model
#     #     results = pull()
#     #     return model, results
    
#     # def assign_model(self, model):
#     #     """클러스터링 결과를 데이터에 할당합니다."""
#     #     self.clustered_data = assign_model(model)
#     #     return self.clustered_data

#     # def plot_model(self, model, plot_type):
#     #     """
#     #     클러스터링 모델을 시각화합니다.
#     #     plot_type: 'cluster', 'distance', 'distribution'
#     #     """
#     #     plot_result = plot_model(model, plot=plot_type)
#     #     return plot_result
    
#     # @classmethod
#     # def predict_data(cls, model, data):
#     #     """모델을 사용하여 데이터를 예측합니다."""
#     #     predictions = predict_model(model, data=data)
#     #     return predictions

#     def setup(self, profile=True, session_id = 768, verbose=False):
#             """클러스터링 모델을 설정합니다."""
#             self.clustering_setup = setup(data=self.data, 
#                                         profile=profile,
#                                         verbose=verbose,
#                                         session_id=session_id)
            
#             result = pull()
#             result = result.iloc[:-5, :]

#             return self.clustering_setup, result

#     def create_model(self, model_name, num_clusters=None):
#         """클러스터링 모델을 생성하고 결과를 반환합니다."""
#         model = create_model(model_name, num_clusters=num_clusters) if num_clusters else create_model(model_name)
#         self.clustering_model = model
#         self.models_dict[f"군집분석 모델"] = model
#         results = pull()
#         return self.models_dict, model, results

#     def assign_model(self, model):
#         """클러스터링 결과를 데이터에 할당합니다."""
#         self.clustered_data = assign_model(model)
#         return self.clustered_data
    
    
#     def save_model(self, model_name, save_directory):
#         """모델을 지정된 디렉토리에 저장합니다."""
#         save_path = os.path.join(save_directory, model_name)
#         save_model(self.clustering_model, save_path)

#     def visualize_model(self, model, plot_type):
#         """
#         클러스터링 모델을 시각화합니다.
#         plot_type: 'cluster', 'distance', 'distribution'
#         """
#         plt.figure()
#         plot_result = plot_model(model, plot=plot_type, display_format='streamlit')
#         return plot_result
    
#     def cluster_analysis(self, df):
#         """클러스터링 결과를 분석합니다."""
        
#         # 군집별로 데이터를 분리합니다.
#         clusters = df['Cluster'].unique()
#         cluster_groups = {clust: df[df['Cluster'] == clust] for clust in clusters}
        
#         # 군집별 기술 통계치를 계산합니다.
#         descriptive_stats = {clust: group.describe() for clust, group in cluster_groups.items()}
#         print(descriptive_stats)
        
#         # # 군집별 통계치 시각화를 위한 박스플롯을 그립니다.
#         # plt.figure()
#         # for column in df.columns[:-1]:  # 마지막 'Cluster' 열을 제외한 모든 열에 대해
#         #     plt.figure(figsize=(10, 6))
#         #     sns.boxplot(x='Cluster', y=column, data=self.clustered_data)
#         #     plt.title(f'Feature: {column} by Cluster')
#         #     plt.show()
        
#         return descriptive_stats
   

#     @classmethod
#     def predict_data(self, model, data):
#         """모델을 사용하여 데이터를 예측합니다."""
#         return predict_model(model, data=data)

   
    

    



