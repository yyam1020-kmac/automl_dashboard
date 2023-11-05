import os
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
# from AutoML_Classification import Classification
# from AutoML_Regression import Regression
# from AutoML_Clustering import Clustering
import base64

# -*- coding: utf-8-sig -*-

# # 폰트 설정
# font_path = '/Users/yerin/Library/Fonts/NanumBarunGothic.ttf'
# font_name = plt.matplotlib.font_manager.FontProperties(fname=font_path).get_name()
# plt.rcParams['font.family'] = font_name


# EDA 완료 상태를 설정하는 함수
def set_eda_complete():
    st.session_state.eda_complete = True  # EDA 완료 상태를 True로 설정

def start_setup():
    st.session_state.setup_started = True  # Setup 시작 상태를 True로 설정

# 페이지 제목
st.title('📍AutoML을 활용한 데이터 분석')

# 페이지 설명
st.write('''
    자동화된 머신러닝(AutoML) 기법을 사용하여 데이터를 분석하고, 모델을 비교, 최적화할 수 있습니다. 
    데이터를 업로드하고, 관심 있는 결과를 얻어보세요.
''')

st.sidebar.title('문제해결은행🏛️')

# 데이터 파일 업로드
uploaded_file = st.sidebar.file_uploader("데이터 파일 업로드 (CSV, Excel)", type=['csv', 'xlsx'])
df = None

# 업로드된 파일로부터 데이터프레임 로드
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

# 모델 종류 선택
model_type = st.sidebar.selectbox("모델 종류 선택", ["분류", "예측", "군집분석"])
st.session_state['model_type'] = model_type  # 세션 상태에 모델 종류 저장

# 타겟 변수 선택 (군집분석 제외)
target_column = None
if model_type != "군집분석" and df is not None:
    target_column = st.sidebar.selectbox("타겟 변수 선택", df.columns)

# 활용 컬럼 선택
selected_columns = []
if df is not None:
    all_columns = df.columns.tolist()
    if target_column:
        all_columns.remove(target_column)  # 타겟 변수 제외
    selected_columns = st.sidebar.multiselect("분석에 사용할 컬럼 선택", all_columns, default=all_columns)

# 메인 콘텐츠 영역
tab1, tab2, tab3, tab4 = st.tabs(['데이터 EDA' , '분석 모델링', '모델 성능 평가', '모델 활용'])

with tab1:
    st.markdown('## 📊 데이터 EDA')
    st.write('데이터 EDA는 데이터에 대해 확인하는 데이터 분석을 위한 준비절차입니다.')
    

    if df is not None and selected_columns:
        # 필터링된 데이터프레임
        filtered_df = df[selected_columns + ([target_column] if target_column else [])]

    if df is not None:
        # 세션 상태에서 모델 타입을 참조
        model_type = st.session_state['model_type']

        # 모델 클래스 인스턴스화 및 세션 상태에 저장
        if st.session_state['model_type'] == "분류": #model_type == "분류":
            from AutoML_Classification import Classification
            st.session_state['model'] = Classification(None, target_column)
        elif model_type == "예측":
            from AutoML_Regression import Regression
            st.session_state['model'] = Regression(None, target_column)
        elif model_type == "군집분석":
            from AutoML_Clustering import Clustering
            st.session_state['model'] = Clustering(None, target_column)

        # 모델 데이터 로드
        st.session_state['model'].load_data(dataframe=filtered_df)

        # 데이터프레임 필터링 옵션
        if st.session_state['model_type'] == "분류":# in ["분류"]:
            if st.checkbox("타겟 변수에 대한 데이터만 보기"):
                filtered_value = st.selectbox("타겟 변수 값 선택", df[target_column].unique())
                st.dataframe(df[df[target_column] == filtered_value])
            else:
                st.dataframe(df)

        elif model_type in ["예측"]:
            if st.checkbox("범위로 데이터 필터링"):
                min_val, max_val = st.slider("범위 선택", float(df[target_column].min()), float(df[target_column].max()), (float(df[target_column].min()), float(df[target_column].max())))
                st.dataframe(df[df[target_column].between(min_val, max_val)])
            else:
                st.dataframe(df)

        elif model_type in ["군집분석"]:
            st.dataframe(df)

        # 기본 통계 요약 메서드 호출
        if 'model' in st.session_state:
            st.markdown("### 수치형 데이터 통계")
            data_description = st.session_state['model'].explore_data()
            st.write(data_description)

            # feature_type() 메서드를 사용하여 범주형과 수치형 변수를 구분합니다.
        if 'model' in st.session_state:
            categorical_features, numerical_features = st.session_state['model'].feature_type()
            # 수치형 데이터 분포 시각화
            st.markdown("### 수치형 데이터 분포")
            st.write("이 그래프는 각 수치형 변수의 분포를 보여줍니다. 분포의 형태, 중앙값, 이상치 등을 확인할 수 있습니다.")
            numerical_fig = st.session_state['model'].visualize_numerical_distribution()
            st.pyplot(numerical_fig)

            # 범주형 데이터 분포 시각화 (범주형 변수가 없으면 패스)
            categorical_fig = st.session_state['model'].visualize_categorical_distribution()
            if categorical_fig:
                st.markdown("### 범주형 데이터 분포")
                st.write("범주형 변수의 각 카테고리 별 빈도수를 나타내는 그래프입니다. 각 범주의 데이터 분포를 확인할 수 있습니다.")
                st.pyplot(categorical_fig)
            else:
                st.markdown("### 범주형 데이터 분포")
                st.warning("이 데이터셋에는 범주형 변수가 없습니다.")
            

        # 결측치 분포 시각화
        if 'model' in st.session_state:
            st.markdown("### 결측치 분포")
            st.write('''이 차트는 데이터셋의 각 변수에서 결측치의 비율을 보여줍니다. 
                    높은 결측치 비율을 가진 변수는 주의 깊게 살펴볼 필요가 있습니다.
                    결측치 처리 임계값을 넘어가는 데이터는 삭제됩니다.''')
            missing_df, missing_fig = st.session_state['model'].visualize_missing_distribution()
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(missing_df, height=300)  # 데이터 프레임 높이 조절
            with col2:
                st.pyplot(missing_fig)

        # 결측치 처리 및 시각화
        missing_threshold = st.sidebar.slider("결측치 처리 임계값", 0, 100, 30)
        cleaned_missing_df, cleaned_missing_fig = st.session_state['model'].handle_and_visualize_missing(threshold=missing_threshold)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(cleaned_missing_df, height=300)  # 데이터 프레임 높이 조절
        with col2:
            st.pyplot(cleaned_missing_fig)

        # 수치형 변수 상관계수 시각화
        if 'model' in st.session_state:
            numerical_corr_fig = st.session_state['model'].numerical_correlation()
            # st.pyplot(numerical_corr_fig)
            
            # 범주형 변수 상관계수 시각화
            categorical_corr_fig = st.session_state['model'].categorical_correlation()
            # st.pyplot(categorical_corr_fig)
            st.markdown("### 변수 간 상관계수")
            st.write("변수 간의 상관관계를 나타내는 히트맵입니다. 값이 높을수록 강한 상관관계를 나타냅니다.")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 수치형 변수 상관계수")
                st.pyplot(numerical_corr_fig)
            with col2:
                st.markdown("#### 범주형 변수 상관계수")
                st.pyplot(categorical_corr_fig)
            set_eda_complete()  # EDA 완료 상태 설정

        # Check if the selected model type is 'Clustering'
        if model_type == "군집분석":
            st.write('\n')
            st.markdown("#### 군집 수 결정을 위한 그래프 확인")
            
            # Set the range for number of clusters
            range_n_clusters = list(range(2, 12))  # Typically 2 to 11 clusters
            
            # Plotting silhouette and elbow curves
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 실루엣 그래프")
                silhouette_fig = st.session_state['model'].plot_silhouette_scores(range_n_clusters)
                st.pyplot(silhouette_fig)
                
            with col2:
                st.markdown("#### 엘보우 커브 그래프")
                elbow_fig = st.session_state['model'].plot_elbow_curve(range_n_clusters)
                st.pyplot(elbow_fig)
    
    if "eda_complete" in st.session_state and st.session_state.eda_complete:
        st.write('\n')
        st.write('\n')
        st.write('-------------------------------------------------')
        st.write(' #### ❗️데이터 EDA를 마쳤습니다.')
        st.write(''' 
                다음으로 분석 모델링을 진행해볼까요?    
                상단의 **분석 모델링 탭**을 클릭해주세요!
                ''')


with tab2:
    st.markdown('## 💡분석 모델링')
    if df is not None and selected_columns:
        # 필터링된 데이터프레임
        filtered_df = df[selected_columns + ([target_column] if target_column else [])]

        # 모델 설정 옵션
        st.markdown('### 모델 설정 옵션')

        # 모델 종류별 옵션
        if st.session_state['model_type'] == "분류": #model_type == "분류":
            model = Classification(None, target_column)
            # model = st.session_state['model']
            model.load_data(dataframe=filtered_df)

            remove_outliers = st.checkbox("이상치 제거", value=False, help="데이터에서 이상치를 제거할지 여부.")
            remove_multicollinearity = st.checkbox("다중공선성 제거", value=True, help="변수 간 고도의 상관관계(다중공선성) 제거 여부.")
            multicollinearity_threshold = st.slider("다중공선성 임계값", 0.0, 1.0, 0.9, help="다중공선성을 제거할 상관관계 임계값.")
            train_size = st.slider("훈련 데이터 크기", 0.1, 1.0, 0.7, help="전체 데이터 중 훈련 데이터로 사용할 비율.")
            fold_strategy = st.selectbox("교차 검증 전략", ['stratifiedkfold', 'kfold'], index=0, help="교차 검증 시 사용할 전략, 예: stratifiedkfold, kfold.")
            fold = st.number_input("교차 검증 폴드 수", min_value=2, max_value=10, value=5, help="교차 검증 시 데이터를 나눌 폴드의 수.")
            profile = st.checkbox("프로파일링 활성화", value=True, help="데이터 프로파일링 기능 활성화 여부.")
            session_id = st.number_input("세션 ID", value=786, help="실험의 재현성을 위한 세션 ID.")
            fix_imbalance = st.checkbox("데이터 불균형 처리", value=True, help="클래스 불균형이 존재하는 데이터셋에 대한 처리 여부.")
            fix_imbalance_method = st.selectbox("불균형 처리 방법 ", ['SMOTE', 'None'], index=0, help="데이터 불균형 처리 방법 선택, 예: SMOTE.")
            # verbose = st.checkbox("상세 출력", value=False, help="모델 설정 및 훈련 과정에서 상세 정보 출력 여부.")

            # setup 시작 버튼
            if st.button("Setup 시작", on_click=start_setup):
                # setup 메서드 실행
                st.session_state['model'].setup(fix_imbalance=fix_imbalance, 
                            fix_imbalance_method=fix_imbalance_method, 
                            remove_outliers=remove_outliers, 
                            remove_multicollinearity=remove_multicollinearity,
                            multicollinearity_threshold=multicollinearity_threshold,
                            train_size=train_size,
                            fold_strategy=fold_strategy,
                            fold=fold,
                            profile=profile,
                            session_id=session_id, 
                            verbose=False)
                _, setup_results = model.setup()  # 수정된 부분
                st.success("Setup 완료!")

                # setup 결과 표시
                st.table(setup_results)
        
            st.write('\n')
            # 모델 비교 및 최적화 설정
            if "setup_started" in st.session_state and st.session_state.setup_started:
                
                st.markdown('### 모델 비교 및 최적화 설정')
                n_select = st.number_input("비교할 상위 모델의 수", min_value=1, max_value=10, value=3, step=1)
                n_iter = st.number_input("최적화 반복 횟수 ", min_value=10, max_value=100, value=50, step=10)

                st.write('\n')
                if st.button("모델 비교 및 최적화 시작"):
                    with st.spinner('모델을 비교하고 최적화하는 중...'):
                        # 모델 비교 및 최적화
                        model_dict, tuned_models, compare_result, optimization_results = model.compare_and_optimize_models(n_select=n_select, n_iter=n_iter)
                        st.session_state['models_dict'] = model_dict
                        st.success('모델 비교 및 최적화 완료!')

                        # 결과 표시 및 세션 상태 업데이트
                        st.session_state['optimization_completed'] = True
                        st.write('\n')
                        st.write('모델 성능 비교 결과')
                        st.dataframe(compare_result)

                        # 최적화된 모델 결과 표시
                        st.write('##### 최적화된 모델 결과')
                        for i, (tuned_model, result_df) in enumerate(zip(tuned_models, optimization_results)):
                            st.markdown(f'**모델 {i+1}:** {str(tuned_model)}')
                            st.dataframe(result_df)  # 각 모델의 최적화 결과를 데이터 프레임 형태로 표시합니다.
                
            if st.session_state.get('optimization_completed', False):
                # # 앙상블 모델 생성 및 최적화 설정
                # st.markdown('### 앙상블 모델 생성 및 최적화')
                # ensemble_optimize = st.selectbox("앙상블 모델 최적화 기준", ['Accuracy', 'Recall', 'Precision', 'F1'], index=0)
                # if st.button("앙상블 모델 생성"):
                #     with st.spinner('앙상블 모델을 생성하는 중...'):
                #         ensemble_model, ensemble_result = model.create_ensemble_model(optimize=ensemble_optimize)
                #         st.success('앙상블 모델 생성 완료!')

                #         # 앙상블 모델 결과 표시
                #         st.write(f'##### 앙상블 모델 성능: {str(ensemble_model)}')
                #         st.dataframe(ensemble_result)
    
                # 최고 성능 모델 선택
                st.write('\n')
                st.markdown('### 최고 성능 모델 선택')
                best_model_optimize = st.selectbox("최고 성능 모델 선택 기준", ['Accuracy', 'Recall', 'Precision', 'F1'], index=0)
                if st.button("최고 성능 모델 선택"):
                    with st.spinner('최고 성능 모델을 선택하는 중...'):
                        best_model = model.select_best_model(optimize=best_model_optimize)
                        st.session_state['models_dict']['최고 성능 모델'] = best_model
                        st.success('최고 성능 모델 선택 완료!')
                        # st.dataframe(result_df)

                        # 최고 성능 모델 정보 표시
                        st.write('\n')
                        st.markdown('##### 선택된 최고 성능 모델')
                        st.write(f'**{str(best_model)}**')
                        
                        # 세션 상태 업데이트
                        st.session_state['model_selected'] = True


                # 모델 저장
                if st.session_state.get('model_selected', False):
                    st.write('\n')
                    st.markdown('### 모델 저장 설정')
                    model_name = st.text_input("저장할 모델의 이름을 입력하세요", "best_model")
                    save_path = st.text_input("모델을 저장할 경로를 입력하세요", "/path/to/directory")

                    if st.button("모델 저장하기"):
                        with st.spinner('모델을 저장하는 중...'):
                            model.save_model(model_name, save_path)
                            st.success('모델 저장 완료!')
                            st.write(f"'{save_path}' 경로에 모델 '{model_name}'을 저장했습니다.")

    
        elif model_type == "예측":
            model = Regression(None, target_column)
            model.load_data(dataframe=filtered_df)

            remove_outliers = st.checkbox("이상치 제거", value=False, help="데이터에서 이상치를 제거할지 여부.")
            remove_multicollinearity = st.checkbox("다중공선성 제거", value=True, help="변수 간 고도의 상관관계(다중공선성) 제거 여부.")
            multicollinearity_threshold = st.slider("다중공선성 임계값", 0.0, 1.0, 0.9, help="다중공선성을 제거할 상관관계 임계값.")
            train_size = st.slider("훈련 데이터 크기", 0.1, 1.0, 0.7, help="전체 데이터 중 훈련 데이터로 사용할 비율.")
            # fold_strategy = st.selectbox("교차 검증 전략", ['kfold'], index=0, help="교차 검증 시 사용할 전략, 예: kfold.")
            fold = st.number_input("교차 검증 폴드 수", min_value=2, max_value=10, value=5, help="교차 검증 시 데이터를 나눌 폴드의 수.")
            # profile = st.checkbox("프로파일링 활성화", value=True, help="데이터 프로파일링 기능 활성화 여부.")
            session_id = st.number_input("세션 ID", value=786, help="실험의 재현성을 위한 세션 ID.")
            normalize = st.checkbox("데이터 정규화", value=True, help="데이터 정규화 여부.")
            normalize_method = st.selectbox("정규화 방법", ['zscore', 'minmax', 'maxabs', 'robust'], index=0, help="데이터 정규화 방법 선택, 예: zscore.")
            feature_selection = st.checkbox("변수 선택 여부", value=False, help="변수 선택 여부.")
            feature_selection_method = st.selectbox("변수 선택 방법", ['classic', 'univariate', 'sequential'], index=0, help="변수 선택 방법 선택, 예: classic.")
            feature_selection_estimator = st.selectbox("변수 선택 알고리즘", ['lr', 'rf', 'lightgbm', 'xgboost', 'catboost'], index=0, help="변수 선택 알고리즘 선택, 예: lr.")
            # verbose = st.checkbox("상세 출력", value=False, help="모델 설정 및 훈련 과정에서 상세 정보 출력 여부.")

            # setup 시작 버튼
            if st.button("Setup 시작", on_click=start_setup):
                # setup 메서드 실행
                model.setup(remove_outliers=remove_outliers, 
                            remove_multicollinearity=remove_multicollinearity,
                            multicollinearity_threshold=multicollinearity_threshold,
                            train_size=train_size,
                            fold_strategy='kfold',
                            fold=fold,
                            # profile=profile,
                            session_id=session_id,
                            feature_selection=feature_selection,
                            feature_selection_method=feature_selection_method,
                            feature_selection_estimator=feature_selection_estimator,
                            verbose=False)
                _, setup_results = model.setup()
                st.success("Setup 완료!")

                # setup 결과 표시
                st.table(setup_results)

                # setup이 시작되었다는 것을 st.session_state에 기록
                st.session_state.setup_started = True  # 상태 추가

            st.write('\n')
            # 모델 비교 및 최적화 설정
            if st.session_state.get('setup_started', False):
                st.markdown('### 모델 비교 및 최적화 설정')
                n_select = st.number_input("비교할 상위 모델의 수", min_value=1, max_value=10, value=3, step=1)
                n_iter = st.number_input("최적화 반복 횟수", min_value=10, max_value=100, value=50, step=10)

                if st.button("모델 비교 및 최적화 시작"):
                    with st.spinner('모델을 비교하고 최적화하는 중...'):
                        # 모델 비교 및 최적화
                        model_dict, tuned_models, compare_result, optimization_results = model.compare_and_optimize_models(n_select=n_select, n_iter=n_iter)
                        st.session_state['models_dict'] = model_dict
                        st.success('모델 비교 및 최적화 완료!')

                        # 결과 표시 및 세션 상태 업데이트
                        st.session_state['optimization_completed'] = True
                        st.write('\n')
                        st.write('모델 성능 비교 결과')
                        st.dataframe(compare_result)

                        # 최적화된 모델 결과 표시
                        st.write('##### 최적화된 모델 결과')
                        for i, (tuned_model, result_df) in enumerate(zip(tuned_models, optimization_results)):
                            st.markdown(f'**모델 {i+1}:** {str(tuned_model)}')
                            st.dataframe(result_df)  # 각 모델의 최적화 결과를 데이터 프레임 형태로 표시합니다.
        
                if st.session_state.get('optimization_completed', False):
                    # 최고 성능 모델 선택
                    st.markdown('### 최고 성능 모델 선택')
                    best_model_optimize = st.selectbox("최고 성능 모델 선택 기준", ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE'], index=0)
                    if st.button("최고 성능 모델 선택"):
                        with st.spinner('최고 성능 모델을 선택하는 중...'):
                            best_model = model.select_best_model(optimize=best_model_optimize)
                            st.session_state['models_dict']['최고 성능 모델'] = best_model
                            st.success('최고 성능 모델 선택 완료!')

                            # 최고 성능 모델 정보 표시
                            st.markdown('##### 선택된 최고 성능 모델')
                            st.write(f'**{str(best_model)}**')
                            st.session_state['model_selected'] = True

                    # 모델 저장
                    if st.session_state.get('model_selected', False):
                        st.markdown('### 모델 저장 설정')
                        model_name = st.text_input("저장할 모델의 이름을 입력하세요", "best_model")
                        save_path = st.text_input("모델을 저장할 경로를 입력하세요", "/path/to/directory")

                        if st.button("모델 저장하기"):
                            with st.spinner('모델을 저장하는 중...'):
                                model.save_model(model_name, save_path)
                                st.success('모델 저장 완료!')
                                st.write(f"'{save_path}' 경로에 모델 '{model_name}'을 저장했습니다.")

        elif model_type == "군집분석":
            model = Clustering(None, target_column)
            model.load_data(dataframe=filtered_df)

            # setup 시작 버튼
            if st.button("Setup 시작", on_click=start_setup):

                
                # setup 메서드 실행
                _, setup_results = model.setup(session_id=786, verbose=False)
                st.success("Setup 완료!")

                # setup 결과 표시
                st.table(setup_results)  # setup_results는 ClusteringExperiment 객체일 수 있습니다. 이를 테이블로 표시할 수 있는지 확인해야 합니다.

                # setup이 시작되었다는 것을 st.session_state에 기록
                st.session_state.setup_started = True  # 상태 추가

            st.write('\n')
            # 모델 생성 및 군집 할당
            if "setup_started" in st.session_state and st.session_state.setup_started:
                st.markdown('### 모델 생성 및 군집 할당')
                
                # 모델 선택
                model_name = st.selectbox("군집 모델 선택", ['kmeans', 'kmodes'])
                
                # 클러스터 수 선택
                num_clusters = st.slider("클러스터 수 선택", 2, 11, 3)
                
                # 모델 생성 버튼
                if st.button("모델 생성"):
                    with st.spinner('모델 생성 중...'):
                        # create_model 메서드 실행
                        model_dict, created_model, model_results = model.create_model(model_name, num_clusters=num_clusters)
                        st.success('모델 생성 완료!')
                        
                        st.dataframe(model_results)  # 모델 생성 결과를 데이터 프레임 형태로 표시합니다.
                        st.write(f'생성된 모델: {str(created_model)}')  # 생성된 모델의 정보를 표시합니다.
                        st.session_state['models_dict'] = model_dict

                        st.session_state['optimization_completed'] = True  # 세션 상태 업데이트

                        # 군집 할당 및 데이터프레임 저장
                        clustered_data, clustered_result = model.assign_model(created_model)
                        st.session_state['clustered_data'] = clustered_data  # 군집화된 데이터를 세션 상태에 저장

                        st.session_state['model_selected'] = True

                # 모델 저장
                if st.session_state.get('model_selected', False):
                    st.write('\n')
                    st.markdown('### 모델 저장 설정')
                    model_name = st.text_input("저장할 모델의 이름을 입력하세요", "clustering_model")
                    save_path = st.text_input("모델을 저장할 경로를 입력하세요", "/path/to/directory")

                    if st.button("모델 저장하기"):
                        with st.spinner('모델을 저장하는 중...'):
                            model.save_model(model_name, save_path)
                            st.success('모델 저장 완료!')
                            st.write(f"'{save_path}' 경로에 모델 '{model_name}'을 저장했습니다.")
                
                


with tab3:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown('## 🔎 모델 성능 평가')

    # 모델 성능 평가 탭 설명
    st.write('''
        모델 성능 평가는 모델의 성능을 확인하고, 최적의 모델을 선택할 수 있도록 도와줍니다.
    ''')
    if st.session_state.get('optimization_completed', False):
        if 'model_type' in st.session_state:
            if st.session_state['model_type'] == "분류":
                # 모델이 '분류'인 경우
            
                # 모델 선택
                model_options = list(st.session_state.get('models_dict', {}).keys())
                selected_model_name = st.selectbox("분석할 모델 선택", model_options, index=None)  # 기본값 없음

                if selected_model_name:
                    st.write(f"**{selected_model_name}** 모델을 선택했습니다.")
                    st.write(f"선택 모델 정보: {st.session_state['models_dict'][selected_model_name]}")
                    selected_model = st.session_state['models_dict'][selected_model_name]

                    # Confusion Matrix 생성
                    with st.spinner("Confusion Matrix 생성 중..."):
                        st.markdown('##### Confusion Matrix')
                        st.session_state['model'].visualize_model(selected_model, 'confusion_matrix')


                # 기타 시각화 유형 선택
                st.write('\n')
                st.markdown('##### 성능 시각화')
                plot_types = ['auc', 'pr', 'calibration', 'lift', 'gain', 'tree']
                selected_plot = st.selectbox("추가 시각화 유형 선택", plot_types)

                if st.button("시각화 보기") and selected_model_name:
                    with st.spinner(f"{selected_plot} 시각화 생성 중..."):
                        st.session_state['model'].visualize_model(selected_model, selected_plot)

                # 모델 해석
                st.write('\n')
                st.markdown('##### 모델 해석')
                interpret_types = ['summary', 'correlation', 'shap']
                selected_interpret = st.selectbox("모델 해석 유형 선택", interpret_types)

                if st.button("해석 보기") and selected_model_name:
                    with st.spinner(f"{selected_interpret} 해석 생성 중..."):
                        try:
                            interpret_result = st.session_state['model'].interpret_model(selected_model, selected_interpret)
                        except TypeError as e:
                            if "This function only supports tree based models for binary classification" in str(e):
                                st.warning("선택한 모델은 해석 제공이 불가합니다.")
                            else:
                                st.error(f"오류 발생: {e}")
                    
                    # interpret_result = model.interpret_model(selected_model, selected_interpret)

            elif st.session_state['model_type'] == "예측":
            # 모델이 '예측'인 경우
                # 모델 선택
                model_options = list(st.session_state.get('models_dict', {}).keys())
                selected_model_name = st.selectbox("분석할 모델 선택", model_options, index=None)  # 기본값 없음

                if selected_model_name:
                    st.write(f"**{selected_model_name}** 모델을 선택했습니다.")
                    st.write(f"선택 모델 정보: {st.session_state['models_dict'][selected_model_name]}")
                    selected_model = st.session_state['models_dict'][selected_model_name]

                    # 기타 시각화 유형 선택
                    st.write('\n')
                    st.markdown('##### 성능 시각화')
                    plot_types = ['residuals', 'error', 'cooks', 'vc', 'rfe', 'learning', 'manifold', 'calibration', 'dimension', 'feature', 'feature_all', 'parameter', 'lift', 'gain', 'tree', 'ks']
                    selected_plot = st.selectbox("추가 시각화 유형 선택", plot_types)

                    if st.button("시각화 보기") and selected_model_name:
                        with st.spinner(f"{selected_plot} 시각화 생성 중..."):
                            st.session_state['model'].visualize_model(selected_model, selected_plot)

                    # # 모델 해석
                    # st.write('\n')
                    # st.markdown('##### 모델 해석')
                    # interpret_types = ['summary', 'correlation', 'reason', 'shap']
                    # selected_interpret = st.selectbox("모델 해석 유형 선택", interpret_types)

                    # if st.button("해석 보기") and selected_model_name:
                    #     with st.spinner(f"{selected_interpret} 해석 생성 중..."):
                    #         try:
                    #             interpret_result = st.session_state['model'].interpret_model(selected_model, selected_interpret)
                    #         except TypeError as e:
                    #             if "This function only supports tree based models for binary classification" in str(e):
                    #                 st.warning("선택한 모델은 해석 제공이 불가합니다.")
                    #             else:
                    #                 st.error(f"오류 발생: {e}")
                    
                    # interpret_result = st.session_state['model'].interpret_model(selected_model, selected_interpret)

            elif st.session_state['model_type'] == "군집분석":
                # 모델이 '군집분석'인 경우
                # 모델 선택
                model_options = list(st.session_state.get('models_dict', {}).keys())
                selected_model_name = st.selectbox("분석할 모델 선택", model_options, index=None)

                if selected_model_name:
                    st.write(f"**{selected_model_name}** 모델을 선택했습니다.")
                    st.write(f"선택 모델 정보: {st.session_state['models_dict'][selected_model_name]}")
                    selected_model = st.session_state['models_dict'][selected_model_name]

                    # 선택한 모델의 할당된 군집 데이터 프레임을 표시하는 부분
                    if 'clustered_data' in st.session_state:
                        st.dataframe(st.session_state['clustered_data'])

                    # 시각화 보기 버튼
                    if st.button("시각화 보기"):
                        with st.spinner('군집 분포 시각화 중...'):
                            # 모델 시각화 실행
                            # plot_result = model.plot_model(selected_model, 'cluster')
                            st.session_state['model'].visualize_model(selected_model, 'distribution')
                            st.write('\n')

                            st.session_state['model'].visualize_model(selected_model, 'cluster')
                            st.write('\n')

                            st.session_state['model'].visualize_model(selected_model, 'distance')
                            st.write('\n')

                            # 상세 분석을 위한 상태 표시
                            st.session_state['visualization_shown'] = True

                        # # 군집 상세보기 버튼
                        # if st.session_state.get('visualization_shown', False):
                        #     if st.button("군집 상세보기"):
                        #         with st.spinner('군집 상세 분석 중...'):
                        #             # 군집 분석 실행
                        #             descriptive_stats = model.cluster_analysis(st.dataframe(st.session_state['clustered_data']))

                        #             # 상세 분석 결과를 표시합니다.
                        #             for cluster_id, stats in descriptive_stats.items():
                        #                 st.write(f"Cluster {cluster_id} Descriptive Statistics:")
                        #                 st.dataframe(stats)


                        # # 군집 상세보기 버튼
                        # if st.session_state.get('visualization_shown', False):
                        #     if st.button("군집 상세보기"):
                        #         with st.spinner('군집 상세 분석 중...'):
                        #             # 군집 분석 실행
                        #             st.write(clustered_result)
                        #             descriptive_stats = st.session_state['model'].cluster_analysis(clustered_result)
                        #             st.write(descriptive_stats)

                        #             # 상세 분석 결과를 표시합니다.
                        #             for cluster_id, stats in descriptive_stats.items():
                        #                 st.write(f"Cluster {cluster_id} Descriptive Statistics:")
                        #                 st.dataframe(stats)

                        if st.session_state.get('visualization_shown', False):
                            if st.button("군집 상세보기"):
                                with st.spinner('군집 상세 분석 중...'):
                                    # 군집 분석 실행
                                    descriptive_stats = st.session_state['model'].cluster_analysis(clustered_result)
                                    st.write(descriptive_stats)

                                    # 상세 분석 결과를 표시합니다.
                                    for cluster_id, stats in descriptive_stats.items():
                                        st.write(f"Cluster {cluster_id} Descriptive Statistics:")
                                        st.dataframe(stats)

                        # # 기타 시각화 유형 선택
                        # st.write('\n')
                        # st.markdown('##### 성능 시각화')
                        # plot_types = ['cluster', 'distance', 'distribution']
                        # selected_plot = st.selectbox("시각화 유형 선택", plot_types)

                        
    else:
        st.error("모델 비교 및 최적화를 먼저 완료해야 합니다.")



with tab4:
    st.markdown('## 🪄 모델 활용')
    st.write("여기에서는 선택한 모델을 사용하여 새로운 데이터의 결과를 예측할 수 있습니다.")
    

    # 사용자가 모델 선택
    if 'models_dict' in st.session_state and st.session_state['models_dict']:
        model_options = list(st.session_state['models_dict'].keys())
        selected_model_name = st.selectbox("모델 선택", model_options, index=None)

        if selected_model_name:
            if 'models_dict' in st.session_state:
                st.write(f"**{selected_model_name}** 을 선택했습니다.")
                st.write(f"선택 모델 정보: {st.session_state['models_dict'][selected_model_name]}")

            selected_model_info = st.session_state['models_dict'][selected_model_name]
            selected_model = selected_model_info


            # 예측 방식 선택
            st.write('\n')
            st.write('-------------------------------------------------')
            st.write('##### 예측 방식 선택')
            predict_option = st.radio("", ("직접 입력", "파일 업로드"))

            if model_type == "분류":
                if predict_option == "직접 입력":
                    input_data = {}
                    for col in selected_columns:  # 'selected_columns'를 활용
                        input_data[col] = st.text_input(f"{col} 입력", "0")
                
                    # 예측 버튼
                    if st.button("예측하기"):
                        # 데이터를 DataFrame으로 변환
                        input_df = pd.DataFrame([input_data])
                        # 예측 수행
                        predictions = st.session_state['model'].predict_data(selected_model, input_df)  # `predict_data` 메서드 사용
                        # 결과 표시
                        st.write(predictions)

                elif predict_option == "파일 업로드":
                    st.write('\n')
                    st.write('-------------------------------------------------')
                    st.write('##### 예측할 데이터 ')
                    uploaded_file = st.file_uploader("파일 업로드 (CSV, Excel)", type=['csv', 'xlsx'])
                    if uploaded_file:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        elif uploaded_file.name.endswith('.xlsx'):
                            df = pd.read_excel(uploaded_file)

                        # 타겟 데이터 컬럼 삭제
                        if target_column and target_column in df.columns:
                            df = df.drop(target_column, axis=1)

                        if set(selected_columns) != set(df.columns):
                            st.write("선택된 컬럼: ", selected_columns)
                            st.write("파일 컬럼: ", df.columns.tolist())
                            st.error("학습용 데이터와 동일한 형태의 파일을 제공해주세요.")
                        else:
                            if st.button("예측하기"):
                                # 예측 수행
                                predictions = st.session_state['model'].predict_data(selected_model, data=df)
                                st.write(predictions)

            if model_type == "예측":
                if predict_option == "직접 입력":
                    input_data = {}
                    for col in selected_columns:  # 'selected_columns'를 활용
                        input_data[col] = st.text_input(f"{col} 입력", "0")
                
                    # 예측 버튼
                    if st.button("예측하기"):
                        # 데이터를 DataFrame으로 변환
                        input_df = pd.DataFrame([input_data])
                        # 예측 수행
                        predictions = st.session_state['model'].predict_data(selected_model, input_df)  # `predict_data` 메서드 사용
                        # 결과 표시
                        st.write(predictions)

                elif predict_option == "파일 업로드":
                    st.write('\n')
                    st.write('-------------------------------------------------')
                    st.write('##### 예측할 데이터 ')
                    uploaded_file = st.file_uploader("파일 업로드 (CSV, Excel)", type=['csv', 'xlsx'])
                    if uploaded_file:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        elif uploaded_file.name.endswith('.xlsx'):
                            df = pd.read_excel(uploaded_file)

                        # 타겟 데이터 컬럼 삭제
                        if target_column and target_column in df.columns:
                            df = df.drop(target_column, axis=1)

                        if set(selected_columns) != set(df.columns):
                            st.write("선택된 컬럼: ", selected_columns)
                            st.write("파일 컬럼: ", df.columns.tolist())
                            st.error("학습용 데이터와 동일한 형태의 파일을 제공해주세요.")
                        else:
                            if st.button("예측하기"):
                                # 예측 수행
                                predictions = st.session_state['model'].predict_data(selected_model, data=df)
                                st.write(predictions)
                
            if model_type == "군집분석":
                if predict_option == "직접 입력":
                    input_data = {}
                    for col in selected_columns:  # 'selected_columns'를 활용
                        input_data[col] = st.text_input(f"{col} 입력", "0")

                    # 예측 버튼
                    if st.button("예측하기"):
                        # 데이터를 DataFrame으로 변환
                        input_df = pd.DataFrame([input_data])
                        # 예측 수행
                        predictions = st.session_state['model'].predict_data(selected_model, input_df)
                        # 결과 표시
                        st.write(predictions)

                elif predict_option == "파일 업로드":
                    st.write('\n')
                    st.write('-------------------------------------------------')
                    st.write('##### 예측할 데이터 ')
                    uploaded_file = st.file_uploader("파일 업로드 (CSV, Excel)", type=['csv', 'xlsx'])
                    if uploaded_file:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        elif uploaded_file.name.endswith('.xlsx'):
                            df = pd.read_excel(uploaded_file)

                        # 타겟 데이터 컬럼 삭제
                        if target_column and target_column in df.columns:
                            df = df.drop(target_column, axis=1)

                        if set(selected_columns) != set(df.columns):
                            st.write("선택된 컬럼: ", selected_columns)
                            st.write("파일 컬럼: ", df.columns.tolist())
                            st.error("학습용 데이터와 동일한 형태의 파일을 제공해주세요.")
                        else:
                            if st.button("예측하기"):
                                # 예측 수행
                                predictions = st.session_state['model'].predict_data(selected_model, data=df)
                                st.write(predictions)

   
            

                            # # CSV 파일 저장 버튼
                            # if st.button("CSV 파일로 저장하기"):
                            #     # CSV 파일로 저장
                            #     csv = predictions.to_csv(index=False, encoding='utf-8-sig')
                            #     b64 = base64.b64encode(csv.encode()).decode()  # 문자열로 인코딩
                            #     href = f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv">Download CSV file</a>'
                            #     st.markdown(href, unsafe_allow_html=True)
                        # # 예측 수행
                        # predictions = Classification.predict_data(selected_model, data=df)
                        # # 예측 결과를 원본 데이터프레임에 병합
                        # # df = pd.concat([df, predictions], axis=1)
                        # st.write(predictions)
                        
                        # # CSV 파일로 저장
                        # csv = df.to_csv(index=False, encoding='utf-8-sig')
                        # b64 = base64.b64encode(csv.encode()).decode()  # 문자열로 인코딩
                        # href = f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv">Download CSV file</a>'
                        # st.markdown(href, unsafe_allow_html=True)

        


    

