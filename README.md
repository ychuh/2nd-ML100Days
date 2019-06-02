# 2nd-ML100Days
***
## [Preprocessing]

**[Day_1] Basic Calculating and Plot - 4/19**
- Mean Square Error: 𝑀𝑆𝐸=1𝑛∑𝑛𝑖=1(𝑌𝑖−𝑌̂ 𝑖)2
- plt.plot(data_x, data_y, 'b-', label = 'label_name')

**[Day_2] Extract Data By Row, Column - 4/20**
- dir_data = './data/'
- f_app = os.path.join(dir_data, 'application_train.csv')
- DataFrame.shape/describe/head/tail
- DataFrame['Col1', 'Col2']: extract col1 and col2 only

**[Day_3] Build DataFrame and Get/Extract Figure from URL- 4/21**
- df.loc[condition, tartget_columns]
- np.random.randint(range, number)
- requests.get(URL)
- result = img2arr_fromURLs(df.loc[0:4]['URL']): get images
- plt.imshow()/ plt.show()

**[Day_4] Concept of Sub DataFrame, LabelEncoder, OneHotEncoder - 4/23**
- sub_df = pd.DataFrame(df['col1', 'col2'])

**[Day_5] Calculate Mean, Max, Min of Data and Plot Histogram - 4/24**
- df.mean()/df.max()/df.min()
- plt.hist()

**[Day_6] Detect Outliers - 4/29**
- df.boxplot()
- Emprical Cumulative Density Plot (ECDP)
- df.loc

[Day_7]

[Day_8]

[Day_9]

[Day_10]

[Day_11]

[Day_12]

[Day_13]

**[Day_14] Learn Subplot Method - 5/5**
- plt.supplot(#row, #col, location)

**[Day_15] Concept of Heatmap, PairPlot - 5/5**
- plt.figure(figsize = (width, height))
- sns.heatmap()
- np.random.sample([nrow, ncol]): create a random value matrix
- sns.PairGrid()

**[Day_16] First Model - 5/5**
- LogisticRegression()
- submit.to_csv(filename, index)

***
## [Feature Engineering]

**[Day_17] Introducing Feature Engineering - 5/7**
- LabelEncoder(): labeling object data
- MinMaxScaler()

**[Day_18] Distinquish Data Type and Split It Into Different Groups - 5/10**
- Int, Float, Object etc.
- zip(df.dtypes, df.columns)

**[Day_19] Dealing With Missing Data - 5/10**
- DataFrame.fillna(): fill missing data with 0, -1, mean, median to make dataset meaningful
- MinMaxScaler(), StandardScaler()
- cross_val_score(estimator, training set, target, cross_validation_times)

**[Day_20] Dealing With Outliers - 5/10**
- DataFrame.clip(): set a upper/lower limits and transform outliers into upper/lower limit value
- Drop outlier directly

**[Day_21] Reduce Skewness (偏度) - 5/11**
- Stats.boxcox()

**[Day_22] Impact on LogisticRegression with LabelEncoder/OneHotEncoder - 5/11**
- LabelEncoder(), pd.get_dummies()

**[Day_23] Dealing With Oject Data by Mean Encoder - 5/12**
- Smoothing
- pd.concat(df_1, df_2, axis=1): axis=1 align by column
- data = pd.concat([df[:train_num], train_Y], axis=1): initiate for mean encoding, adding target column back
- mean_df = data.groupby([c])['Survived'].mean().reset_index()
- mean_df.columns = [c, f'{c}_mean']
- data = pd.merge(data, mean_df, on=c, how='left')
- data = data.drop([c] , axis=1)

**[Day_24] CountEncoder And FeatureHash - 5/14**

**[Day_25] Time Feature - 5/15**
- datetime.weekday()
- datetime.isoweekday()
- datetime.isocalendar()
- DataFrame.apply(lambda x: x.weekday()) **v.s.** DataFrame.map(lambda x: math.sin(x*math.pi))

**[Day_26] - 5/16**

**[Day_27] Groupby Encoder - 5/18**
- 創立兩種以上的群聚編碼特徵(mean、median、mode、max、min、count)
- Feature 寧爛勿缺、多多益善

**[Day_28] Feature Selection: Filter, Wrapper, Embedded - 5/18**
- 過濾法 (Filter) : 選定統計數值與設定⾨門檻，刪除低於⾨門檻的特徵
- 包裝法 (Wrapper) : 根據⽬目標函數，逐步加入特徵或刪除特徵
- 嵌入法 (Embedded) : 使⽤用機器學習模型，根據擬合後的係數，刪除係數低於⾨門檻的特徵

**[Day_29] Feature Importance - 5/19**
- estimator = GradientBoostingClassifier()
- estimator.fit(df.values, train_Y)
- feats = pd.Series(data=estimator.feature_importances_, index=df.columns)
- feats = feats.sort_values(ascending=False)

**[Day_30] Leaf Encodeing - 5/25**

***
## [Machine Learning]

**[Day_31] Concenpt of Machine Learning - 5/25**

**[Day_32] Process of Machine Learning - 5/28**
- Project of Personalized Hey Siri (Speaker Recognition, DNN, RNN)

**[Day_33] How Does A Machine Learn? -5/28**
- Over-fitting, 過擬合代表模型可能學習到資料中的噪⾳音，導致在實際應⽤用時預測失準

**[Day_34] Train_Test_Split and K-Fold - 5/30**
- Train_Test_Split(test_size)

**[Day_35]

**[Day_36] Evaluation Metrics - 6/2**
- ROC
- AUC
- Precision-Recall
- TP, TN, FP, FN
- F_beta_score
- Confusion Metrics

**[Day_37] Regression Model - 6/2**
