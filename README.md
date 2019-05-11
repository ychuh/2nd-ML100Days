# 2nd-ML100Days
***
[Day_1]

[Day_2]

[Day_3]

[Day_4]

[Day_5]

[Day_6]

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

**[Day_16] First Modle - 5/5**
- LogisticRegression()
- submit.to_csv(filename, index)

***
# [資料科學特徵工程技術]

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
