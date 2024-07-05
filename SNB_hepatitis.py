
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np
import pandas as pd
import math
import statistics

# importing data split packages
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold, train_test_split


def loadcsv(dataset_csv, dataset_columns):
    df = pd.read_csv(dataset_csv, header=None)
    df.columns = dataset_columns
    # data.info()
    return df

# 做10-bin discretization
def discretization(x_data, n_bins):
    disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    disc.fit(x_data)
    # show 界線值
    for i in range(len(disc.bin_edges_)):
        print("第", i, "類別的界線值:", disc.bin_edges_[i])
    # transform
    disc_x_data = disc.transform(x_data)
    # print(disc_x_data)
    return disc_x_data

def calculate_prior_laplace(df, Y):
    classes = sorted(list(df[Y].unique()))
    # print(classes)
    class_num = len(classes)
    prior = []
    for i in classes:
        prior_each_class = (len(df[df[Y] == i]) + 1) / (len(df) + (class_num * 1))
        prior.append(prior_each_class)
    # print(prior)
    return prior, classes

def five_fold_split(df, n):
    df_num = len(df)
    every_epoch_num = math.ceil((df_num / n))  #取上限值
    fold_index = [0]
    for index in range(n):
        if index < n - 1:
            df_tem = every_epoch_num * (index + 1)
        else:
            df_tem = df_num
        fold_index.append(df_tem)
    # print(fold_index)
    return fold_index

# Naive Bayes Classifier Class
class NaiveBayesClassifier:

    def __init__(self, train_data, X_columms, Y_columns, prior, all_classes_name):
        self.df = train_data
        self.X_columms = X_columms
        self.Y_columns = Y_columns
        self.prior = prior
        self.classes = all_classes_name

    # Approach2: Calculate P(X=x | Y = y) categorically
    def calculate_likelihood_categorical(self, feat_name, df_test_loc, label):
        # 計算train data中的P(x1=2|c1)、P(x1=2|c2)....
        attri_value = sorted(list(self.df[feat_name].unique()))
        df = self.df[self.df[self.Y_columns] == label]
        p_x_given_y = (len(df[df[feat_name] == df_test_loc[feat_name]])+1) / (len(df) + len(attri_value))
        return p_x_given_y

    # ACalculate P(X=x1|Y=y)P(X=x2|Y=y)...P(X=xn|Y=y) * P(Y=y) for all y and find the maximum
    def naive_bayes_categorical(self, test_data):
    # get feature names
        Y_pred = []
        # loop over every data sample
        for x in range(len(test_data)):
            # calculate likelihood
            likelihood = [1] * len(self.classes)
            for j in range(len(self.classes)):
                for i in range(len(self.X_columms)):
                    df_test_loc = test_data.iloc[x]
                    # print(df_test_loc)
                    likelihood[j] *= self.calculate_likelihood_categorical(self.X_columms[i], df_test_loc, self.classes[j])

            # calculate posterior probability
            post_prob = [1] * len(self.classes)
            for j in range(len(self.classes)):
                post_prob[j] = likelihood[j] * self.prior[j]
            # print(post_prob)
            pred = np.argmax(post_prob)
            Y_pred.append(self.classes[pred])

        return np.array(Y_pred)



if __name__ == '__main__':
    hepatitis_csv ='/Users/jiayu/Desktop/Grad NCKU/ML/Naive bayesion project/hepatitis/hepatitis.data'
    hepatitis_columns = ['Class','AGE','SEX','STEROID', 'ANTIVIRALS', 'FATIGUE','MALAISE','ANOREXIA','LIVER_BIG',
                     'LIVER_FIRM','SPLEEN_PALPABLE','SPIDERS','ASCITES','VARICES', 'BILIRUBIN','ALK_PHOSPHATE',
                     'SGOT','ALBUMIN','PROTIME','HISTOLOGY']

    hepatitis_attribute = ['AGE','SEX','STEROID', 'ANTIVIRALS', 'FATIGUE','MALAISE','ANOREXIA','LIVER_BIG',
                       'LIVER_FIRM','SPLEEN_PALPABLE','SPIDERS','ASCITES','VARICES', 'BILIRUBIN','ALK_PHOSPHATE',
                       'SGOT','ALBUMIN','PROTIME','HISTOLOGY']

    # DIE(1), LIVE(2)  #no(1),yes(2) #male(1),female(2)

    # 匯入資料
    df = loadcsv(hepatitis_csv, hepatitis_columns)
    # print(df)
    replacements = {'?': np.nan}     # 處理空值
    df.replace(replacements, inplace=True)
    DataSet = df.dropna().reset_index(drop=True)
    print(DataSet)


    # 做10-bin discretization
    numerical_attributes = ['AGE', 'BILIRUBIN', 'ALK_PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME']
    continuous_dataset = DataSet[numerical_attributes]
    disc_continuous_dataset = discretization(continuous_dataset, 10)
    disc_continuous_dataset = pd.DataFrame(disc_continuous_dataset, columns=numerical_attributes)

    DataSet.drop(labels=numerical_attributes, axis="columns", inplace=True)
    # print(DataSet)
    final_disc_DataSet = pd.concat([DataSet, disc_continuous_dataset], axis=1, join='outer')
    print(final_disc_DataSet)

    # 算prior
    all_classes_prior, all_classes_name = calculate_prior_laplace(final_disc_DataSet, 'Class')
    print("classes/Ci：", all_classes_name)
    print("Prior/P(Ci)：", all_classes_prior)



    # select att.
    each_combine_acc = []
    # SNB
    for x in range(len(hepatitis_attribute)):
        select_attribute = []
        if ((hepatitis_attribute[x] in select_attribute)==True):
            continue
        select_attribute.append(hepatitis_attribute[x])
        print(select_attribute)

        # 做5-fold
        fold_acc = []
        fold_index = five_fold_split(final_disc_DataSet, 5)     # 得到切割點

        for i in range(len(fold_index) - 1):
            test_df = final_disc_DataSet.iloc[fold_index[i]:fold_index[i + 1]].reset_index(drop=True)
            train_df = final_disc_DataSet.drop(index=final_disc_DataSet.index[fold_index[i]:fold_index[i + 1]]).reset_index(drop=True)
            # print(test_df)
            # print(train_df)

            # NaiveBayesClassifier
            NB = NaiveBayesClassifier(train_data=train_df, X_columms=select_attribute, Y_columns='Class',
                                      prior=all_classes_prior, all_classes_name=all_classes_name)
            Y_pred = NB.naive_bayes_categorical(test_data=test_df)

            # calculate accuracy
            acc = np.sum(Y_pred == test_df['Class']) / len(test_df['Class'])
            fold_acc.append(acc)
        each_combine_acc.append(statistics.mean(fold_acc))
        # print("five-fold ACC:", fold_acc)
        print("avg ACC:", statistics.mean(fold_acc))

    print("each_combine_acc", each_combine_acc)
    max_acc_loc = np.argmax(each_combine_acc)
    print("max_acc:", each_combine_acc[max_acc_loc])


