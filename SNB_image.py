
import numpy as np
import pandas as pd
import math
import statistics

# importing data split packages
from sklearn.preprocessing import KBinsDiscretizer


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
    image_csv1 ='/Users/jiayu/Desktop/Grad NCKU/ML/Naive bayesion project/image segmentation/segmentation.data'
    image_columns = ['Class','region-centroid-col','region-centroid-row','region-pixel-count', 'short-line-density-5',
                         'short-line-density-2','vedge-mean','vegde-sd','hedge-mean','hedge-sd','intensity-mean',
                         'rawred-mean','rawblue-mean','rawgreen-mean','exred-mean','exblue-mean','exgreen-mean',
                         'value-mean','saturatoin-mean','hue-mean']

    image_attribute = ['region-centroid-col','region-centroid-row','region-pixel-count', 'short-line-density-5',
                         'short-line-density-2','vedge-mean','vegde-sd','hedge-mean','hedge-sd','intensity-mean',
                         'rawred-mean','rawblue-mean','rawgreen-mean','exred-mean','exblue-mean','exgreen-mean',
                         'value-mean','saturatoin-mean','hue-mean']



    # 匯入資料
    DataSet = loadcsv(image_csv1, image_columns)
    # DataSet.info()
    # print(DataSet)


    # 做10-bin discretization
    disc_x_data = discretization(DataSet[image_attribute], 10)
    disc_DataSet = pd.DataFrame(disc_x_data, columns=image_attribute).join(DataSet['Class'])   # 轉成df
    print(disc_DataSet)

    # 用train算prior
    all_classes_prior, all_classes_name = calculate_prior_laplace(disc_DataSet, 'Class')
    print("classes/Ci：", all_classes_name)
    print("Prior/P(Ci)：", all_classes_prior)

    # 先將資料打混
    train_test_df = disc_DataSet.sample(frac=1.0, random_state=1).reset_index(drop=True)
    # print(train_test_df)

    # select att SNB
    each_combine_acc = []

    for x in range(len(image_attribute)):
        select_attribute = []
        if ((image_attribute[x] in select_attribute) == True):
            continue
        select_attribute.append(image_attribute[x])
        print(select_attribute)

        # 做5-fold
        fold_acc = []
        fold_index = five_fold_split(train_test_df, 5)  # 得到切割點
        for i in range(len(fold_index) - 1):
            test_df = train_test_df.iloc[fold_index[i]:fold_index[i + 1]].reset_index(drop=True)
            train_df = train_test_df.drop(index=train_test_df.index[fold_index[i]:fold_index[i + 1]]).reset_index(drop=True)
            # print(test_df)
            # print(train_df)

        for i in range(len(fold_index) - 1):
            test_df = train_test_df.iloc[fold_index[i]:fold_index[i + 1]].reset_index(drop=True)
            train_df = train_test_df.drop(index=train_test_df.index[fold_index[i]:fold_index[i + 1]]).reset_index(
                drop=True)

            # NaiveBayesClassifier
            NB = NaiveBayesClassifier(train_data=train_df, X_columms=select_attribute,
                                      Y_columns='Class', prior=all_classes_prior, all_classes_name=all_classes_name)
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