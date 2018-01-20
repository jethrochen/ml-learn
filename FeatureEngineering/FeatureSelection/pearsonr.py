
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
import numpy as np

# https://www.jianshu.com/p/2624521f87eb

iris = load_iris()

# 选择K个最好的特征，返回特征选择后的数据


def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for ret in map(lambda x: pearsonr(x, y), X.T):
        scores.append(abs(ret[0]))
        pvalues.append(ret[1])
    return np.array(scores), np.array(pvalues)


# 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
# 参数k为选择的特征个数
transformer = SelectKBest(score_func=multivariate_pearsonr, k=2)
Xt_pearsonr = transformer.fit_transform(iris.data, iris.target)
print(Xt_pearsonr)