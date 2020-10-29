from sklearn.datasets import load_iris
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


iris = load_iris()
# np.savetxt("data/iris_data.txt",iris.data)

#データセットの読み込み
# data=np.loadtxt("data/iris_data.txt")
data=iris.data
y_train=iris.target

print(data.shape)
print(y_train.shape)

#主成分分析
pca=PCA(2)
data = pca.fit_transform(data)

#累積寄与率
print(f"累積寄与率：{str(np.cumsum(pca.explained_variance_ratio_)[-1])}")

X_train, X_test ,y_train,y_test= train_test_split(data,y_train, test_size=0.1,random_state=0)
print(X_train.shape)

neighbors=range(1,len(y_train))
scores=[]
max_score=0
max_K=0
max_scores=[]
max_K_list=[]

#Kの値を変化させながら最も正答率の高いKの値を見つける
for i in neighbors:
    model = KNeighborsClassifier(n_neighbors = i)  
    # モデルの学習
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
    if max_score<=model.score(X_test, y_test):
        max_score=model.score(X_test, y_test)
        max_K=i
        max_scores.append(max_score)
        max_K_list.append(max_K)


#Kの値とその時のテストデータに対しての正答率の推移
fig1=plt.figure()
ax1=fig1.add_subplot(1,1,1)
ax1.set_title('score×neighbors')

ax1.plot(neighbors,scores)
ax1.set_xlabel('neighbors')
ax1.set_ylabel('score')

print("最高正答率：{}".format(str(max_scores[0])))
print("最高正答率のときのK値")
print(max_K_list)

#次元削減後のアヤメデータの散布図
fig2=plt.figure()
ax2=fig2.add_subplot(1,2,2)

ax2.set_title('iris scatter plot')
#setosa
ax2.scatter(data[:50,0],data[:50,1],c="red",label="setosa")
#versicolor
ax2.scatter(data[50:100,0],data[50:100,1],c="blue",label="versicolor")
#virginica
ax2.scatter(data[100::,0],data[100::,1],c="green",label="virginica")

ax2.legend(loc="upper right", fontsize=10) # (7)凡例表示

plt.show()

