#%%
import warnings
warnings.filterwarnings("ignore")
import itertools
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import tensorflow as tf

#%%
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#%%
Counter(y_test)

#%%
sns.countplot(y_test)
plt.title("Зурагнуудын тархалт")
plt.xlabel("Зурагнууд")
plt.show()

#%%
x_train = x_train / 255.0
x_test  = x_test  / 255.0

#%%
# 3 болон 5 цифрүүд нь төстэй болохоор хамгийн андуурагдан
# ангилагддаг тул энэ хоёр цифрийг ангилахаар сонгов.

num_3, num_5 = 3, 5

x_sub_train = x_train[(y_train==num_3) | (y_train==num_5)]
y_sub_train = y_train[(y_train==num_3) | (y_train==num_5)]

x_sub_test = x_test[(y_test==num_3) | (y_test==num_5)]
y_sub_test = y_test[(y_test==num_3) | (y_test==num_5)]

print("x_train ", x_sub_train.shape)
print("y_test  ", x_sub_test.shape )
print("y_train ", y_sub_train.shape)
print("y_test  ", y_sub_test.shape )

#%%
from sklearn.model_selection import train_test_split

x_train_flat = x_sub_train.flatten().reshape(x_sub_train.shape[0], 28*28)
x_test_flat  = x_sub_test .flatten().reshape(x_sub_test .shape[0], 28*28)

y_sub_train_encoded = tf.keras.utils.to_categorical([1 if value==num_3 else 0 for value in y_sub_train])

X_train, X_val, Y_train, Y_val = train_test_split(
    x_train_flat       ,
    y_sub_train_encoded,
    test_size    = 0.1 ,
    random_state = 42
    )

#%%
print(
    X_train.shape,
    X_val.shape  ,
    Y_train.shape,
    Y_val.shape
)

print(Y_train)

#%%
# Анхдагч модель
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Dense(
        units      = 2, 
        activation = 'softmax'
    )
)
model.compile(
    loss      = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics   = ['accuracy']
    )

#%%
# batch хэмжээг нь их өгдөг шалтгаан нь эхний моделийг
# тааруухан сургахын тулд юм. Ингэснээр meta labelling
# хэрэглэхэд модель сайжирсан эсхийг мэдэх бололцоотой.
model.fit(
    x               = X_train,
    y               = Y_train,
    validation_data = (X_val, Y_val),
    epochs          = 3,
    batch_size      = 320
)

#%%
def plot_roc(actual, prediction):
    # Calculate ROC / AUC
    fpr, tpr, thresholds = metrics.roc_curve(actual, prediction, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    # Plot
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


#%%
prediction = model.predict(X_train)
prediction = np.array([i[1] for i in prediction])
actual     = np.array([i[1] for i in Y_train   ])


#%%
# Босго утгаар анхны модель нь хэр өндөр recall-тай байх вэ
# гэдгийг тодорхойлдог. Эндээс 0.30 босго бол хамгийн сайн
# утга гэдгийг харж болно. ROC муруй нь аль босго хамгийн 
# сайн түвшин бэ гэдгийг тодорхойлоход хэрэгтэй.
plot_roc(actual, prediction)


#%%
threshold      = 0.30
prediction_int = np.array(prediction) > threshold

print(metrics.classification_report(actual, prediction_int))

cm = metrics.confusion_matrix(actual, prediction_int)
print("Confusion matrix")
print(cm)


#%%



