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
threshold      = 0.10
prediction_int = np.array(prediction) > threshold

print(metrics.classification_report(actual, prediction_int))

cm = metrics.confusion_matrix(actual, prediction_int)
print("Confusion matrix")
print(cm)


#%%
threshold      = 0.30
prediction_int = np.array(prediction) > threshold

print(metrics.classification_report(actual, prediction_int))

cm = metrics.confusion_matrix(actual, prediction_int)
print("Confusion matrix")
print(cm)


#%%
actual, prediction_int

#%%
# Meta модель сургахад бэлтгэх
meta_labels        = (prediction_int.astype(int) & actual.astype(int)).astype(float)
meta_label_encoded = tf.keras.utils.to_categorical(meta_labels)

prediction_int = prediction_int.reshape((-1, 1))

# MNIST data + forecast_int
new_features = np.concatenate((prediction_int, X_train), axis=1)


#%%
meta_model = tf.keras.Sequential()
meta_model.add(tf.keras.layers.Dense(units=32, activation='relu'   ))
meta_model.add(tf.keras.layers.Dense(units=2 , activation='softmax'))

meta_model.compile(
    loss      = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics   = ['accuracy']
)

#%%
meta_model.fit(
    x          = new_features      ,
    y          = meta_label_encoded,
    epochs     = 4,
    batch_size = 32
    )


#%%
def test_meta_label(primary_model, secondary_model, x, y, threshold):
    # Primary model
    actual = np.array([i[1] for i in y]) == 1

    primary_prediction     = primary_model.predict(x)
    primary_prediction     = np.array([i[1] for i in primary_prediction]).reshape((-1, 1))
    primary_prediction_int = primary_prediction > threshold # binary labels

    print('Анхдагч моделийн гүйцэтгэл:')
    print(metrics.classification_report(actual, primary_prediction > 0.50))
    print('Confusion Matrix')
    print(metrics.confusion_matrix(actual, primary_prediction_int))
    accuracy = (actual == primary_prediction_int.flatten()).sum() / actual.shape[0]
    print('Нарийвчлал: ', round(accuracy, 4))
    print('')


    # Secondary model
    new_features = np.concatenate((primary_prediction_int, x), axis=1)

    meta_prediction     = secondary_model.predict(new_features)
    meta_prediction     = np.array([i[1] for i in meta_prediction])
    meta_prediction_int = meta_prediction > 0.5 # binary labels

    # Now combine primary and secondary model in a final prediction
    final_prediction = (meta_prediction_int & primary_prediction_int.flatten())

    print('Мета моделийн гүйцэтгэл: ')
    print(metrics.classification_report(actual, final_prediction))
    print('Confusion Matrix')
    print(metrics.confusion_matrix(actual, final_prediction))
    accuracy = (actual == final_prediction).sum() / actual.shape[0]
    print('Нарийвчлал: ', round(accuracy, 4))



#%%
print("##### Сургасан өгөгдөл дээрхи гүйцэтгэл #####\n\n")
test_meta_label(
    primary_model   = model, 
    secondary_model = meta_model, 
    x               = X_train, 
    y               = Y_train, 
    threshold       = threshold
    )

#%%
print("##### Бататгах өгөгдөл дээрхи гүйцэтгэл ##### \n\n")
test_meta_label(
    primary_model   = model, 
    secondary_model = meta_model, 
    x               = X_val, 
    y               = Y_val, 
    threshold       = threshold
    )


#%%
print("##### Сургалтаас гадуурхи өгөгдөл дээрхи гүйцэтгэл ##### \n\n")
x_test_flat        = x_sub_test.flatten().reshape(x_sub_test.shape[0], 28*28)
y_sub_test_encoded = tf.keras.utils.to_categorical(
    [1 if value == num_3 else 0 for value in y_sub_test]
    )

test_meta_label(
    primary_model   = model, 
    secondary_model = meta_model, 
    x               = x_test_flat, 
    y               = y_sub_test_encoded, 
    threshold       = threshold
    )  

#%%



