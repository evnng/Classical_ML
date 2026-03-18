import itertools
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

# Функция, которая строит гистограмму, KDE, отмечает на графике минимум, максимум, среднее, медиану, среднее + 3 сигмы

def plot_hist_with_KDE(dataset, variable, set_bins = 30):
    
    plt.figure(figsize = (8, 8))
    
    plt.axvline(x = 0, c = 'blue') # Добавление оси ординат (ax vertical line)
    plt.axhline(y = 0, c = 'blue') # Добавление оси aбсцисс (ax vertical line)
    plt.hist(dataset[variable], bins=set_bins, density=True, alpha=0.5, edgecolor='black') # Добавление гистограммы
    sns.kdeplot(dataset[variable]) # Добавление сглаженной версии гистограммы (аппроксимация)
    
    # Добавляем линии среднего и медианы, минимума и максимума
    mean_age = dataset[variable].mean()
    median_age = dataset[variable].median()
    min_age = dataset[variable].min()
    max_age = dataset[variable].max()
    std_age = dataset[variable].std()
    plt.axvline(min_age, color='yellow', linestyle='-', linewidth=2, label=f'Минимум = {min_age:.2f}')
    plt.axvline(median_age, color='green', linestyle='-', linewidth=2, label=f'Медиана = {median_age:.2f}')
    plt.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Среднее = {mean_age:.2f}')
    plt.axvline(mean_age + 3 * std_age, color='pink', linestyle='-', linewidth=2, label=f'3 sigma = {mean_age + 3 * std_age:.2f}')
    plt.axvline(max_age, color='purple', linestyle='-', linewidth=2, label=f'Максимум = {max_age:.2f}')
    
    # Подписи и заголовок
    plt.title(f'Гистограмма {variable}')
    plt.xlabel(f'Значения признака {variable}')
    plt.ylabel('Частота')
    
    # Легенда, чтобы различать линии
    plt.legend()
    
    plt.show()
    

# Функция, которая строит ROC-кривую

def plot_roc_curve_with_auc(ytest, pred):

    fpr, tpr, threshold = roc_curve(ytest, pred)
    roc_auc = auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Функция, которая строит Confusion Matrix

def plot_confusion_matrix(ytest, pred, classes, cmap=plt.cm.Greens, size=(6, 6)):
    cm = confusion_matrix(ytest, pred)
    plt.figure(figsize=size)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title('Confusion matrix', size=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))- 0.5
    
    plt.xticks(tick_marks, classes, horizontalalignment="left")
    plt.yticks(tick_marks, classes, rotation=90)

    thresh = (cm.max()+cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center", 
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylim(len(classes)-0.5, -0.5)
    plt.grid()
    plt.ylabel('True label', size=20)
    plt.xlabel('Predicted label', size=20)
    plt.tight_layout()