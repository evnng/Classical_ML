import itertools
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

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

# Функция, которая сжимает изображение в монохромном формате с помощью метода главных компонент (PCA)

def PCA_image_monochrome_compression(img_original, 
                                     num_of_components, 
                                     show_pixels = False, 
                                     show_part_of_variance = False, 
                                     show_graph_of_variance = False, 
                                     show_images = True, 
                                     show_compression_stats=False):
    if show_pixels:
        print(f'На картинке {img_original.shape[0] * img_original.shape[1]} пикселей')

    img_scaled = img_original / 255.0
    
    pca = PCA(n_components=num_of_components)
    img_PCA = pca.fit_transform(img_scaled)

    prop_var = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_

    PC_numbers = np.arange(pca.n_components_) + 1

    if show_part_of_variance:
        print(f'Доля дисперсии, объясняемая первыми {len(PC_numbers)} главными компонентами: {sum(prop_var)}')

    if show_graph_of_variance: 
        plt.plot(PC_numbers,
                 prop_var,
                 'ro-')
        plt.title(f'Доля дисперсии, объясняемая каждой из {len(PC_numbers)} главных компонент', fontsize = 10)
        plt.ylabel('Proportion of Variance', fontsize = 8)
        plt.xlabel('Number of Component', fontsize = 8)
        plt.show()

    img_compressed = pca.inverse_transform(img_PCA)

    if show_images:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_original, cmap='gray')
        plt.title('Оригинал')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_compressed, cmap='gray')
        plt.title(f'Восстановленное (PCA c {len(PC_numbers)} компонентами)')
        plt.axis('off')
        plt.show()

    # --- Вычисление объёмов данных ---
    if show_compression_stats:
        H, W = img_original.shape
        k = num_of_components

        # Исходное количество чисел
        original_elements = H * W
        # Количество чисел после PCA (проекции + компоненты)
        compressed_elements = H * k + k * W  # = k*(H+W)

        # Коэффициент сжатия
        compression_ratio = original_elements / compressed_elements

        print(f"\n=== Статистика сжатия ===")
        print(f"Размер изображения: {H}×{W} = {original_elements} пикселей")
        print(f"Количество компонент: {k}")
        print(f"Храним: проекции ({H}×{k}) + компоненты ({k}×{W}) = {compressed_elements} чисел")
        print(f"Коэффициент сжатия (по количеству чисел): {compression_ratio:.2f}x")

        # Оценка в байтах (если исходные пиксели uint8, а компоненты и проекции float64)
        original_bytes = original_elements * 1  # uint8 = 1 байт
        compressed_bytes = (H * k + k * W) * 8  # float64 = 8 байт
        print(f"Оригинал (uint8): ~{original_bytes/1024:.1f} KB")
        print(f"Сжатое представление (float64): ~{compressed_bytes/1024:.1f} KB")
        print(f"Реальное сжатие в байтах: {original_bytes/compressed_bytes:.2f}x (с учётом типа данных)")
        print("==========================\n")

    return img_compressed

# Функция, которая сжимает изображение в цветном RGB-формате с помощью метода главных компонент (PCA)

def PCA_image_color_compression(img_original, 
                                num_of_components, 
                                show_images=True, 
                                show_compression_stats=False):

    # Разделяем на каналы (исходное изображение в RGB)
    R = img_original[:, :, 0]
    G = img_original[:, :, 1]
    B = img_original[:, :, 2]
    
    # Сжимаем каждый канал монохромной функцией (без отображения, чтобы не плодить графики)
    R_compressed = PCA_image_monochrome_compression(R, num_of_components, show_images=False)
    G_compressed = PCA_image_monochrome_compression(G, num_of_components, show_images=False)
    B_compressed = PCA_image_monochrome_compression(B, num_of_components, show_images=False)
    
    # Объединяем каналы обратно
    img_compressed = np.dstack((R_compressed, G_compressed, B_compressed))

    # Если нужно, показываем результат
    if show_images:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_original)
        plt.title(f'Оригинал {R.shape}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.clip(img_compressed, 0, 1))
        plt.title(f'Сжатое (PCA, {num_of_components} компонент)')
        plt.axis('off')
        plt.show()

    # Вывод суммарной статистики по цветному изображению
    if show_compression_stats:
        H, W, C = img_original.shape
        k = num_of_components

        # Для цветного изображения храним отдельно проекции и компоненты для каждого канала
        # Общее количество чисел после сжатия: 3 * (H*k + k*W) = 3*k*(H+W)
        original_elements = H * W * C   # C=3 для RGB
        compressed_elements = 3 * (H * k + k * W)   # три канала

        compression_ratio = original_elements / compressed_elements

        print(f"\n=== Суммарная статистика сжатия (цветное изображение) ===")
        print(f"Размер изображения: {H}×{W}×{C} = {original_elements} чисел (пикселей×каналы)")
        print(f"Количество компонент на канал: {k}")
        print(f"Храним: для каждого канала (проекции {H}×{k} + компоненты {k}×{W})")
        print(f"Общее число чисел: {compressed_elements}")
        print(f"Коэффициент сжатия (по количеству чисел): {compression_ratio:.2f}x")

        original_bytes = original_elements * 1  # uint8
        compressed_bytes = compressed_elements * 8  # float64
        print(f"Оригинал (uint8): ~{original_bytes/1024:.1f} KB")
        print(f"Сжатое представление (float64): ~{compressed_bytes/1024:.1f} KB")
        print(f"Реальное сжатие в байтах: {original_bytes/compressed_bytes:.2f}x")
        print("============================================================\n")
    
    return img_compressed