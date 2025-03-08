import cv2
import numpy as np
import os
import random
from sklearn.utils.class_weight import compute_class_weight #type: ignore

def random_blur(image, ksize=(5, 5)):
    """Rastgele bulaniklik uygular."""
    return cv2.GaussianBlur(image, ksize, 0)

def random_saturation(image, saturation_range=(0.5, 1.5)):
    """Rastgele doygunluk değisimi uygular."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(*saturation_range), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_hue_shift(image, hue_range=(-10, 10)):
    """Rastgele hue kaymasi uygular."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(*hue_range)) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_sharpen(image):
    """Rastgele keskinlestirme uygular."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def random_gamma_correction(image, gamma_range=(0.5, 1.5)):
    """Rastgele gama düzeltmesi uygular."""
    gamma = random.uniform(*gamma_range)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Giris ve çikis klasörlerini tanimlayalim
input_data_dir = os.path.join(os.path.dirname(__file__), "darknet/traffic_sign_data/images_org/TRAIN")
output_data_dir = os.path.join(os.path.dirname(__file__), "darknet/traffic_sign_data/dataset4")
os.makedirs(output_data_dir, exist_ok=True)

# Atlanacak siniflar
#skip_classes = [8,12,13,14,4,15,17] dataset2
# skip_classes= [8,10,12,13,14,4,2,3,16] #dataset3
skip_classes= [0,2,3,4,5,8,10,12,13,14,15,16,17] #dataset4

# Görüntü dosyalarini yükleyin
image_files = [f for f in os.listdir(input_data_dir) if f.endswith(('.jpg', '.png'))]

# Sinif ağirliklarini hesapla
all_labels = []
for img_name in image_files:
    label_path = os.path.join(input_data_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
    with open(label_path, 'r') as f:
        all_labels.extend([int(line.split()[0]) for line in f.readlines()])

class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(all_labels), class_weights)}

# Veri artirma uygula
for img_name in image_files:
    img_path = os.path.join(input_data_dir, img_name)
    label_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt')

    image = cv2.imread(img_path)
    with open(label_path, 'r') as f:
        labels = f.readlines()

    if any(int(label.split()[0]) in skip_classes for label in labels):
        continue

    # Rastgele bulaniklik
    augmented_image = random_blur(image)
    new_img_name = f"{os.path.splitext(img_name)[0]}_blur{os.path.splitext(img_name)[1]}"
    cv2.imwrite(os.path.join(output_data_dir, new_img_name), augmented_image)
    with open(os.path.join(output_data_dir, new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')), 'w') as f:
        f.writelines(labels)

    # Rastgele doygunluk
    augmented_image = random_saturation(image)
    new_img_name = f"{os.path.splitext(img_name)[0]}_saturation{os.path.splitext(img_name)[1]}"
    cv2.imwrite(os.path.join(output_data_dir, new_img_name), augmented_image)
    with open(os.path.join(output_data_dir, new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')), 'w') as f:
        f.writelines(labels)

    # Rastgele hue shift
    augmented_image = random_hue_shift(image)
    new_img_name = f"{os.path.splitext(img_name)[0]}_hue{os.path.splitext(img_name)[1]}"
    cv2.imwrite(os.path.join(output_data_dir, new_img_name), augmented_image)
    with open(os.path.join(output_data_dir, new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')), 'w') as f:
        f.writelines(labels)

    # Eğer sinif az bulunuyorsa ek veri artirma uygula
    unique_classes = set(int(label.split()[0]) for label in labels)
    if any(class_weight_dict[cls] > 1.5 for cls in unique_classes):  # Nadir siniflara uygula
        # Rastgele keskinlestirme
        augmented_image = random_sharpen(image)
        new_img_name = f"{os.path.splitext(img_name)[0]}_sharpen{os.path.splitext(img_name)[1]}"
        cv2.imwrite(os.path.join(output_data_dir, new_img_name), augmented_image)
        with open(os.path.join(output_data_dir, new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')), 'w') as f:
            f.writelines(labels)

        # Rastgele gama düzeltmesi
        augmented_image = random_gamma_correction(image)
        new_img_name = f"{os.path.splitext(img_name)[0]}_gamma{os.path.splitext(img_name)[1]}"
        cv2.imwrite(os.path.join(output_data_dir, new_img_name), augmented_image)
        with open(os.path.join(output_data_dir, new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')), 'w') as f:
            f.writelines(labels)

print("Veri artirma islemi tamamlandi!")
