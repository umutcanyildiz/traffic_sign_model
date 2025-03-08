import cv2
import numpy as np
import os
import random
from sklearn.utils.class_weight import compute_class_weight #type: ignore

def random_blur(image, ksize=(5, 5)):
    """Rastgele bulaniklik uygular."""
    return cv2.GaussianBlur(image, ksize, 0)

def random_saturation(image, saturation_range=(0.5, 1.5)):
    """Rastgele doygunluk değişimi uygular."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(*saturation_range), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_hue_shift(image, hue_range=(-10, 10)):
    """Rastgele hue kaymasi uygular."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(*hue_range)) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_tint(image, intensity_range=(-30, 30)):
    """Rastgele ton değişimi uygular."""
    b, g, r = cv2.split(image)
    tint = np.random.randint(intensity_range[0], intensity_range[1], 3)
    b = np.clip(b + tint[0], 0, 255).astype(np.uint8)
    g = np.clip(g + tint[1], 0, 255).astype(np.uint8)
    r = np.clip(r + tint[2], 0, 255).astype(np.uint8)
    return cv2.merge((b, g, r))

def random_shadow(image):
    """Rastgele gölge ekleme."""
    h, w, _ = image.shape
    shadow_mask = np.zeros_like(image, dtype=np.uint8)
    x1, x2 = random.randint(0, w // 2), random.randint(w // 2, w)
    y1, y2 = random.randint(0, h // 2), random.randint(h // 2, h)
    cv2.rectangle(shadow_mask, (x1, y1), (x2, y2), (50, 50, 50), -1)
    alpha = random.uniform(0.3, 0.7)
    return cv2.addWeighted(image, 1, shadow_mask, alpha, 0)

# Giriş ve çikiş klasörlerini tanimlayalim
input_data_dir = os.path.join(os.path.dirname(__file__), "darknet/traffic_sign_data/images_org/TRAIN")
output_data_dir = os.path.join(os.path.dirname(__file__), "darknet/traffic_sign_data/dataset5")
os.makedirs(output_data_dir, exist_ok=True)

# Atlanacak siniflar
#skip_classes= [0,2,3,4,5,8,10,12,13,14,15,16,17] # dataset4
skip_classes= [8,10,12,13,14,4,2,3,16] #dataset3
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

     # Rastgele ton değişimi
    augmented_image = random_tint(image)
    new_img_name = f"{os.path.splitext(img_name)[0]}_tint{os.path.splitext(img_name)[1]}"
    cv2.imwrite(os.path.join(output_data_dir, new_img_name), augmented_image)
    with open(os.path.join(output_data_dir, new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')), 'w') as f:
        f.writelines(labels)

    # Rastgele gölgeleme
    augmented_image = random_shadow(image)
    new_img_name = f"{os.path.splitext(img_name)[0]}_shadow{os.path.splitext(img_name)[1]}"
    cv2.imwrite(os.path.join(output_data_dir, new_img_name), augmented_image)
    with open(os.path.join(output_data_dir, new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')), 'w') as f:
        f.writelines(labels)

    # Eğer sinif az bulunuyorsa ek veri artirma uygula
    unique_classes = set(int(label.split()[0]) for label in labels)
    if any(class_weight_dict[cls] > 1.5 for cls in unique_classes):  # Nadir siniflara uygula
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
       

print("Veri artirma işlemi tamamlandi!")
