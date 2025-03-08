import cv2
import numpy as np
import os
import random
from sklearn.utils.class_weight import compute_class_weight #type: ignore

# Veri artirma fonksiyonlari
def random_rotation(image, angle_range=(-20, 20)):
    """Rastgele döndürme uygular."""
    angle = random.uniform(angle_range[0], angle_range[1])
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated_image, rotation_matrix

def random_shift(image, shift_range=(-0.2, 0.2)):
    """Rastgele kaydirma uygular."""
    h, w = image.shape[:2]
    tx = random.uniform(shift_range[0], shift_range[1]) * w
    ty = random.uniform(shift_range[0], shift_range[1]) * h
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted_image = cv2.warpAffine(image, translation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return shifted_image, translation_matrix

def random_zoom(image, zoom_range=(0.8, 1.2)):
    """Rastgele yakinlaştirma uygular."""
    zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
    h, w = image.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if zoom_factor < 1:
        top = (h - new_h) // 2
        bottom = h - new_h - top
        left = (w - new_w) // 2
        right = w - new_w - left
        zoomed_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        zoomed_image = resized_image[start_y:start_y + h, start_x:start_x + w]
    return zoomed_image, zoom_factor

def random_flip(image):
    """Rastgele yatay çevirme uygular."""
    if random.random() > 0.5:
        flipped_image = cv2.flip(image, 1)
        return flipped_image, True
    return image, False

def random_brightness(image, brightness_range=(-50, 50)):
    """Rastgele parlaklik değiştirme uygular."""
    brightness = random.randint(brightness_range[0], brightness_range[1])
    image = cv2.convertScaleAbs(image, alpha=1, beta=brightness)
    return image, brightness

def random_contrast(image, contrast_range=(0.5, 1.5)):
    """Rastgele kontrast değiştirme uygular."""
    contrast = random.uniform(contrast_range[0], contrast_range[1])
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    return image, contrast

def random_noise(image, noise_factor=0.01):
    """Rastgele gürültü ekler."""
    h, w, c = image.shape
    noise = np.random.randn(h, w, c) * noise_factor * 255
    noisy_image = cv2.add(image, noise.astype(np.uint8))
    return noisy_image, noise_factor

def random_color_jitter(image, jitter_range=(-50, 50)):
    """Rastgele renk jitter uygular."""
    b, g, r = cv2.split(image)
    b_jitter = random.randint(jitter_range[0], jitter_range[1])
    g_jitter = random.randint(jitter_range[0], jitter_range[1])
    r_jitter = random.randint(jitter_range[0], jitter_range[1])
    b = cv2.add(b, b_jitter)
    g = cv2.add(g, g_jitter)
    r = cv2.add(r, r_jitter)
    jittered_image = cv2.merge([b, g, r])
    return jittered_image, (b_jitter, g_jitter, r_jitter)

def random_rotation_with_scale(image, angle_range=(-20, 20), scale_range=(0.8, 1.2)):
    """Rastgele döndürme ve ölçeklendirme uygular."""
    angle = random.uniform(angle_range[0], angle_range[1])
    scale = random.uniform(scale_range[0], scale_range[1])
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated_image, rotation_matrix

# YOLO etiketlerini güncellemek için fonksiyon
def update_yolo_labels(labels, original_size, transformation):
    updated_labels = []
    orig_width, orig_height = original_size
    
    for label in labels:
        class_id, x_center, y_center, width, height = map(float, label.split())
        x_center *= orig_width
        y_center *= orig_height
        width *= orig_width
        height *= orig_height
        
        # Flip işlemi
        if transformation.get('flip', False):
            x_center = orig_width - x_center
            if class_id in {2, 3, 4, 8, 9, 10}:  # Sağ-sol değiştirme
                class_id = {2: 3, 3: 2, 4: 8, 8: 4, 9: 10, 10: 9}.get(class_id, class_id)
        
        # Döndürme işlemi
        if 'rotation_matrix' in transformation:
            rotation_matrix = transformation['rotation_matrix']
            points = np.array([[x_center, y_center]], dtype=np.float32)
            points = cv2.transform(np.array([points]), rotation_matrix)[0]
            x_center, y_center = points[0]
        
        # Yakınlaştırma işlemi
        if 'zoom_factor' in transformation:
            zoom_factor = transformation['zoom_factor']
            x_center = (x_center - orig_width / 2) * zoom_factor + orig_width / 2
            y_center = (y_center - orig_height / 2) * zoom_factor + orig_height / 2
            width *= zoom_factor
            height *= zoom_factor
        
        # Kaydırma işlemi
        if 'translation_matrix' in transformation:
            translation_matrix = transformation['translation_matrix']
            x_center += translation_matrix[0, 2]
            y_center += translation_matrix[1, 2]
        
        # Koordinatları normalize et
        x_center /= orig_width
        y_center /= orig_height
        width /= orig_width
        height /= orig_height
        
        # Koordinatları [0, 1] aralığına sınırla
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        updated_labels.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return updated_labels

# Giriş ve çikiş klasörlerini tanimlayalim
input_data_dir = os.path.join(os.path.dirname(__file__), "darknet/traffic_sign_data/images_org/TRAIN")
output_data_dir = os.path.join(os.path.dirname(__file__), "darknet/traffic_sign_data/dataset")
os.makedirs(output_data_dir, exist_ok=True)

# Hedef görüntü boyutu
target_size = (416, 416)

# Atlanacak siniflari belirleyin
skip_classes = [0,10, 12, 13, 17, 15]

# Az temsil edilen siniflar
underrepresented_classes = [1, 2, 3, 4, 6, 7, 8, 11]  # Örnek olarak belirli siniflar

# Görüntü ve etiket dosyalarini yükleyin
image_files = [f for f in os.listdir(input_data_dir) if f.endswith('.jpg') or f.endswith('.png')]
label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]

# Sinif ağirliklarini hesapla
labels = []
for label_name in label_files:
    with open(os.path.join(input_data_dir, label_name), 'r') as f:
        for line in f:
            class_id = int(line.split()[0])
            labels.append(class_id)

unique_classes = np.unique(labels)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=labels)

# Sinif ağirliklarini bir sözlüğe dönüştür (eğer model eğitimi sirasinda kullanilacaksa)
class_weight_dict = dict(zip(unique_classes, class_weights))

print("Sinif Ağirliklari:", class_weight_dict)

# Veri artirma uygula
for img_name, label_name in zip(image_files, label_files):
    img_path = os.path.join(input_data_dir, img_name)
    label_path = os.path.join(input_data_dir, label_name)

    image = cv2.imread(img_path)
    original_size = (image.shape[1], image.shape[0])

    with open(label_path, 'r') as f:
        labels = f.readlines()

    has_skip_class = any(int(label.split()[0]) in skip_classes for label in labels)
    if has_skip_class:
        continue

    # Rastgele yatay çevirme
    augmented_image, flip = random_flip(image)
    transformation = {'flip': flip}
    updated_labels = update_yolo_labels(labels, original_size, transformation)
    new_img_name = f"{os.path.splitext(img_name)[0]}_flipped{os.path.splitext(img_name)[1]}"
    new_img_path = os.path.join(output_data_dir, new_img_name)
    new_label_name = new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')
    new_label_path = os.path.join(output_data_dir, new_label_name)
    with open(new_label_path, 'w') as f:
        for label in updated_labels:
            f.write(label + '\n')
    cv2.imwrite(new_img_path, augmented_image)

    # Rastgele kontrast değiştirme (etiketleri güncelleme yok)
    augmented_image, contrast = random_contrast(image)
    new_img_name = f"{os.path.splitext(img_name)[0]}_contrast{os.path.splitext(img_name)[1]}"
    new_img_path = os.path.join(output_data_dir, new_img_name)
    new_label_name = new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')
    new_label_path = os.path.join(output_data_dir, new_label_name)
    with open(new_label_path, 'w') as f:
        for label in labels:
            f.write(label)
    cv2.imwrite(new_img_path, augmented_image)

    # Rastgele kaydirma
    augmented_image, translation_matrix = random_shift(image)
    transformation = {'translation_matrix': translation_matrix}
    updated_labels = update_yolo_labels(labels, original_size, transformation)
    new_img_name = f"{os.path.splitext(img_name)[0]}_shifted{os.path.splitext(img_name)[1]}"
    new_img_path = os.path.join(output_data_dir, new_img_name)
    new_label_name = new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')
    new_label_path = os.path.join(output_data_dir, new_label_name)
    with open(new_label_path, 'w') as f:
        for label in updated_labels:
            f.write(label + '\n')
    cv2.imwrite(new_img_path, augmented_image)

    # Rastgele yakinlaştirma
    augmented_image, zoom_factor = random_zoom(image)
    transformation = {'zoom_factor': zoom_factor}
    updated_labels = update_yolo_labels(labels, original_size, transformation)
    new_img_name = f"{os.path.splitext(img_name)[0]}_zoomed{os.path.splitext(img_name)[1]}"
    new_img_path = os.path.join(output_data_dir, new_img_name)
    new_label_name = new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')
    new_label_path = os.path.join(output_data_dir, new_label_name)
    with open(new_label_path, 'w') as f:
        for label in updated_labels:
            f.write(label + '\n')
    cv2.imwrite(new_img_path, augmented_image)

    # Rastgele parlaklik değiştirme (etiketleri güncelleme yok)
    augmented_image, brightness = random_brightness(image)
    new_img_name = f"{os.path.splitext(img_name)[0]}_brightness{os.path.splitext(img_name)[1]}"
    new_img_path = os.path.join(output_data_dir, new_img_name)
    new_label_name = new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')
    new_label_path = os.path.join(output_data_dir, new_label_name)
    with open(new_label_path, 'w') as f:
        for label in labels:
            f.write(label)
    cv2.imwrite(new_img_path, augmented_image)

    # Az temsil edilen siniflar için ek artirim
    if any(int(label.split()[0]) in underrepresented_classes for label in labels):
        for _ in range(7):  # Her bir az temsil edilen sinif için 6 ek artirim
            augmented_image = image.copy()
            transformation = {}

            # Rastgele döndürme (3 kez)
            for i in range(4):
                augmented_image, rotation_matrix = random_rotation(image)
                transformation['rotation_matrix'] = rotation_matrix
                updated_labels = update_yolo_labels(labels, original_size, transformation)
                new_img_name = f"{os.path.splitext(img_name)[0]}_rotated_{_}_{i}{os.path.splitext(img_name)[1]}"
                new_img_path = os.path.join(output_data_dir, new_img_name)
                new_label_name = new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')
                new_label_path = os.path.join(output_data_dir, new_label_name)
                with open(new_label_path, 'w') as f:
                    for label in updated_labels:
                        f.write(label + '\n')
                cv2.imwrite(new_img_path, augmented_image)

            # Rastgele gürültü ekleme (etiketleri güncelleme yok)
            augmented_image, noise_factor = random_noise(image)
            new_img_name = f"{os.path.splitext(img_name)[0]}_noise{os.path.splitext(img_name)[1]}"
            new_img_path = os.path.join(output_data_dir, new_img_name)
            new_label_name = new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            new_label_path = os.path.join(output_data_dir, new_label_name)
            with open(new_label_path, 'w') as f:
                for label in labels:
                    f.write(label)
            cv2.imwrite(new_img_path, augmented_image)

            # Rastgele renk jitter (etiketleri güncelleme yok)
            augmented_image, color_jitter = random_color_jitter(image)
            new_img_name = f"{os.path.splitext(img_name)[0]}_color_jitter{os.path.splitext(img_name)[1]}"
            new_img_path = os.path.join(output_data_dir, new_img_name)
            new_label_name = new_img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            new_label_path = os.path.join(output_data_dir, new_label_name)
            with open(new_label_path, 'w') as f:
                for label in labels:
                    f.write(label)
            cv2.imwrite(new_img_path, augmented_image)

print("Veri artirma ve etiket güncelleme tamamlandi!")