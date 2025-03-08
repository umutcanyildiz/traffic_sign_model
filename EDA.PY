import os
import glob
import cv2
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
from collections import Counter

# Yol ve ayarlar
image_dir = os.path.join(os.path.dirname(__file__),"darknet/traffic_sign_data/dataset3")  # Görsellerin bulunduğu klasör
annotation_dir = os.path.join(os.path.dirname(__file__),"darknet/traffic_sign_data/dataset3")  # Anotasyonların bulunduğu klasör
annotation_files = glob.glob(os.path.join(annotation_dir, "*.txt"))

# Class ID -> Class Name eşlemesi
class_map = {
    0: "trespass_sign",
    1: "no_entry",
    2: "straight_or_left_only",
    3: "straight_or_right_only",
    4: "left_only_ahead",
    5: "20_speed_limit_end",
    6: "30_speed_limit",
    7: "20_speed_limit",
    8: "right_only_ahead",
    9: "no_right_turn",
    10:"no_left turn",
    11:"stop",
    12:"no_parking",
    13:"park",
    14:"bus_stop",
    15:"red_light",
    16:"yellow_light",
    17:"green_light"    # Bu sözlükte sınıf adlarını ve sınıf kimliklerini tutuyoruz
}

# Verileri toplamak için listeler
class_counts = Counter()
widths = []
heights = []
ratios = []
x_centers = []
y_centers = []

# Verileri okuma
for file in annotation_files:
    with open(file, 'r') as f:
        for line in f:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
            class_counts[int(class_id)] += 1
            widths.append(bbox_width)
            heights.append(bbox_height)
            ratios.append(bbox_width / bbox_height)
            x_centers.append(x_center)
            y_centers.append(y_center)

# sinif dagilimini goster
plt.figure(figsize=(12, 8))
sns.barplot(
    x=[class_map.get(class_id, f"Class {class_id}") for class_id in class_counts.keys()],
    y=list(class_counts.values()),
    palette="viridis"
)
# x ekseni sınıf adlarıyla etiketlenir
plt.xlabel("Class Name")
plt.ylabel("Frequency")
plt.title("Class Distribution")
plt.xticks(rotation=45,fontsize=7)
plt.show()

#2. Bounding Box Boyut Dağılımı
plt.figure(figsize=(10, 6))
plt.hist(widths, bins=20, alpha=0.7, label="Widths")
plt.hist(heights, bins=20, alpha=0.7, label="Heights")
plt.xlabel("Normalized Dimension")
plt.ylabel("Frequency")
plt.legend()
plt.title("Bounding Box Width and Height Distribution")
plt.show()

# 3. Bounding Box Oran Dağılımı
plt.figure(figsize=(10, 6))
plt.hist(ratios, bins=20, color='purple', alpha=0.7)
plt.xlabel("Aspect Ratio (Width / Height)")
plt.ylabel("Frequency")
plt.title("Bounding Box Aspect Ratio Distribution")
plt.show()

# 4. Bounding Box Merkez Dağılımı
plt.figure(figsize=(10, 6))
plt.scatter(x_centers, y_centers, alpha=0.5, s=10, color='orange')
plt.xlabel("X Center")
plt.ylabel("Y Center")
plt.title("Bounding Box Center Distribution")
plt.show()

