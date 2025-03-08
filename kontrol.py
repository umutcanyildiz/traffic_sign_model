import os

def check_yolo_labels(folder_path, class_range=(0, 17)):
    """
    YOLO formatindaki etiket dosyalarini kontrol eder.
    
    :param folder_path: Etiket dosyalarinin bulunduğu klasör.
    :param class_range: Geçerli sinif araliği (min, max).
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):  # Sadece .txt dosyalarini işle
            file_path = os.path.join(folder_path, file_name)
            
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                
                if len(parts) != 5:
                    print(f"Hatali format: {file_name}, Satir {line_num}: {line.strip()}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    if not (class_range[0] <= class_id <= class_range[1]):
                        print(f"Geçersiz class_id: {file_name}, Satir {line_num}: {class_id}")
                    
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                        print(f"Geçersiz bbox değeri: {file_name}, Satir {line_num}: {x_center, y_center, width, height}")
                
                except ValueError:
                    print(f"Sayisal olmayan değer: {file_name}, Satir {line_num}: {line.strip()}")

# Kullanim
etiket_klasoru =os.path.join(os.path.dirname(__file__), "darknet/traffic_sign_data/images/TEST")
  # Klasör adini kendi yolunuza göre değiştirin
check_yolo_labels(etiket_klasoru)
print("Kontrol tamamlandi.")
