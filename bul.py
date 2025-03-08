import os
import shutil

def find_and_copy_unique_files(folder_a, folder_b, output_folder):
    # Çıktı klasörünü oluştur (eğer yoksa)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Klasörlerdeki dosyaları listele
    files_in_a = set(os.listdir(folder_a))
    files_in_b = set(os.listdir(folder_b))

    # A klasöründe olup B klasöründe olmayan dosyaları bul
    unique_to_a = files_in_a - files_in_b

    # B klasöründe olup A klasöründe olmayan dosyaları bul
    unique_to_b = files_in_b - files_in_a

    # Tüm benzersiz dosyaları birleştir
    unique_files = unique_to_a.union(unique_to_b)

    # Benzersiz dosyaları çıktı klasörüne kopyala
    for filename in unique_files:
        if filename in unique_to_a:
            src_path = os.path.join(folder_a, filename)  # A klasöründen kopyala
        else:
            src_path = os.path.join(folder_b, filename)  # B klasöründen kopyala

        dst_path = os.path.join(output_folder, filename)
        shutil.copy2(src_path, dst_path)
        print(f"{filename} benzersiz dosya olarak kopyalandı.")

    print(f"Toplam {len(unique_files)} benzersiz dosya kopyalandı.")

# Klasör yollarını belirtin
folder_a = os.path.join(os.path.dirname(__file__),"darknet/traffic_sign_data/labels_v4_30/TRAIN")  # Kaynak klasör yolu
folder_b =  os.path.join(os.path.dirname(__file__),"darknet/traffic_sign_data/labels/TRAIN")  # Hedef klasör yolu
output_folder =  os.path.join(os.path.dirname(__file__),"darknet/traffic_sign_data/ekleme/labels_TRAIN")  # Yedekleme yapılacak klasör yolu

# Fonksiyonu çalıştır
find_and_copy_unique_files(folder_a, folder_b, output_folder)

