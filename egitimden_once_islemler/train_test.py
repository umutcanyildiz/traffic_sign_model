import os

def save_image_paths_to_txt(folder_path, output_file):
    # Desteklenen resim uzantıları
    image_extensions = ['.jpg']

    # Çıktı dosyasını aç
    with open(output_file, 'w', encoding='utf-8') as file:
        # Klasördeki tüm dosyaları ve alt klasörleri tarar
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                # Dosya uzantısını kontrol et
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    # Tam dosya yolunu al
                    full_path = os.path.join(root, filename)
                    # Yolu txt dosyasına yaz
                    file.write(full_path + '\n')
    print(f"Resim yollari başariyla '{output_file}' dosyasina kaydedildi.")

# Kullanım
folder_path = os.path.join(os.path.dirname(__file__),"darknet/traffic_sign_data/images/TEST")  # Resimlerin bulunduğu klasörün yolu
output_file = "image_paths.txt"  # Kaydedilecek txt dosyasının adı
save_image_paths_to_txt(folder_path, output_file)
