import os

# Fotoğraf ve etiket dosyalarının bulunduğu klasör yolu
directory = os.path.join(os.path.dirname(__file__), "darknet/traffic_sign_data/images/TEST")

# Desteklenen fotoğraf formatları
image_extensions = [".jpg"]

# Klasördeki tüm dosyaları listele
files = os.listdir(directory)

# Fotoğraf ve etiket dosyalarını ayır
image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
label_files = [f for f in files if os.path.splitext(f)[1].lower() == ".txt"]

# Fotoğraf dosyalarının isimlerinden uzantıyı kaldır
image_names = {os.path.splitext(f)[0] for f in image_files}
label_names = {os.path.splitext(f)[0] for f in label_files}

# Etiketi olmayan fotoğrafları bul
unlabeled_images = image_names - label_names

# Etiketi olmayan fotoğrafları sil
if unlabeled_images:
    print("Etiketi olmayan fotoğraflar siliniyor...")
    for img in unlabeled_images:
        for ext in image_extensions:
            image_path = os.path.join(directory, img + ext)
            if os.path.exists(image_path):
                #os.remove(image_path)
                print(f"Silindi: {image_path}")
else:
    print("Tüm fotoğraflarin etiket dosyasi mevcut, silinecek bir şey yok!")
