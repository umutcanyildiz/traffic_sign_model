import os

def check_extra_labels(image_folder):
    extra_labels = []
    
    for file in os.listdir(image_folder):
        if file.endswith(".txt"):  # Sadece TXT dosyalarini kontrol et
            jpg_file = os.path.splitext(file)[0] + ".jpg"
            if not os.path.exists(os.path.join(image_folder, jpg_file)):
                extra_labels.append(file)
    
    if extra_labels:
        print("Fazla etiket dosyalari:")
        for txt in extra_labels:
            print(txt)
    else:
        print("Tüm etiket dosyalari ilgili resimlere sahip.")

# Kullanim örneği
image_folder =os.path.join(os.path.dirname(__file__), "darknet/traffic_sign_data/images/TEST") # Buraya resimlerin bulunduğu klasörün yolunu yazin
check_extra_labels(image_folder)
