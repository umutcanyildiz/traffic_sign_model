from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Chrome WebDriver'i başlat (mevcut tarayici oturumunu kullan)
options = webdriver.ChromeOptions()
options.add_argument("--user-data-dir=/tmp/chrome-profile")  # Mevcut tarayici oturumunu kullan
driver = webdriver.Chrome(options=options)

# Sürekli kontrol etmek için döngü
while True:
    try:
        # Eğer CAPTCHA veya "Hala orda misiniz?" sorusu çikarsa
        captcha = driver.find_element(By.XPATH, "//div[contains(text(), 'Hala orda misiniz?')]")
        if captcha:
            # "Robot değilim" kutusunu işaretle
            checkbox = driver.find_element(By.XPATH, "//input[@type='checkbox']")
            checkbox.click()
            print("CAPTCHA çözüldü.")
            break  # CAPTCHA çözüldükten sonra döngüden çik
    except:
        print("CAPTCHA bulunamadi. Tekrar kontrol ediliyor...")
        time.sleep(10)  # 10 saniye bekle ve tekrar kontrol et

# WebDriver'i kapat
driver.quit()