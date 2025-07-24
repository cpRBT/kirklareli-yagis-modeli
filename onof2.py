# Yağış Türü ve Olasılık Tahmini Modeli (Kırklareli Merkez)
# ------------------------------------------
# Bu Python scripti, bir CSV dosyasındaki verileri kullanarak
# makine öğrenmesi ile yağış türü ve olasılığı tahmini yapar.
# Ayrıca Open-Meteo API kullanarak bugünün ve yarının sıcaklık değerlerini otomatik çeker.
# Tkinter GUI ile otomatik olarak her açıldığında tahminleri gösterir.

import pandas as pd
import requests, datetime
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. ADIM: Veriyi Yükle
veri = pd.read_csv("yagis_model_verisi.csv")

# 2. ADIM: Bağımsız (X) ve Bağımlı (y) değişkenleri ayır
X = veri[["gun", "ay", "tavg"]]
y = veri["yagis_turu"]

# 3. ADIM: Veriyi Eğitim/Test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ADIM: Modeli oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. ADIM: Model Performansı
rapor = classification_report(y_test, model.predict(X_test))
print("\n--- Model Doğruluk Raporu ---")
print(rapor)

# 6. ADIM: Open-Meteo'dan sıcaklık çek

def get_temperature_for_date(target_date):
    lat, lon = 41.735, 27.224  # Kırklareli koordinatları
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max&timezone=Europe/Istanbul"
    )
    response = requests.get(url)
    data = response.json()
    dates = data['daily']['time']
    temps = data['daily']['temperature_2m_max']
    for i, d in enumerate(dates):
        if d == target_date.strftime("%Y-%m-%d"):
            return temps[i]
    return None

# 7. ADIM: Tahmin fonksiyonu

def yagis_tahmin_et(gun, ay, tavg):
    sample = pd.DataFrame([[gun, ay, tavg]], columns=["gun", "ay", "tavg"])
    tahmin = model.predict(sample)[0]
    olasilik = model.predict_proba(sample)[0]
    etiketler = {0: "Yağış Yok", 1: "Yağmur", 2: "Kar"}
    return (f"Tarih: {gun}.{ay} | Sıcaklık: {tavg} °C\n"
            f"Tahmin Edilen Yağış Türü: {etiketler[tahmin]}\n"
            f"Yağış Yok Olasılığı: % {round(olasilik[0]*100, 2)}\n"
            f"Yağmur Olasılığı: % {round(olasilik[1]*100, 2)}\n"
            f"Kar Olasılığı: % {round(olasilik[2]*100, 2)}")

# 8. ADIM: Otomatik GUI başlatıldığında bugünün ve yarının tahminini yap

def gui_otomatik():
    pencere = tk.Tk()
    pencere.title("Kırklareli Yağış Tahmini (Bugün & Yarın)")

    bugun = datetime.datetime.today()
    yarin = bugun + datetime.timedelta(days=1)

    sonuc_metni = ""
    for tarih in [bugun, yarin]:
        tavg = get_temperature_for_date(tarih)
        if tavg:
            sonuc_metni += yagis_tahmin_et(tarih.day, tarih.month, tavg) + "\n\n"
        else:
            sonuc_metni += f"{tarih.strftime('%Y-%m-%d')}: Sıcaklık verisi alınamadı.\n\n"

    tk.Label(pencere, text=sonuc_metni, justify="left").pack(padx=10, pady=10)
    pencere.mainloop()

# GUI'yi başlat
if __name__ == "__main__":
    gui_otomatik()
