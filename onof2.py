# Yağış Türü ve Olasılık Tahmini Modeli (Kırklareli Merkez)
# ------------------------------------------
# Yağış Türü ve Olasılık Tahmini Modeli (Kırklareli Merkez)
# ------------------------------------------
# Bu Python scripti, bir CSV dosyasındaki verileri kullanarak
# makine öğrenmesi ile yağış türü ve olasılığı tahmini yapar.
# Ayrıca Open-Meteo API kullanarak bugünün ve yarının sıcaklık değerlerini gösterir (sadece bilgi amaçlı).
# Tahmin ise geçmiş veri (CSV) üzerindeki sıcaklık ve WMA bilgilerine dayanır.
# Yeni özellik: Günlük tahmin çıktıları CSV dosyasına da yazılıyor.
# GÜNCELLEME: Model olarak XGBoost entegre edildi (RandomForest yerine)

import pandas as pd
import requests, datetime
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier

# 1. ADIM: Veriyi Yükle ve tarih sütunu oluştur
veri = pd.read_csv("yagis_model_verisi.csv")
base_date = pd.to_datetime("2023-01-01")
veri["tarih"] = base_date + pd.to_timedelta(range(len(veri)), unit="D")
veri = veri.sort_values("tarih")

# 2. ADIM: 3 günlük ağırlıklı hareketli ortalama hesapla
weights = [0.2, 0.3, 0.5]
veri["tavg_3g_wma"] = veri["tavg"].rolling(window=3).apply(lambda x: sum(w*x_i for w, x_i in zip(weights, x)), raw=True)
veri = veri.dropna(subset=["tavg_3g_wma"])

# 3. ADIM: Özellikleri ve hedef değişkeni ayır
X = veri[["gun", "ay", "tavg", "tavg_3g_wma"]]
y = veri["yagis_turu"]

# 4. ADIM: Oversampling ile sınıfları dengele
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# 5. ADIM: Eğitim ve test veri setlerini oluştur
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 6. ADIM: Modeli oluştur ve eğit (XGBoost)
model = XGBClassifier(eval_metric="mlogloss")
model.fit(X_train, y_train)

# 7. ADIM: Model Performansı
rapor = classification_report(y_test, model.predict(X_test))
print("\n--- Model Doğruluk Raporu (XGBoost + Oversampling) ---")
print(rapor)

# 8. ADIM: Open-Meteo'dan çoklu hava durumu verilerini çek (sadece bilgilendirme amaçlı)

def get_api_weather_data(target_date):
    lat, lon = 41.735, 27.224
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min"
        f"&timezone=Europe/Istanbul"
    )
    response = requests.get(url)
    data = response.json()

    dates = data['daily']['time']
    tmax = data['daily']['temperature_2m_max']
    tmin = data['daily']['temperature_2m_min']

    for i, d in enumerate(dates):
        if d == target_date.strftime("%Y-%m-%d"):
            # Ortalama sıcaklığı hesapla
            return {
                "tavg": round((tmax[i] + tmin[i]) / 2, 1),
                "tmax": tmax[i],
                "tmin": tmin[i]
            }
    return None

# 9. ADIM: Tahmin fonksiyonu

def yagis_tahmin_et(gun, ay, tavg, tavg_3g_wma):
    sample = pd.DataFrame([[gun, ay, tavg, tavg_3g_wma]], columns=["gun", "ay", "tavg", "tavg_3g_wma"])
    tahmin = model.predict(sample)[0]
    olasilik = model.predict_proba(sample)[0]
    etiketler = {0: "Yağış Yok", 1: "Yağmur", 2: "Kar"}
    return tahmin, olasilik, etiketler[tahmin]

# 10. ADIM: GUI - Bugün ve yarın için tahmin ve çıktıyı CSV'ye yaz

def gui_otomatik():
    pencere = tk.Tk()
    pencere.title("Kırklareli Yağış Tahmini (Bugün & Yarın)")

    bugun = datetime.datetime.today()
    yarin = bugun + datetime.timedelta(days=1)

    sonuc_metni = ""
    cikti_listesi = []

    for tarih in [bugun, yarin]:
        veri_api = get_api_weather_data(tarih)
        if veri_api:
            tavg = veri_api['tavg']
            tavg_3g_wma = veri["tavg_3g_wma"].iloc[-1]  # son değeri kullan
            tahmin, olasilik, etiket = yagis_tahmin_et(tarih.day, tarih.month, tavg, tavg_3g_wma)
            sonuc_metni += (f"Tarih: {tarih.day}.{tarih.month} | Sıcaklık: {tavg} °C\n"
                            f"Tahmin Edilen Yağış Türü: {etiket}\n"
                            f"Yağış Yok Olasılığı: % {round(olasilik[0]*100, 2)}\n"
                            f"Yağmur Olasılığı: % {round(olasilik[1]*100, 2)}\n"
                            f"Kar Olasılığı: % {round(olasilik[2]*100, 2)}\n\n")
            cikti_listesi.append({
                "Tarih": tarih.strftime("%Y-%m-%d"),
                "Tahmin": etiket,
                "Yağış Yok %": round(olasilik[0]*100, 2),
                "Yağmur %": round(olasilik[1]*100, 2),
                "Kar %": round(olasilik[2]*100, 2)
            })
        else:
            sonuc_metni += f"{tarih.strftime('%Y-%m-%d')}: API'den veri alınamadı.\n\n"

    if cikti_listesi:
        df_cikti = pd.DataFrame(cikti_listesi)
        df_cikti.to_csv("gunluk_yagis_tahmini.csv", index=False)

    tk.Label(pencere, text=sonuc_metni, justify="left").pack(padx=10, pady=10)
    pencere.mainloop()

# GUI'yi başlat
if __name__ == "__main__":
    gui_otomatik()
