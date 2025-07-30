# Yağış Türü ve Olasılık Tahmini Modeli (Kırklareli Merkez)

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

# 3. ADIM: Eksik 'nem' ve 'ruzgar' varsa oluştur
if "nem" not in veri.columns:
    veri["nem"] = 70
if "ruzgar" not in veri.columns:
    veri["ruzgar"] = 10

# 4. ADIM: Özellikleri ve hedef değişkeni ayır
X = veri[["gun", "ay", "tavg", "tavg_3g_wma", "nem", "ruzgar"]]
y = veri["yagis_turu"]

# 5. ADIM: Oversampling ile sınıfları dengele
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# 6. ADIM: Eğitim ve test veri setlerini oluştur
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 7. ADIM: Modeli oluştur ve eğit
model = XGBClassifier(eval_metric="mlogloss")
model.fit(X_train, y_train)

# 8. ADIM: Performans Raporu
rapor = classification_report(y_test, model.predict(X_test))
print("\n--- Model Doğruluk Raporu (XGBoost + Oversampling) ---")
print(rapor)

# 9. ADIM: API'den veri çek
def get_api_weather_data(target_date):
    lat, lon = 41.735, 27.224
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_max"
        f"&timezone=Europe/Istanbul"
    )
    response = requests.get(url)
    data = response.json()

    dates = data['daily']['time']
    tmax = data['daily']['temperature_2m_max']
    tmin = data['daily']['temperature_2m_min']
    precipitation = data['daily']['precipitation_sum']
    wind_speed = data['daily']['wind_speed_10m_max']
    humidity = data['daily']['relative_humidity_2m_max']

    for i, d in enumerate(dates):
        if d == target_date.strftime("%Y-%m-%d"):
            return {
                "tavg": round((tmax[i] + tmin[i]) / 2, 1),
                "tmax": tmax[i],
                "tmin": tmin[i],
                "nem": humidity[i],
                "ruzgar": wind_speed[i],
                "yagis_mm": precipitation[i]
            }
    return None

# 10. ADIM: Tahmin fonksiyonu
def yagis_tahmin_et(gun, ay, tavg, tavg_3g_wma, nem, ruzgar):
    sample = pd.DataFrame([[gun, ay, tavg, tavg_3g_wma, nem, ruzgar]],
                          columns=["gun", "ay", "tavg", "tavg_3g_wma", "nem", "ruzgar"])
    tahmin = model.predict(sample)[0]
    olasilik = model.predict_proba(sample)[0]
    etiketler = {0: "Yağış Yok", 1: "Yağmur", 2: "Kar"}
    return tahmin, olasilik, etiketler[tahmin]

# 11. ADIM: GUI ve CSV çıktı
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
            nem = veri_api['nem']
            ruzgar = veri_api['ruzgar']
            tavg_3g_wma = veri["tavg_3g_wma"].iloc[-1]
            tahmin, olasilik, etiket = yagis_tahmin_et(tarih.day, tarih.month, tavg, tavg_3g_wma, nem, ruzgar)

            sonuc_metni += (
                f"Tarih: {tarih.day}.{tarih.month} | Sıcaklık: {tavg} °C\n"
                f"Tahmin Edilen Yağış Türü: {etiket}\n"
                f"Yağış Yok Olasılığı: % {round(olasilik[0]*100, 2)}\n"
                f"Yağmur Olasılığı: % {round(olasilik[1]*100, 2)}\n"
                f"Kar Olasılığı: % {round(olasilik[2]*100, 2)}\n"
                f"Nem: %{nem}, Rüzgar: {ruzgar} m/s, Yağış Miktarı: {veri_api['yagis_mm']} mm\n\n"
            )

            cikti_listesi.append({
                "Tarih": tarih.strftime("%Y-%m-%d"),
                "Tahmin": etiket,
                "Yağış Yok %": round(olasilik[0]*100, 2),
                "Yağmur %": round(olasilik[1]*100, 2),
                "Kar %": round(olasilik[2]*100, 2),
                "Nem %": nem,
                "Rüzgar m/s": ruzgar,
                "Yağış mm": veri_api['yagis_mm']
            })
        else:
            sonuc_metni += f"{tarih.strftime('%Y-%m-%d')}: API'den veri alınamadı.\n\n"

    if cikti_listesi:
        df_cikti = pd.DataFrame(cikti_listesi)
        df_cikti.to_csv("gunluk_yagis_tahmini.csv", index=False)

    tk.Label(pencere, text=sonuc_metni, justify="left").pack(padx=10, pady=10)
    pencere.mainloop()

# GUI başlat
if __name__ == "__main__":
    gui_otomatik()
