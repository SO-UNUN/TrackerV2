# SHM Motion Tracker — Web App

ติดตามตำแหน่ง marker ในวิดีโอ และวิเคราะห์ Simple Harmonic Motion (SHM)

**University of Phayao | Faculty of Science | Physics Department**

---

## คุณสมบัติ

- 📤 อัปโหลดวิดีโอ (.mp4, .mov, .avi, .mkv, .webm)
- 🎯 ติดตาม marker 2 แบบ: Color-based และ CSRT
- 📊 กราฟ interactive (Plotly)
- 🔬 3 โหมด: Core / Plus / Advanced
- 💾 Export CSV และ JSON
- 📐 รองรับ video 30/60/120/240 fps

---

## การติดตั้ง

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

หรือบน Windows ดับเบิ้ลคลิก **`setup.bat`** ครั้งแรก แล้วใช้ **`start_app.bat`**

---

## โครงสร้างไฟล์

```
shm_tracker/
├── app.py                       # Streamlit main app
├── requirements.txt
├── README.md
├── student_manual.html          # คู่มือนักศึกษา
├── setup.bat                    # Windows first-time setup
├── start_app.bat                # Windows launcher
├── .streamlit/
│   └── config.toml
└── modules/
    ├── __init__.py
    ├── video_loader.py
    ├── marker_tracker.py        # Color + CSRT tracking
    ├── oscillator_analysis.py   # FFT, peaks, damped fit
    └── exporter.py
```

---

## วิธีใช้งานโดยย่อ

1. **Upload** — อัปโหลดวิดีโอ
2. **Marker Setup** — เลือกสี marker
3. **Track** — กด START TRACKING
4. **Results** — ดูกราฟและ export

---

## ข้อมูลทางฟิสิกส์ที่ได้

### Core
- ตำแหน่ง x(t), y(t)

### Plus
- Period T, Frequency f, ω, Amplitude
- Peak detection
- Phase space plot

### Advanced
- FFT spectrum + dominant frequency
- Damped oscillator fit:
  x(t) = A·exp(-γt)·cos(ωt + φ) + offset
- Damping ratio ζ, Quality factor Q
- Spring constant k (ถ้าใส่ mass)

---

## License

For academic use at University of Phayao.
