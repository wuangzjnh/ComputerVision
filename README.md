1.  pip install flask opencv-python keras tensorflow numpy
2. Cấu trúc thư mục dự án 
project/
├── app.py               # File chính chạy Flask server
├── templates/
│   └── index.html       # File HTML giao diện chính
├── static/
│   └── style.css        # File CSS (nếu muốn thêm styling)
├── haarcascade_frontalface_default.xml # File cascade để nhận diện khuôn mặt
└── Emotion_Detection.keras # Mô hình AI của bạn

3. Chạy python app.py ( cho phép truy cập camera )
