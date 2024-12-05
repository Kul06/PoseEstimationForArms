import sys
import time
import cv2
import mediapipe as mp
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
import qdarkstyle

# MediaPipe Pose ayarları
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Renk Ayarları
LEFT_ARM_COLOR = (0, 128, 255)  # Sol Kol    (Turuncu ton)
RIGHT_ARM_COLOR = (0, 0, 255)  # Sağ Kol (Kırmızı ton)
NODE_COLOR = (255, 255, 255)  # Düğümler (Beyaz)


# Hareket durumlarını takip etmek için önceki açılar ve pozisyonlar
previous_left_angle = None
previous_right_angle = None
previous_left_position = None
previous_right_position = None

def calculate_angle(a, b, c):
    """Üç noktadan açı hesaplar (2D düzlemde)."""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))
    return 360 - angle if angle > 180.0 else angle

def display_text(frame, text, position, color, font_scale=0.8, thickness=2):
    """Ekranda belirtilen pozisyona metin yazar."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def process_video(input_path, output_path, update_status):
    """Videoyu işleyerek işlenmiş videoyu kaydeder ve hareket analizini yapar."""
    update_status("Video işleme başladı...")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        update_status(f"Video açılamadı: {input_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    global previous_left_angle, previous_right_angle, previous_left_position, previous_right_position
    last_update_time = time.time()  # Son güncelleme zamanı
    movement_buffer = []  # Hareketlerin birikmesi için liste

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        movement_status = "HAREKET: N/A"  # Varsayılan hareket bilgisi

        if results.pose_landmarks:
            h, w, _ = frame.shape
            keypoints = results.pose_landmarks.landmark

            def get_point(idx):
                lm = keypoints[idx]
                return (int(lm.x * w), int(lm.y * h)) if lm.visibility > 0.3 else None

            left_shoulder, left_elbow, left_wrist = get_point(11), get_point(13), get_point(15)
            right_shoulder, right_elbow, right_wrist = get_point(12), get_point(14), get_point(16)

            left_angle, right_angle = None, None

            # Sol kol hareket analizi
            if left_shoulder and left_elbow and left_wrist:
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                if previous_left_angle is not None:
                    if left_angle > previous_left_angle:
                        movement_status = "Dirsek Acma"
                    elif left_angle < previous_left_angle:
                        movement_status = "Dirsek Kapatma"

                if previous_left_position is not None:
                    if left_elbow[0] > previous_left_position[0]:
                        movement_status = "Yatay Ekstansiyon"
                    elif left_elbow[0] < previous_left_position[0]:
                        movement_status = "Yatay Fleksiyon"

                    if left_elbow[1] < previous_left_position[1]:
                        movement_status = "Kol Kaldirma"
                    elif left_elbow[1] > previous_left_position[1]:
                        movement_status = "Kol Indirme"

                previous_left_angle = left_angle
                previous_left_position = left_elbow

                # Sol kol çizimi
                cv2.line(frame, left_shoulder, left_elbow, LEFT_ARM_COLOR, 2)
                cv2.line(frame, left_elbow, left_wrist, LEFT_ARM_COLOR, 2)
                for point in [left_shoulder, left_elbow, left_wrist]:
                    cv2.circle(frame, point, 6, NODE_COLOR, -1)

            # Sağ kol hareket analizi
            if right_shoulder and right_elbow and right_wrist:
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                if previous_right_angle is not None:
                    if right_angle > previous_right_angle:
                        movement_status = "Dirsek Acma"
                    elif right_angle < previous_right_angle:
                        movement_status = "Dirsek Kapatma"

                if previous_right_position is not None:
                    if right_elbow[0] > previous_right_position[0]:
                        movement_status = "Yatay Ekstansiyon"
                    elif right_elbow[0] < previous_right_position[0]:
                        movement_status = "Yatay Fleksiyon"

                    if right_elbow[1] < previous_right_position[1]:
                        movement_status = "Kol Kaldirma"
                    elif right_elbow[1] > previous_right_position[1]:
                        movement_status = "Kol Indirme"

                previous_right_angle = right_angle
                previous_right_position = right_elbow

                # Sağ kol çizimi
                cv2.line(frame, right_shoulder, right_elbow, RIGHT_ARM_COLOR, 2)
                cv2.line(frame, right_elbow, right_wrist, RIGHT_ARM_COLOR, 2)
                for point in [right_shoulder, right_elbow, right_wrist]:
                    cv2.circle(frame, point, 6, NODE_COLOR, -1)

            movement_buffer.append(movement_status)
            # Açı bilgilerini sağ orta tarafta göster
            if left_angle is not None:
                display_text(frame, f"SOL KOL:{int(left_angle)} DERECE", (w - 300, h // 2 - 30), LEFT_ARM_COLOR)
            if right_angle is not None:
                display_text(frame, f"SAG KOL:{int(right_angle)} DERECE", (w - 300, h // 2), RIGHT_ARM_COLOR)

                if time.time() - last_update_time >= 1:
                    if movement_buffer:
                        most_frequent_movement = max(set(movement_buffer), key=movement_buffer.count)
                        movement_status = most_frequent_movement
                        movement_buffer.clear()  # Listeyi sıfırla
                    last_update_time = time.time()

        # Hareket bilgisini ekranın üst kısmında göster
        display_text(frame, movement_status, (w // 2 - 150, 40), (0, 255, 0), font_scale=1, thickness=2)

        out.write(frame)
        cv2.imshow("Islenen  Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            update_status("Isleme kullanici tarafından durduruldu.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    update_status(f"Video isleme tamamlandi ve kaydedildi: {output_path}")

def closeEvent(self, event):
    """Pencere kapanırken kaynakları serbest bırak."""
    if self.capture is not None:
        self.capture.release()
    self.timer.stop()
    cv2.destroyAllWindows()
    super().closeEvent(event)


# MASA USTU UYGULAMA TASARIMI ICIN
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kol Açı Analizi Uygulaması")
        self.setGeometry(100, 100, 600, 400)

        self.main_widget = QWidget()
        self.layout = QVBoxLayout()

        self.select_button = QPushButton("Girdi Videosu Seç")
        self.select_button.clicked.connect(self.select_video)
        self.layout.addWidget(self.select_button)

        self.process_button = QPushButton("İşleme Başla")
        self.process_button.clicked.connect(self.process_selected_video)
        self.layout.addWidget(self.process_button)

        self.status_label = QLabel("Durum: Bekleniyor...")
        self.layout.addWidget(self.status_label)

        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

    def select_video(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Video Seç", "", "Video Dosyaları (*.mp4 *.avi)",
                                                   options=options)
        if file_path:
            self.selected_video = file_path
            self.status_label.setText(f"Seçilen Video: {file_path}")
        else:
            self.status_label.setText("Video seçilmedi.")

    def process_selected_video(self):
        if hasattr(self, 'selected_video'):
            output_path, _ = QFileDialog.getSaveFileName(self, "Kaydedilecek Video", "", "Video Dosyaları (*.mp4)")
            if output_path:
                self.status_label.setText("Video işleniyor...")
                process_video(self.selected_video, output_path, self.status_label.setText)
            else:
                self.status_label.setText("Kaydedilecek dosya seçilmedi.")
        else:
            self.status_label.setText("Lütfen önce bir video seçin!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
