import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QFormLayout, QLineEdit, QComboBox, QDialog, QMainWindow, QTabWidget
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from layout_colorwidget import Color
from boxes_example import boxes


class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB defect detection")
        self.setMinimumSize(1000, 600)
        self.image = None
        self.processed_image = None
        

        self.setStyleSheet("background-color: white;")
        #main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)


        '''
        widget = QWidget()
        widget.setLayout(image_layout)
        self.setCentralWidget(widget)
        '''
        main_layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()
        self.panel_layout = QHBoxLayout()
        self.button_layout = QVBoxLayout()
        self.defects_layout = QVBoxLayout()
        self.original_label = QLabel("PCB Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setFixedSize(480, 480)

        self.processed_label = QLabel("Image with detected defects")
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setFixedSize(480, 480)


        self.image_layout.addWidget(self.original_label)
        self.main_layout.addLayout(self.image_layout)
        self.image_layout.addWidget(self.processed_label)


        btn1_load = QPushButton("Load Image")
        btn1_load.clicked.connect(self.load_image)
        self.button_layout.addWidget(btn1_load)

        btn2_load = QPushButton("Load Image")
        btn2_load.clicked.connect(self.load_image)
        self.button_layout.addWidget(btn2_load)

        #Legenda z tytułem
        legend_title = QLabel("Legend")
        font = legend_title.font()
        font.setPointSize(15)
        legend_title.setFont(font)
        legend_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.defects_layout.addWidget(legend_title)

        legend = QLabel(" 🔴 - Missing Hole \n 🟢 - Mouse bite \n 🟡 - Open circuit \n 🔵 - Short \n 🟠 - Spur \n 🟣 - Spurious copper")
        font = legend.font()
        font.setPointSize(10)
        legend.setFont(font)
        legend.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.defects_layout.addWidget(legend)

        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addLayout(self.panel_layout)
        self.panel_layout.addLayout(self.button_layout)
        self.panel_layout.addLayout(self.defects_layout)
        self.setLayout(self.main_layout)

  
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            global boxes
            boxes = boxes()
            self.image = cv2.imread(file_name)
            self.show_image(self.image, self.original_label)
            self.boundingboxes(self.image, boxes)
            self.show_image(self.image, self.processed_label)


    def show_image(self, img, label):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def boundingboxes(self, image, boxes):
        colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98)]
        for box in boxes: 
            startpoint = (box[0], box[1])
            endpoint = (box[2], box[3])
            score = box[4]
            label = box[5]
            color = colors[label]
            cv2.rectangle(image, startpoint, endpoint, color, thickness=3)
        
        






  


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec())