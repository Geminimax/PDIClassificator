from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtWidgets import *
from PyQt5.QtGui import (QPixmap)
from functools import partial
from PyQt5.QtCore import Qt

image_labels = []
current_column = 0
current_row = 0
c_count = 4
image_box_size = 256

def file_picker(layout):
    global current_column, current_row
    image = QFileDialog.getOpenFileNames()
    print(image)
    for i in range(len(image[0])):

        qimg = QPixmap(image[0][i])
        # label = QLabel()
        # label.setPixmap(qimg.scaled(128,128,Qt.KeepAspectRatio))
        # layout.addWidget(label, 0, current_column, 2, c_count)
        label = ImageBox()
        label.setPixmap(qimg)
        layout.addWidget(label, current_row, current_column)
        current_column += 1
        if current_column >= c_count:
            current_column = 0
            current_row += 1

        # image_labels.add(label)


class ImageBox(QFrame):
    def __init__(self):
        super().__init__()
        self.class_text = "?"
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.pixmap = QPixmap()
        self.text_label = QLabel("Class : " + self.class_text)
        self.initUI()

    def initUI(self):
        global image_box_size
        self.setFrameShape(QFrame.StyledPanel);
        self.setLineWidth(1)
        self.setMaximumWidth(image_box_size)
        self.setMaximumHeight(image_box_size)
        self.setSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.MinimumExpanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.text_label)

    def resizeEvent(self, QResizeEvent):
        width = self.image_label.width()
        height = self.image_label.height()
        self.image_label.setPixmap(self.pixmap.scaled(width,height,Qt.KeepAspectRatio))

    def setPixmap(self, pmap):
        self.pixmap = pmap
        width = self.image_label.width()
        height = self.image_label.height()
        self.image_label.setPixmap(self.pixmap.scaled(width,height,Qt.KeepAspectRatio))

    def setClassText(self, text):
        self.class_text = text
        self.text_label.setText("Class : " + self.class_text)


app = QApplication([])
app.setStyle('Fusion')
window = QWidget()

menu_frame = QFrame()
menu_frame.setFrameShape(QFrame.StyledPanel)
menu_frame.setLineWidth(1)
menu_frame.setFixedWidth(200)
menu_frame.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.MinimumExpanding)
#menu_frame.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
#gallery_frame = QFrame();
#gallery_frame.setFrameShape(QFrame.StyledPanel);
#gallery_frame.setFixedWidth(480)
#gallery_frame.setFixedHeight(320)


h_layout = QHBoxLayout()
v_layout = QVBoxLayout(menu_frame)

scrollArea = QScrollArea()
scrollArea.setMinimumWidth(image_box_size * c_count)
scrollArea.setMinimumHeight(640)
scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
scrollArea.setWidgetResizable(True)
scrollAreaContent = QWidget()

image_grid = QGridLayout(scrollAreaContent)
image_grid.setAlignment(Qt.AlignTop)
scrollArea.setWidget(scrollAreaContent)

combo_box = QComboBox()
combo_box.addItem("LBP")
combo_box.addItem("Método 2")
combo_box.addItem("Método 3")

add_image_button = QPushButton('Add Image')
add_image_button.clicked.connect(partial(file_picker, image_grid))

add_classify_button = QPushButton("Classify")

# v_layout.addWidget(line)
v_layout.addWidget(combo_box, 0, Qt.AlignTop)
v_layout.addWidget(add_classify_button, 0, Qt.AlignTop)
v_layout.addWidget(add_image_button, 0, Qt.AlignTop)
v_layout.addStretch(0)
v_layout.setAlignment(Qt.AlignVCenter)
# h_layout.addLayout(v_layout)
h_layout.addWidget(menu_frame)
h_layout.addWidget(scrollArea)

window.setLayout(h_layout)
window.show()
app.exec_()
