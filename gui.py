from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtWidgets import *
from PyQt5.QtGui import (QPixmap)
from functools import partial
from PyQt5.QtCore import Qt

def file_picker(layout):
	image = QFileDialog.getOpenFileName()
	print(image)
	for i in range(len(image)):
		qimg = QPixmap(image[i])
		label = QLabel()
		label.setPixmap(qimg)
		layout.addWidget(label)

app = QApplication([])
app.setStyle('Fusion')
window = QWidget()

line = QFrame();
line.setFrameShape(QFrame.StyledPanel);
line.setFrameShadow(QFrame.Sunken);

h_layout = QHBoxLayout()
v_layout = QVBoxLayout()

image_grid = QGridLayout()

combo_box = QComboBox(line)
combo_box.addItem("LBP")
combo_box.addItem("Método 2")
combo_box.addItem("Método 3")

add_image_button = QPushButton('Add Image',line)
add_image_button.clicked.connect(partial(file_picker,image_grid))

add_classify_button = QPushButton("Classify")

v_layout.addWidget(line)
v_layout.addWidget(combo_box,0,Qt.AlignTop)
v_layout.addWidget(add_classify_button,0,Qt.AlignTop)
v_layout.addWidget(add_image_button,0,Qt.AlignTop)
v_layout.addStretch(0);

h_layout.addLayout(v_layout)
h_layout.addLayout(image_grid)
window.setLayout(h_layout)
window.show()
app.exec_()