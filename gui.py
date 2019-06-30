from PyQt5.QtWidgets import *
from PyQt5.QtGui import (QPixmap)
from functools import partial
from PyQt5.QtCore import Qt
from skimage import io
import classifier
from classifier import extract_haralick,extract_lbp,extract_colorHist,multiple_images_predict

unclassified_labels = []
current_column = 0
current_row = 0
c_count = 4
image_box_size = 256
image_box_separator_size = 5
def file_picker(layout):
    global current_column, current_row
    image = QFileDialog.getOpenFileNames()
    print(image)
    for i in range(len(image[0])):

        # label = QLabel()
        # label.setPixmap(qimg.scaled(128,128,Qt.KeepAspectRatio))
        # layout.addWidget(label, 0, current_column, 2, c_count)
        label = ImageBox()
        label.setImage(image[0][i])
        layout.addWidget(label, current_row, current_column)
        current_column += 1
        if current_column >= c_count:
            current_column = 0
            current_row += 1
            
        unclassified_labels.append(label)


def mock_svm(images):
    #Só pra testar enquanto não tem a svm mesmo
    classes = []
    for image in images:
        classes.append("Adler")
    return classes

def clear_layout(layout):
    global unclassified_labels,current_column,current_row
    for i in reversed(range(layout.count())): 
        layout.itemAt(i).widget().setParent(None)
    unclassified_labels = []
    current_column = 0
    current_row = 0
    
def classify(method_combo_box):
    method =  method_combo_box.currentText()
    print("Classifiying using : " + method)
    
    images = []
    for label in unclassified_labels:
        images.append(io.imread(label.image_path))
    
    #Mudar esse metodo
    classes = multiple_images_predict(images,method)
    
    for i in range(len(unclassified_labels)):
        unclassified_labels[i].setClassText(classes[i])


class ImageBox(QFrame):
    def __init__(self):
        super().__init__()
        self.image_path = ""
        self.class_text = "?"
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.pixmap = QPixmap()
        self.text_label = QLabel("Class : " + self.class_text)
        self.initUI()

    def initUI(self):
        global image_box_size
        self.setFrameShape(QFrame.StyledPanel);
        self.setFrameShadow(QFrame.Sunken)
        self.setLineWidth(1)
        self.setMinimumWidth(image_box_size)
        self.setMaximumHeight(image_box_size)
        self.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.MinimumExpanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.text_label)

    def resizeEvent(self, QResizeEvent):
        width = self.image_label.width()
        height = self.image_label.height()
        self.image_label.setPixmap(self.pixmap.scaled(width,height,Qt.KeepAspectRatio))

    def setImage(self, path):
        self.image_path = path
        self.pixmap = QPixmap(path)
        width = self.image_label.width()
        height = self.image_label.height()
        self.image_label.setPixmap(self.pixmap.scaled(width,height,Qt.KeepAspectRatio))

    def setClassText(self, text):
        self.class_text = text
        self.text_label.setText("Class : " + self.class_text)


app = QApplication([])
#app.setStyle('Fusion')
window = QWidget()
window.title = "Image Classifier"

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
scrollArea.setMinimumWidth((image_box_size * c_count) + ((c_count + 2) * image_box_separator_size))
scrollArea.setMinimumHeight(640)
scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
scrollArea.setWidgetResizable(True)
scrollAreaContent = QWidget()

image_grid = QGridLayout(scrollAreaContent)
image_grid.setAlignment(Qt.AlignTop)
image_grid.setSpacing(image_box_separator_size)
scrollArea.setWidget(scrollAreaContent)

combo_box = QComboBox()
combo_box.addItem(classifier.LBP)
combo_box.addItem(classifier.HARALICK)
combo_box.addItem(classifier.COLOR_HIST)

add_image_button = QPushButton('Add Images')
add_image_button.clicked.connect(partial(file_picker, image_grid))

classify_button = QPushButton("Classify")
classify_button.clicked.connect(partial(classify, combo_box))

clear_button = QPushButton("Clear images")
clear_button.clicked.connect(partial(clear_layout, image_grid))

# v_layout.addWidget(line)
v_layout.addWidget(combo_box, 0, Qt.AlignTop)
v_layout.addWidget(classify_button, 0, Qt.AlignTop)
v_layout.addWidget(add_image_button, 0, Qt.AlignTop)
v_layout.addWidget(clear_button, 0, Qt.AlignTop)
v_layout.addStretch(0)
v_layout.setAlignment(Qt.AlignVCenter)
# h_layout.addLayout(v_layout)
h_layout.addWidget(menu_frame)
h_layout.addWidget(scrollArea)

window.setLayout(h_layout)
window.show()
app.exec_()
