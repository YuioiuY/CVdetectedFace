import cv2, math
from matplotlib import pyplot
import numpy as np

def blur_face(img):
  (h, w) = img.shape[:2]
  dW = int(w / 3.0)
  dH = int(h / 3.0)
  if dW % 2 == 0:
      dW -= 1
  if dH % 2 == 0:
      dH -= 1
  return cv2.GaussianBlur(img, (dW, dH), 0)

img = cv2.imread('source/DD.jpg')
img2 = img.copy()
img3 = img.copy()
img4 = img.copy()

classifier_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
classifier_eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
figures = classifier_face.detectMultiScale(img, scaleFactor=2, minNeighbors=3, minSize=(50, 50))
glasses = cv2.imread('source/gl.png', cv2.IMREAD_UNCHANGED) 


#поиск лица 
for figure in figures:
    x, y, width, height = figure
    center = (x + width // 2, y + height // 2)
    axes = (width // 2, int(height * 0.7))
    cv2.ellipse(img2, center, axes, angle=0, startAngle=0, endAngle=360, color=(0, 0, 255), thickness=2)

    face = img2[y:y + height, x:x + width]
    eyes = classifier_eye.detectMultiScale(face)

    if len(eyes) >= 2:
        
        eyes = sorted(eyes, key=lambda e: e[0])
        x1, y1, w1, h1 = eyes[0]
        x2, y2, w2, h2 = eyes[1]

        glasses_width =  int((x2 + w2 - x1))
        glasses_height = int(0.6 * glasses_width)  

        glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height))

        alpha = math.degrees(math.atan2(y2 - y1, x2 - x1))  
        center = (glasses_width // 2, glasses_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D((center), -alpha, 1.0)
        glasses_resized = cv2.warpAffine(glasses_resized, rotation_matrix, (glasses_width, glasses_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


        glasses_x = x + x1  
        glasses_y = y + y1 - int(0.1 * glasses_height)  

        #кружочки 
        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0) 
            thickness = 3 
            cv2.circle(face, center, radius, color, thickness)
        #очки
        for i in range(glasses_height):
            for j in range(glasses_width):
                if glasses_resized.shape[2] == 4:  
                    alpha = glasses_resized[i, j, 3] / 255.0 
                    for c in range(3):  
                        img4[glasses_y + i, glasses_x + j, c] = (
                            (1 - alpha) * img4[glasses_y + i, glasses_x + j, c] + alpha * glasses_resized[i, j, c]
                        )

        
    #блюр
    face = img3[y:y + height, x:x + width]
    
    blurred_face = blur_face(face)
    eye_mask = np.zeros_like(face)

    for (x_eye, y_eye, w_eye, h_eye) in eyes:
        cv2.rectangle(eye_mask, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (255, 255, 255), -1)

    np.copyto(face, blurred_face, where=eye_mask == 0)
    img3[y:y + height, x:x + width] = face  
    

    


fig, (ax1, ax2, ax3, ax4, ax5) = pyplot.subplots(1, 5, figsize=(15, 8))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.xaxis.set_ticks([])
ax1.yaxis.set_ticks([])
ax1.set_title('Исходное изображение')

ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
ax2.xaxis.set_ticks([])
ax2.yaxis.set_ticks([])
ax2.set_title('Распознанные лица и глаза')

ax3.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
ax3.xaxis.set_ticks([])
ax3.yaxis.set_ticks([])
ax3.set_title('Размытие')

ax4.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
ax4.xaxis.set_ticks([])
ax4.yaxis.set_ticks([])
ax4.set_title('Очки')

ax5.imshow(cv2.cvtColor(glasses, cv2.COLOR_BGR2RGB))
ax5.xaxis.set_ticks([])
ax5.yaxis.set_ticks([])
ax5.set_title('Очки в натуральную велечину')

pyplot.show()
