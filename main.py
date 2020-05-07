import cv2
import numpy as np

listRows = []
listCols = []
startpointRows = []
startpointCols = []
cropedImg = []

kernel = np.ones((3,3), np.uint8)

img = cv2.imread("test1.png")
empty = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

def thresh_callback(val):
    threshold = val
    lastX = 0
    lastY = 0
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mu = [None] * len(contours)
    for i in range(len(contours)):
        mu[i] = cv2.moments(contours[i])
    mc = [None] * len(contours)
    for i in range(len(contours)):
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
    #Buradan yukarısı moment işlemleri yapıyor moment ile birlikte toplu alanların merkezini buluyor aslında
    i=0
    for c in contours:
    # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)

        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(empty, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(len(contours))
        if i%2==0:
            crop_img = img[y:y+h, x:x+w]
            cv2.imwrite(str(int(i/2))+".jpg", crop_img) 
            cv2.imshow(str(c),crop_img)
        i+=1
        

       

img_erosion = cv2.erode(img, kernel, iterations=3)
#cv2.imshow("1.Adim",img_erosion) #boşlukları doldurduk, ama bu sefer genel siyahlık arttı
img_dilation = cv2.dilate(img_erosion, kernel, iterations=3)
#cv2.imshow("2.Adim",img_dilation) #artan siyahlıkları yok ederek kareleri orjinal hale getirdik

src_gray = cv2.cvtColor(img_dilation, cv2.COLOR_BGR2GRAY) #resmi siyah beyaz hale getirdik çünkü ileride binary'e çevircez.
src_gray = cv2.blur(src_gray, (3,3))

max_thresh = 255
thresh = 100

cv2.imshow("Orjinal",img)
cv2.createTrackbar('Orjinal:', 'Source', thresh, max_thresh, thresh_callback) #burada resmi binary'e çeviriyorduk galiba
thresh_callback(thresh)

cv2.waitKey(0)
