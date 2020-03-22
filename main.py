import cv2
import numpy as np

listRows = []
listCols = []
startpointRows = []
startpointCols = []
cropedImg = []

kernel = np.ones((9,9), np.uint8)

img = cv2.imread("test1.png")
empty = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

def thresh_callback(val):
    threshold = val
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    _, contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mu = [None] * len(contours)
    for i in range(len(contours)):
        mu[i] = cv2.moments(contours[i])
    mc = [None] * len(contours)
    for i in range(len(contours)):
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (127,127,127)
        cv2.drawContours(drawing, contours, i, color, 2)
        cv2.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, (50,50,50), -1)
    #Buradan yukarısı moment işlemleri yapıyor moment ile birlikte toplu alanların merkezini buluyor aslında

    print("Alan sayısı",len(contours))

    rowsCounter = 0
    colsCounter = 0
    checkBlue = 0
    checkRed = 0

    for x in range(0,len(contours),2): # Bu for döngüsünde alanların merkezinden başlayarak aşağı ve yana doğru tarama yapıyor eğer siyah dışında bir piksel görürse duruor ve kaç piksel boyunda olduğunu anlıyor
        while True:
            blue = drawing[int(mc[0 + x][1]) + rowsCounter,    int(mc[0 + x][0])  , 0]
            red = drawing[int(mc[0 + x][1]),                   int(mc[0 + x][0]) + colsCounter   , 1]

            #Satır taraması
            if blue < 120 and checkBlue is 0:
                rowsCounter = rowsCounter + 1
            elif blue > 120 and checkBlue is 0:
                listRows.append(rowsCounter)
                rowsCounter = 0
                checkBlue = 1

            #Sütün Taraması
            if red < 120 and checkRed is 0:
                colsCounter = colsCounter + 1
            elif red > 120 and checkRed is 0:
                listCols.append(colsCounter)
                rowsCounter = 0
                checkRed = 1

            if checkRed is 1 and checkBlue is 1:
                checkRed = 0
                checkBlue = 0
                break


    for iter in range(0,(int(len(contours)/2)),1):#Croplama işlemi burada yapılıyor.

        startpointCols.append(int(mc[iter*2][0]-listCols[iter]))
        startpointRows.append(int(mc[iter*2][1]-listRows[iter]))

        cropedImg.append(img[startpointRows[iter]:startpointRows[iter]+(listRows[iter]*2), startpointCols[iter] : startpointCols[iter] + (listCols[iter]*2)])

        cv2.imshow('dr'+str(iter),cropedImg[iter])

   #print(mc[7][1],mc[7][0]) # mc[][0] sütun



img_erosion = cv2.erode(img, kernel, iterations=3)
cv2.imshow("1.Adim",img_erosion) #boşlukları doldurduk, ama bu sefer genel siyahlık arttı
img_dilation = cv2.dilate(img_erosion, kernel, iterations=3)
cv2.imshow("2.Adim",img_dilation) #artan siyahlıkları yok ederek kareleri orjinal hale getirdik

src_gray = cv2.cvtColor(img_dilation, cv2.COLOR_BGR2GRAY) #resmi siyah beyaz hale getirdik çünkü ileride binary'e çevircez.
src_gray = cv2.blur(src_gray, (3,3))

max_thresh = 255
thresh = 100

cv2.imshow("Orjinal",img)
cv2.createTrackbar('Orjinal:', 'Source', thresh, max_thresh, thresh_callback) #burada resmi binary'e çeviriyorduk galiba
thresh_callback(thresh)

cv2.waitKey(0)
