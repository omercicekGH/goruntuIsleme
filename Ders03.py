# %%
#görüntüyü yeniden boyutlandırma , kaydırma ve döndürme 

import cv2
import numpy as np

img1=cv2.imread("Desktop/goruntuIslemeKurs/1.jpeg")
#resim okurken relative path (yakın yolu) kullansakta oluyor
cv2.imshow("resim1",img1)

print(img1.shape)

#gorseli yeniden boyutlandırma:

img2=cv2.resize(img1, (400,400))

cv2.imshow("resim2",img2)

#gorseli bir katsayiya gore yeniden boyutlandırma:

img3=cv2.resize(img1,None,fx=0.5,fy=0.5)

#none= hedef boyutun kaç piksel olacağını söylemeyeceğim , ölçek katsayılarından hesapla


cv2.imshow("resim3",img3)

#Yeniden boyutlandırmada pikseller interpolasyon yöntemine göre hesaplanır.
#Yani yeni piksel değerleri komşu piksellerden tahmin edilir; küçültmede ise çevre 
#piksellerin ortalaması alınır.

img4=cv2.resize(img1,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)

#en çok kullanılan bu üçü:
#cv2.INTER_CUBIC = Resim büyütmede kullanılır. 16 pikselden destek alır . En yavaş ama en düzgün sonuç veren.
#cv2.INTER_AREA = Resim küçültmede (özellikle çok fazla küçültüyorsak) /alan ortalaması kullanır
#cv2.INTER_LINEAR = Genel Kullanım . 4 pikselin ortalamasını alır

cv2.imshow("resim4",img4)

#görüntü üzerinde yer değiştirme(Translation):
#T veya M isminde matris oluşturarak yaparız
# M = { 1 0 tx
#       0 1 ty } tx= x de kayması gereken piksel sayısı
                #ty= y de kayması gereken piksel sayısını yazarız

translation_matrix=np.float32([
    [1,0,50],
    [0,1,50]
    ])

rows,cols=img1.shape[:2] #img1 in sizeında ki ilk iki parametreyi aldık

img_translation=cv2.warpAffine(img1, translation_matrix , (cols,rows))
#3. parametre yer değiştirme işleminden sonra ki görüntünün istediğimiz boyutları .
#Sütun , satır sırası ile girmek gerekiyor.
#warpAffine() : Görüntüdeki her (x, y) pikseli, belirlediğimiz matris ile çarpılarak 
#yeni koordinatlarına (x', y') taşınır.

#resmimizi boyutlarını değiştirmeden sağa ve aşağıya 50 birim ötelemiş olduk

cv2.imshow("resim5",img_translation)


img_translation2=cv2.warpAffine(img1, translation_matrix , (cols+50,rows+50))

cv2.imshow("resim6",img_translation2)

#resmin boyutunu 50şer birim artırarak resimde ki kırpılma işlemini kaldırdık

#görüntüyü döndürme:
#görüntüyü döndürmek için de bir matrise ihtiyacımız var

#getRotationMatrix2D() : bir görüntüyü döndürmek için ihtiyacımız olan 2x3'lük
#Affine matrisini bizim için otomatik olarak hesaplayan fonksiyondur.

rotation_matrix=cv2.getRotationMatrix2D((cols/2,rows/2),60,0.7)
#parametreler : center, angle, scale
#yine sütun satır sıralaması ile

img_rotation=cv2.warpAffine(img1, rotation_matrix, ((cols,rows)))

cv2.imshow("resim7",img_rotation)

cv2.waitKey()
cv2.destroyAllWindows()



# %% Affine Transformation

#bir görüntüdeki noktaların, çizgilerin ve düzlemlerin yapısal özelliklerini koruyan 
#geometrik bir dönüşüm türüdür. Bu dönüşümün en belirleyici özelliği, orijinal 
#görüntüde birbirine paralel olan doğruların, dönüşümden sonra da paralel kalmasıdır.
#Bu işlem sırasında görüntü ötelenebilir, döndürülebilir, ölçeklenebilir veya eğilebilir

#Bir Affine dönüşüm matrisi oluşturmak için görüntünün orijinal halinden 3 nokta ve 
#bu noktaların hedefteki yeni yerlerini bilmeniz yeterlidir. OpenCV, bu 3 noktanın
#değişimine bakarak 2*3 boyutundaki dönüşüm matrisini otomatik olarak hesaplar.

#Affine = şekli boz ama yapıyı koru

import cv2
import numpy as np

img1=cv2.imread("Desktop/goruntuIslemeKurs/1.jpeg")

cv2.imshow("resim",img1)

rows , cols =img1.shape[:2]

src_points=np.float32([
    [0,0],
    [cols-1,0],
    [0,rows-1]
    ])

#üç notamızı seçtik

#bu üç noktamızı götürmek istediğimiz noktalar için matris:
    
dst_points=np.float32([
    [0,0], #ilk seçtiğimiz piksel aynı yerinde kalsın istedik
    [int(0.6*(cols-1)),0], #ikinci noktamızı sola doğru yüzde 60 kaydırdık
    [int(0.4*(cols-1)),rows-1] #üçüncü noktamızı yüzde 40 sağa kaydırdık
    ])

affine_matrix=cv2.getAffineTransform(src_points, dst_points)
#affine matrisimizi elde ettik şimdi bu matrisi resmimiz ile çarpmalıyız

img_output=cv2.warpAffine(img1, affine_matrix, (cols,rows))

cv2.imshow("resim2",img_output)

cv2.waitKey()
cv2.destroyAllWindows()




# %% Projective Transformation (Perspektif Dönüşümü)


#görüntüyü kamera bakış açısına göre eğip bükebilen, yani 
#perspektifi değiştiren en genel 2D dönüşümlerden biridir
#affine dönüşümden temel farkı paralel çizgiler dönüşümden sonra 
#paralel kalmak zorunda değildir ve 4 nokta referans alınır

import cv2
import numpy as np

img1=cv2.imread("Desktop/goruntuIslemeKurs/1.jpeg")

cv2.imshow("resim",img1)

rows , cols =img1.shape[:2]

#dört noktamızı seçeriz (resmin içindeki sudokunun köşelerini seçtik)
src_points=np.float32([
    [67,57], 
    [419,43], 
    [20,503],
    [474,490]
    ])

#bu dört noktamızı götürmek istediğimiz kordinatları seçeriz:
#resmin köşe noktalarını seçtik (çerçevenin başlayıp bittiği noktaları)    
dst_points=np.float32([
    [0,0],
    [cols-1,0],
    [0,rows-1],
    [cols-1,rows-1]
    ])

#dönüşüm matrisi:

M=cv2.getPerspectiveTransform(src_points, dst_points)    

#resim1 ile dönüşüm matrisimizi işleme soktuk:
img_res=cv2.warpPerspective(img1, M, (cols,rows))

#sudoku kağıdımızın üzerinde bulunduğu arka planı resimden atıp
#sudoku kağıdımızı ekranda karşıdan dik görünecek şekilde ayarladık
cv2.imshow("resim2",img_res)

cv2.waitKey()
cv2.destroyAllWindows()

# %% fare ile köşeleri seçtiğimiz perspektif transformation uygulaması

import cv2
import numpy as np

img1=cv2.imread("Desktop/goruntuIslemeKurs/uyg.jpg")

cv2.namedWindow("img",cv2.WINDOW_NORMAL) #bir görüntü çerçevesi oluşturduk
#pencereyi boyutlandırabilmek için cv2.NORMAL_WINDOW kullandık

cv2.namedWindow("output",cv2.WINDOW_NORMAL)
#çıktı alcağımız pencerede de boyutlandırma yapabilmek için

rows , cols =img1.shape[:2]

#seçtiğimiz dört noktayı götürmek istediğimiz kordinatları seçeriz   
dst_points=np.float32([
    [0,0],
    [cols-1,0],
    [0,rows-1],
    [cols-1,rows-1]
    ])
#seçtiğimiz dört nokta çerçevenin köşelerine gidecek

click_count=0 #fare tıklama sayacı
a=[] #fare ile tıkladığımız kordinatları kaydetmek için değişken

#fare olaylarını işleyecek fonksiyonumuzu oluşturduk
def draw(event,x,y,flags,param) :
    global click_count
    global a
    
    if click_count<4:
        if event ==cv2.EVENT_LBUTTONDBLCLK:
            click_count +=1 #her çift tıklamada sayacı bir artırdık
            a.append((x,y))  #çift tıkladığımız kordinatları a nın içerisine ekledik
    else: #4 çift tıklamadan sonra else gireceği için perspektif dönüşümümüzü burada yaparız
        
        #çift tıklayarak seçtiğimiz dört noktayı src_pointe girdik
        src_points=np.float32([
            [a[0][0],a[0][1]], #sayacın 0. X'i , sayacın 0. Y'si
            [a[1][0],a[1][1]], 
            [a[2][0],a[2][1]],
            [a[3][0],a[3][1]]
            ])
        
        #resmimizi çevirmeden işleme sokması için sırası ile sol üst
        #sağ üst , sol alt , sağ alt şeklinde seçim yapmamız gerekiyor
        
        M=cv2.getPerspectiveTransform(src_points, dst_points)
        #dönüşüm matrisimizi oluşturduk
        
        img_out=cv2.warpPerspective(img1, M, (cols,rows))
        #resmimizi dönüşüm matrisi ile işleme soktuk
        
        cv2.imshow("output",img_out)#oluşturduğumuz resmi gösterdik
        
        click_count=0
        a=[] #sayacı sıfırladık anın içini temizledik
    pass

cv2.setMouseCallback("img", draw)
#fare hareketleriyle işlem yapabilmek için kullandığımız fonksiyon

while(1): #tıklama işlemi yaptığımız sürece resimleri göstermesini
#istediğimiz için döngü içine aldık
    cv2.imshow("img",img1)    
    
    if cv2.waitKey(1)==ord("q"): #q ya basınca döngüden çıkmasını istedik
        break

cv2.destroyAllWindows()


#internette ki kameradan görüntü okumak

# url="http://192.############"

# cam=cv2.VideoCapture(url)

# while cam.isOpened():
#     ret , frame= cam.read()
    
#     cv2.imshow("goruntu",frame)
    
#     if not ret:
#         print("goruntu okunamadi")
        
#     if cv2.waitKey(1)==ord("q"):
#         break        
# cv2.destroyAllWindows()




# %% tresholding

#Eşikleme, bir görüntüyü basitleştirmek ve içindeki nesneleri 
#arka plandan koparıp almak için kullanılır.

#Öncelikle resmi gri tonlamaya (Grayscale) çeviririz. Bu durumda her piksel 0 (Siyah) 
#ile 255 (Beyaz) arasında bir değer alır. Sonra bir "Eşik Değeri" (Threshold) belirleriz,
#resimdeki eşik değerinden yüksek her piksel değerini 255 değerine sabitleriz(beyaz olur) ,
#eşik değerinden küçük her piksel değerini 0 a eşitleriz (siyah yaparız)

#Böylece resimdeki detaylar, gölgeler ve renk geçişleri silinir; 
#geriye sadece nesnenin keskin silüeti kalır.

#Basit (Global) Eşikleme
#Tüm resim için senin belirlediğin tek bir sabit değeri kullanır. Eğer 
#ortamdaki ışık her yerde aynıysa harika çalışır.

import cv2
import matplotlib.pyplot as plt

resim=cv2.imread("Desktop/goruntuIslemeKurs/image_2.jpg",0)
#siyah beyaz okumak için 0 parametresini kullandık

ret , thresh=cv2.threshold(resim, 180, 255, cv2.THRESH_BINARY)
#bu fonksiyon 2 değer dönderir . ret : eşik değeri , thresh : thresholding işlemi yapılmış görüntü
#source ,eşik değeri, max value, type
#max value= eşik değerinin üstünde ki değerleri eşitlemek istediğimiz değer
#eşik değerini 0 alır

#cv2.namedWindow("resim",cv2.WINDOW_NORMAL)

#cv2.imshow("resim",thresh)


def threshold(src,thresh,maxval):
    """
    Bu fonksiyon ile cv2.threshold() fonksiyonu ile yaptığımız
    basit eşiklemeyi yapabiliriz
    src = image
    thresh=0....255
    maxval=0....255    
    """
    img=src.copy() #orjinal resmi bozmadan üzerinde işlem yapabilmek için kopyasını aldık.
    rows,cols=img.shape[:2] #shapede ki ilk iki değeri aldık
    
    for i in range(rows):  #her bir pikseli gezerek belirlediğimiz thresh değerine göre 
        for j in range(cols):  #göre işlem yaptık
            if img[i,j]<thresh:
                img[i,j]=0  #kordinattaki piksel eşikten küçükse sıfırla
            else:
                img[i,j]=maxval #diğer durumlarda maximum değere eşitle
    return thresh,img #thresh değerini ve oluşturduğumuz img i bize dönder

#burada hazır fonksiyonun aynısını yapmış olduk

ret2 , thresh2 =threshold(resim, 150, 250)

#cv2.namedWindow("resim2",cv2.WINDOW_NORMAL)

#cv2.imshow("resim2",thresh2)

#diğer türler (types):
_ , thresh3=cv2.threshold(resim, 180, 255, cv2.THRESH_BINARY_INV) #hafızada yer kaplamasın diye _ değişkenine atadık
#terslenmiş hali : eşik degerinden yüksekleri siyah , düşükleri beyaz yapar
_ , thresh4=cv2.threshold(resim, 180, 255, cv2.THRESH_TRUNC)
#truncate (kesme) : verdiğimiz değerin üstündekileri beyaz yapar , geri kalan piksellerin değeri değişmez(kendi rengini korur)
_ , thresh5=cv2.threshold(resim, 180, 255, cv2.THRESH_TOZERO)
#tozero :eşik değerinin altındakiler siyah,üstündekilerin değeri değişmez 
_ , thresh6=cv2.threshold(resim, 180, 255, cv2.THRESH_TOZERO_INV)
#eşik değerinden yüksekler siyah , eşik değerinin altındakilerin rengi değişmez
#renk değişmese bile hepsinin hali hazırda gri skalada alındığını unutma!

resimler=[thresh,thresh2,thresh3,thresh4,thresh5,thresh6]

#tüm eşik değeri uygulanmış resimleri aynı çerçevede görebbilmek için
#liste haline getirdik , matplotlib kütüphanesini kullandık

basliklar=["binary","binary_fonk_olstrdk","bin_invert","trunc","tozero","tozero_inv"]
#

for i in range(6):    
    plt.subplot(2,3,i+1)
    plt.imshow(resimler[i],"gray") #gri skalaya almayı unutmadık
    plt.title(basliklar[i])

plt.show()

# cv2.imshow("binary_terslenmis",thresh3)
# cv2.imshow("trunc",thresh4)
# cv2.imshow("tozero",thresh5)
# cv2.imshow("tozeroinvert",thresh6)


cv2.waitKey()

cv2.destroyAllWindows()



# %% trackbar ile eşik değerini ayarladığımız Basit Eşikleme Uygulaması


import cv2

resim=cv2.imread("Desktop/goruntuIslemeKurs/image_1.jpg",0)

def nothing(x):
    pass

cv2.namedWindow("resim",cv2.WINDOW_NORMAL) #penceremizi oluşturduk
cv2.namedWindow("threshold_image",cv2.WINDOW_NORMAL)
cv2.createTrackbar("esik", "resim", 0, 255, nothing)
#trackbarımızı oluşturduk


while(1): #okuduğumuz resmin gösterilmesi için döngümüz
    
    thresh = cv2.getTrackbarPos("esik", "resim") #trackbarın değerini okuduğumuz değişken
    
    _ , threshold =cv2.threshold(resim, thresh, 255, cv2.THRESH_BINARY)
    #threshold uygulanmis goruntumuzu olusturduk eşik değerine parametre olarak trackbardan alacağımız değeri girdik. 
    
    cv2.imshow("resim",resim)
    cv2.imshow("threshold_image",threshold)
    
    if cv2.waitKey(1) & 0xff==ord("q"):
        break

cv2.destroyAllWindows()



















































































































































































































































































































































