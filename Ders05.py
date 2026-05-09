# %% Görüntüde Kenarları Bulmak
"""
Gradyan, bir resimdeki yoğunluk veya renk değişiminin yönünü ve hızını ifade eder.
Matematiksel olarak bir fonksiyonun türevidir ancak görüntüler dijital (kesikli) 
veriler olduğu için biz bunu pikseller arasındaki fark olarak hesaplarız.

Kenar algılama, görüntüdeki ani parlaklık değişimlerini (gradyanları) bulup 
nesnelerin sınırlarını ortaya çıkarma işlemidir.Çıktı olarak 0 ve 1 lerden oluşan 
binary görüntü verir . Genellikle arka plan siyah , kenar çizgileri beyaz olur.
Kenar algılamada Yüksek Geçiren Filtre kullanılır.(kenarlar yüksek frekanslı içeriklerdir.)

Sobel Filtresi: Gürültüyü azaltmak için hafif bir Gaussian yumuşatma ile türev alma işlemini birleştirir.
Gürültüye karşı dirençli olması için türev alırken hafif bir bulanıklaştırma yapar.

Laplacian: Görüntünün ikinci derece türevini alır. Değişimin başladığı ve bittiği yerleri değil, 
tam değişim noktasını (sıfır geçişini) bulur.Gürültüye karşı çok hassastır,
bu yüzden genellikle önce bir Gaussian Blur uygulanması önerilir.

Canny Kenar Algılama (En Popüler Yöntem)
Canny, tek bir filtre değil, çok aşamalı bir algoritmadır ve "altın standart" olarak kabul edilir.
Adımları:
Gürültü Giderme: Gaussian filtre ile resim yumuşatılır.(5*5 lik bir filtre uygular)
Gradyan Hesaplama: Sobel benzeri bir yöntemle kenar şiddeti bulunur.(görüntünün yoğunluk gradyanını bulur )
Non-Maximum Suppression: Kenar olmayan kalın çizgiler inceltilir, sadece en keskin noktalar kalır. 
Histerezis Eşikleme : İki farklı eşik değeri kullanılır. Güçlü kenarlar tutulur, zayıf olanlar ise
sadece güçlü bir kenara bağlıysa korunur.
Bu adımların hepsini tek bir fonksiyon ile yaparız

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img1=cv2.imread("C:/Users/omerc/Desktop/goruntuIslemeKurs/asd.jpeg",0)

img2=cv2.Sobel(img1, -1, 1,0,ksize=5)
#src, ddepth, dx, dy , ksize
#dx:Dikey çizgileri (soldan sağa değişimleri) yakalar.(yukarıdan aşağıya inen kenarları yakalar)
#dy:Yatay çizgileri (yukarıdan aşağı değişimleri) yakalar.(soldan sağa giden kenarları yakalar)

img3=cv2.Sobel(img1, -1, 0,1,ksize=5)

img4=cv2.Sobel(img1,-1,1,1,ksize=5) #dx ve dy açık iken ortak kenarları algılar.
#(sadece hem x hem de y de değişen kenarları yakalar , örneğin sadece x de değişen kenarlar zayıf görünür)

img5=cv2.Sobel(img1, cv2.CV_64F, 1,0,ksize=5)
#pozitif integer görüntü vermez isek cv2.CV_8U ile sonucumuzu 8 bit pozitif integer sayılara döndeririz

#çok önemli!!
#cv2.CV_64F : Beyazdan siyaha geçişte (255'ten 0'a), gradyan pozitif olur. Ancak 
#siyahtan beyaza geçişte (0'dan 255'e), matematiksel işlem sonucu negatif bir sayı çıkar.
#Eğer uint8 kullanırsan, bu negatif değerler doğrudan 0'a eşitlenir ve resmin yarısındaki
#kenarları kaybedersin. cv2.CV_64F bu negatif sayıların hafızada doğru tutulmasını sağlar.

img6=np.absolute(img5) #görüntünün mutlak değerini alarak tüm değerlerini pozitif yaptık
#mutlak değerini almaz isek beyaz ağırlıklı bir görüntü alıyoruz bu kenar tespiti için kötü bir durum
#kısaca negatif değerleri görünür hale getirdik

img7=np.uint8(img6) #floatı tekrardan integara çevirdik. (resmi standart görüntü dosyası formatına geri dönderdik)

img8=cv2.Sobel(img1, -1, 1,0,ksize=-1)
#ksize= -1 kullanır isek Scharr Filtresini uygular ve özel bir matris kullanır .
#daha iyi performanslı çıktı görüntüleri almamızı sağlar.

img9=cv2.Laplacian(img1, -1) #src , ddepth
#default kernel matrisi vardır

img10=cv2.Laplacian(img1, -1,ksize=5) #kernelı biz de belirleyebiliriz

img11=cv2.Canny(img1, 50, 200) 
#image, threshold1, threshold2 . th1 : min eşik değeri , th2: maksimum eşik değeri

img12=cv2.Canny(img1, 200, 210)
#200 ün altında ki hiçbir şeyi alma 210 nun üstünde ki her şeyi kenar olarak algıla diyoruz
#aradaki değerler üstteki değere bağlanıyorsa (aradaki değerdeki pikselin yanında ki piksel değeri
#max eşik değerinden yüksek ise) kenar olarak al diyoruz .

img13=cv2.Canny(img1, 60, 80)

img14=cv2.Canny(img1, 90, 150)

resimler = [img1, img2, img3,img4,img5,img6,img7,img8,img9,img10,img11,img12,img13,img14]
basliklar = ["img_gray", "sobelX", "sobelY","sobelXY","sobelX_float_neg_pos","sobelX_float_pos",
             "sobelX_float2int","sobelX_-1","laplacian","laplacian_wksize","canny","canny2",
             "canny3","canny4"]

for i in range(len(resimler)):
    
    plt.subplot(4, 4, i + 1)
    
    plt.imshow(resimler[i],"gray")
    plt.title(basliklar[i])
    plt.axis('off') # Eksen çizgilerini ve rakamları gizler

plt.tight_layout() 
plt.show()



# %% Contours Bulma

"""
Contours , görüntüdeki nesnelerin sınırlarını temsil eden nokta kümeleridir.
Kontur arayacağın resim mutlaka siyah-beyaz (Binary) olmalıdır. Zemin tamamen siyah,
aradığın nesneler tamamen beyaz olmalıdır. Bu yüzden kontur bulmadan önce resmi 
mutlaka griye çevirip ya Threshold (Eşikleme) ya da Canny işlemi uygularız.

OpenCV'de bu işlemi cv2.findContours() ile bulup, cv2.drawContours() ile resmin üzerine çizeriz

cv2.findCountours iki değişken dönderir
countours: Bulunan her bir şeklin piksel koordinatlarını içeren liste
hierarchy: Şekillerin iç içe olup olmadığı (hiyerarşi) bilgisi
#hiyerarşi kavramını daha iyi anlayabilmek için 3. kursun 14. dersi 25. dakikadan itibaren izlenmeli

cv2.findCountours() da 2. parametre olarak mode gireriz en sık kullanılan modlar:
(modlar hiyerarşinin yapısını belirlemek içindir)    
cv2.RETR_EXTERNAL: Sadece en dıştaki contourları bulur.
cv2.RETR_TREE: Hem dıştaki hem de içteki tüm contourları bulur ve "bu kontur şu konturun içindedir" 
diye bir soyağacı oluşturur.
cv2.RETR_LIST: Tüm contourları alır ama bir hiyerarşi ilişkisi kurmaz.
cv2.RETR_CCOMP:Bu mod, tüm contourları bulur ama onları sadece iki seviyeli bir hiyerarşiye ayırır.
1. Seviye: Nesnenin dış sınırları (Dış contourlar).
2. Seviye: Nesnenin içindeki boşluklar veya delikler (İç contourlar).
RETR_CCOMP: Eğer amacın sadece nesnelerin dış sınırlarını ve varsa içlerindeki delikleri tespit etmekse 
çok daha hızlı ve basit bir veri yapısı sunar.Çok fazla iç içe geçmiş detayı olan bir görüntüde
hiyerarşi karmaşasından kurtulmak istiyorsan idealdir.


3.parametre olarak method gireriz (contourı oluşturan piksellerin nasıl kaydedileceğini)
cv2.CHAIN_APPROX_NONE: Sınır çizgisindeki tüm piksellerin koordinatlarını kaydeder. Çok hafıza harcar.
cv2.CHAIN_APPROX_SIMPLE: Sınırları sıkıştırır. Örneğin kocaman bir dikdörtgen 
konturu bulduğunda binlerce noktayı kaydetmek yerine, sadece 4 köşe noktasını kaydeder.

drawContours() parametreleri:
((image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset)
contours: kontur listesi, contourIdx kaçıncı kontur çizilecek (-1 hepsi demek)) ,lineType : çizgi fontu
hierarchy:Sadece maxLevel parametresini kullanacaksan bunu vermek zorundasın .
maxLevel : İç içe geçmiş şekillerde hangi seviyeye kadar çizim yapılacağını belirler. (0= sadece şekli çizer içini çizmez)
offset (Kaydırma(0,0)): Çizilecek olan konturların ekrandaki yerini X ve Y yönünde kaydırmak için kullanılır.   
ilk 4 parametreden sonrası isteğe bağlıdır.

"""


import cv2
import numpy as np
from random import randint as rnd #rastgele renk üretmek için kullandık
#contour uygulamamızı trackbarla hsv değerlerini ayarlayarak kamera üzerinde yapacağız

def nothing(x):
    pass

camera = cv2.VideoCapture(0) #kameramızı tanımladık

#trackbar oluşturarak hangi renk nesneleri tepit edeceğimizi trackbar ile otomatik ayarlayacağız

cv2.namedWindow("frame") #trackbar için çerçevemizi oluşturduk

cv2.createTrackbar("Hlow", "frame", 0, 359, nothing) #hsv renk uzayı trackbarlarımızı tek tek oluşturduk
cv2.createTrackbar("HUp", "frame", 0, 359, nothing)
cv2.createTrackbar("Slow", "frame", 0, 255, nothing)
cv2.createTrackbar("SUp", "frame", 0, 255, nothing)
cv2.createTrackbar("Vlow", "frame", 0, 255, nothing)
cv2.createTrackbar("VUp", "frame", 0, 255, nothing)

#görüntü çok kasıyorsa maskemize morfolojik işlem uygulayarak sorunu çözebiliriz
#kernel=np.ones((5,5),np.uint)

font=cv2.FONT_HERSHEY_SIMPLEX #goruntude algilanan sekillerin kordinatlarını yazmak için kullandığımız font

while camera.isOpened():
    _, frame =camera.read()  #kameramızdan görüntüyü aldık
    
    img = frame.copy()
    #contuarlarla orjinal görüntü üzerinde değil kopyasında işlem yapabilmek için 
    #bu sayede renkli görüntü üzerine de direkt şekillerimizi çizebilmiş olacağız
    
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #aldığımız görüntüyü bgr den hsv ye çevirdik
            
    hLow=int(cv2.getTrackbarPos("Hlow", "frame")/2) #bolme islemi yaptigimiz icin floata cevirdil
    hUp=int(cv2.getTrackbarPos("HUp", "frame")/2)    #int kullanildigi icin sonucu integera cevirdik
    sLow=cv2.getTrackbarPos("Slow", "frame")        #trackbardan aldığımız değerleri matrise girmek için
    sUp=cv2.getTrackbarPos("SUp", "frame")         #değişkenlere atadık
    vLow=cv2.getTrackbarPos("Vlow", "frame")
    vUp=cv2.getTrackbarPos("VUp", "frame")
    
    lower = np.array([hLow,sLow,vLow])
    upper=np.array([hUp,sUp,vUp])
    
    mask = cv2.inRange(hsv, lower, upper) #minumum ve maksimum hsv değerlerimizle ,
                                             #kameradan aldığımız hsv görüntüsüne maskemizi uyguladık
    
    #mask=cv2.morphologyEx(mask, cv2.MORP_CLOSE, kernel) 
    #mask=cv2.morphologyEx(mask, cv2.MORP_OPEN, kernel)   #benim kameramda kasma olmadığı için ben uygulamadım.

    res = cv2.bitwise_and(frame,frame,mask=mask)
    #nesnelerin beyaz değil kendi rengini aldığımız bir görüntü oluşturduk(o renk dışında ki her şey hala siyah)
 
    #maskemiz zaten siyah beyaz olduğu için biz contour bulmayı maskemiz ile yapacağız
            
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #(image, mode, method)
    
    #counters kullanarak görüntümüzün üstüne resim çizmek için döngü kullandık.
    
    for i , cnt in enumerate(contours):
        area =cv2.contourArea(cnt) #contours ile aldığımız sınır noktaları ile çizilen kapalı
                                   #alanda toplam kaç adet piksel olduğunu hesaplar 
        if area >50000 or area < 200:  #alan 50000 ile 200 arasında ise işleme soktuk
            continue
        
        x, y, w , h = cv2.boundingRect(cnt) # bulduğumuz şeklin(contourun) sol üst köşesinde ki x,y kordinatını
                                            # genişliğini ve yüksekliğini bize verir
    
        #print(x,y,w,h)
        
        color = (rnd(0,256),rnd(0,256),rnd(0,256)) #drawContours() ' a parametre olarak göndermek için rastgele renk oluşturduk.
                                                #for döngüsünde olduğu için her şekilde rastgele bir renk kullanacağız
        
        #cv2.drawContours(img, contours, i, color,2,cv2.LINE_8,hierarchy,0) 
        #drawContoursu bir değişkene atamadık direkt hedef resim üzerine çizer bu sebeple zaten kopyasını oluşturmuştuk!!!
        
        #drawContours kullanmadan şekil çizebilmek için
        
        try: #try except kullanmamızın sebebi kordinatlar elips oluşturmuyorsa program hata verir
            elips=cv2.fitEllipse(cnt) #bir contour etrafına oturabilecek en uygun elipsi hesaplayan bir OpenCV fonksiyonudur.
            cv2.ellipse(img, elips, color,-1) #oluşturduğumuz elipsi görüntünün içine yerleştirdik.
            #içi dolu elips çizmesi için kalınlığı -1 girdik
        except cv2.error as e:
            print("opps: ",e)
        
        text=f"A:{w*h} | Kord:({x}, {y})" #metin olarak string formatında nesnenin alanını ve kordinatlarını vermesini istedik
        cv2.putText(img, text, (x,y), font, 1, color,2) #boundingRect ile aldığımız kordinatları çıktının üzerine yazmak için
        #(img, text, org:textin ekranda nereye yazılacağını söyler, fontFace, fontScale, color ,thickness)
        
    cv2.imshow("frame",frame)

    cv2.imshow("frame copy",img)
    
    #cv2.imshow("mask",mask)
    
    cv2.imshow("result",res) 
    
    if cv2.waitKey(1) ==ord("q"):
        break

cv2.destroyAllWindows()

camera.release()



# %% Contours ile cismin merkezini bulmak , çizim yapmak
#Çevre Hesabı, kenar ve çokgen tespiti
"""
cv2.moments(), bir contourun momentlerini hesaplar. Görüntü işlemede "moment", o 
şeklin piksellerinin alansal dağılımını özetleyen matematiksel değerler bütünüdür.
Bu fonksiyon bize bir dictionary döndürür ve bu sözlükten 
nesnenin alanı, ağırlık merkezi ve yönelimi gibi çok kritik bilgileri çekebiliriz.
m00 : şeklin toplam alanını verir
m10 ve m01: Şeklin X ve Y eksenlerindeki kütle toplamlarını temsil eder.

cv2.flip():bir görüntüyü belirtilen bir eksen boyunca aynalama (yansıtma) işlemi yapmak için kullanılır.
Özellikle veri artırma (data augmentation) veya ön kamera görüntülerindeki aynalama efektini düzeltmek için 
sıkça başvurulan bir araçtır. //dst = cv2.flip(src, flipCode)

flipCode: 
0 : Görüntüyü X ekseni boyunca çevirir. Yani resmi dikey (üst-alt) olarak takla attırır.
1 : Görüntüyü Y ekseni boyunca çevirir. Bu, bildiğimiz yatay aynalama (sol-sağ) işlemidir.
-1 : Görüntüyü hem X hem de Y ekseni boyunca aynı anda çevirir.(aslında 180 derece döndürmekle aynı sonucu verir).    

"""

import cv2
import numpy as np
from random import randint as rnd
#contour uygulamamızı trackbarla hsv değerlerini ayarlayarak kamera üzerinde yapacağız

def nothing(x):
    pass

camera = cv2.VideoCapture(0) #kameramızı tanımladık

#trackbar oluşturarak hangi renk nesneleri tepit edeceğimizi trackbar ile otomatik ayarlayacağız

cv2.namedWindow("frame") #trackbar için çerçevemizi oluşturduk

cv2.createTrackbar("Hlow", "frame", 0, 359, nothing) #hsv renk uzayı trackbarlarımızı tek tek oluşturduk
cv2.createTrackbar("HUp", "frame", 0, 359, nothing)
cv2.createTrackbar("Slow", "frame", 0, 255, nothing)
cv2.createTrackbar("SUp", "frame", 0, 255, nothing)
cv2.createTrackbar("Vlow", "frame", 0, 255, nothing)
cv2.createTrackbar("VUp", "frame", 0, 255, nothing)

#kameramızın algıladığı nesnenin merkez kordinatlarını boş ekrana çizmek için:
paint=np.ones((480,640,3),np.uint8)*255 #beyaz boş bir sayfa oluşturduk
                                        #3 ile gri skaladan bgr renk kanalına geçirdik

#paint=cv2.flip(paint,1) #görüntüyü ters çevirmek için kullandığımız fonksiyon

while camera.isOpened():
    _, frame =camera.read()  #kameramızdan görüntüyü aldık
    
    #frame=cv2.flip(frame,1)
    
    img = frame.copy()
    #contuarlarla orjinal görüntü üzerinde değil kopyasında işlem yapabilmek için 
    
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #aldığımız görüntüyü bgr den hsv ye çevirdik
            
    hLow=int(cv2.getTrackbarPos("Hlow", "frame")/2) #bolme islemi yaptigimiz icin floata cevirdil
    hUp=int(cv2.getTrackbarPos("HUp", "frame")/2)    #int kullanildigi icin sonucu integera cevirdik
    sLow=cv2.getTrackbarPos("Slow", "frame")        #trackbardan aldığımız değerleri matrise girmek için
    sUp=cv2.getTrackbarPos("SUp", "frame")         #değişkenlere atadık
    vLow=cv2.getTrackbarPos("Vlow", "frame")
    vUp=cv2.getTrackbarPos("VUp", "frame")
    
    lower = np.array([hLow,sLow,vLow])
    upper=np.array([hUp,sUp,vUp])
    
    mask = cv2.inRange(hsv, lower, upper) 
    
    res = cv2.bitwise_and(frame,frame,mask=mask)
     
    #maskemiz zaten siyah beyaz olduğu için biz contour bulmayı maskemiz ile yapacağız
            
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #hiyerarşiyi kullanmayacağız
    
    #counters kullanarak görüntümüzün üstüne resim çizmek için döngü kullandık.
    
    for i , cnt in enumerate(contours):
        area =cv2.contourArea(cnt)
        
        if area >50000 or area < 200:  #alan 50000 ile 200 arasında ise işleme soktuk
            continue
        
        x, y, w , h = cv2.boundingRect(cnt) 
               
        color = (rnd(0,256),rnd(0,256),rnd(0,256)) #drawContours() ' a parametre olarak göndermek için rastgele renk oluşturduk.
                                                #for döngüsünde olduğu için her şekilde rastgele bir renk kullanacağız
        
        cv2.circle(img,(x,y),5,(0,0,225),-1)  #x kordinatlarını görselde gösterebilmek için x ve y merkezli daire oluşturduk
        #algıladığımız nesnenin sol üstüne kırmızı nokta koymuş olduk
        
        # #Algıladığımız nesnenin merkezini bulmak için 1. Yöntem:
        # M=cv2.moments(cnt)
        # center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])) #şeklin merkez kordinatlarını bu şekilde buluruz
        #                                                             #int çevirdik çünkü pikseller ondalıklı olmaz
        # cv2.circle(img,center,5,(0,0,225),-1)  #x kordinatlarını görselde gösterebilmek için x ve y merkezli daire oluşturduk
        # #algıladığımız nesnenin merkezine kırmızı nokta koymuş olduk
        
        #nesnenin merkezini bulmak 2. Yöntem:
        (x2,y2),radius = cv2.minEnclosingCircle(cnt) #bir counterı tamamen içine alan mümkün olan 
                                                     #en küçük daireyi hesaplayan fonksiyondur.
        center2 = (int(x2), int(y2))
        radius = int(radius) #değerlerimizi yine integera çevirdik
        
        cv2.circle(img,center2,radius,(0,255,0),-1)
        #algıladığımız nesnenin tamamını kaplayan yeşil bir daire çizdik
        #radius yerine beş girseydik yukarıda ki gibi yeşil nokta elde edicektik.
        
        cv2.circle(paint,center2,10,color,-1) #nesnemizin merkez noktasını paint ekranına çizdik
        
        #contour çevresi bulmak için diğer bir yöntem:
        #nesnenin sınır çizgisinin toplam uzunluğunu bulmak için fonksiyon:
        perimeter=cv2.arcLength(cnt, True)
        #(curve, close) , close : cisim kapalı mı yoksa çizgi mi
        #arcLength() nesnede ki en geniş uzaklığı alır(ne kadar yanlış algılanan piksel varsa hespini çevre uzuluğu için sayar)
        #(contourda ki tüm girinti çıkıntıları alır)
        #bunu yapmaması için cv2.approxPolyDp fonksiyonunu kullanırız ve bu fonksiyonda bir epsilon değerine ihtiyaç duyar.
        
        #approxPolyDp() =yaklaştırmak . en dış çizgiyi bulmak için kullanılır.
        
        #epsilon=0.1*perimeter #hata payı veya hassasiyet. Bu değer, orijinal contour ile
        #ona yaklaştırılacak sadeleştirilmiş şekil arasındaki maksimum mesafeyi belirler.
        
        epsilon=0.010*perimeter
        
        approx=cv2.approxPolyDP(cnt, epsilon, True) #elindeki binlerce noktadan oluşan konturu contouru,
        #(curve, epsilon, closed)                  # belirlediğimiz epsilon dahilinde daha az noktadan
                                                  #oluşan bir çokgene indirger.
        cv2.drawContours(img, [approx], -1, (0,0,0),15)
        #approx bir liste döndürmez. drawcontours liste olarak algılasın diye kapalı parantez kullandık.
        #siyah kullandık ki farkı görelim
        
        hull=cv2.convexHull(cnt)#contouru dışarıdan çevreleyen ve 
                            #hiçbir iç bükey köşesi olmayan dış bükey kabuğu hesaplar.
        #Bir el düşünelim. Parmak aralarındaki boşluklar iç bükeydir. convexHull 
        #bu boşlukları atlayıp elin dış hatlarını birleştirir:
                    
        cv2.drawContours(img, hull, -1, (255,255,288),8) 
        
        cv2.drawContours(img, contours, i, color,4) 
        
        #approx ile nesnenin köşe sayısını belirleme:(approx sizeın ilk parametresi ile anlarız([5,1,2]=beş kenarlı))
        #bunu yapabilmek için perimeter ı çok çok küçük bir rakamla(0.010 gibi) çarparak epsilon değerini bulmalıyız
        if len(approx)==3:
            cv2.putText(img, "ucgen", (x,y), cv2.FONT_ITALIC, 1, 0,2)
        elif len(approx)==4:
            cv2.putText(img, "dortgen", (x,y), cv2.FONT_ITALIC, 1, 0,2)
        elif len(approx)==5:
            cv2.putText(img, "besgen", (x,y), cv2.FONT_ITALIC, 1, 0,2)
        elif 5<len(approx)<12:
            cv2.putText(img, "cokgen", (x,y), cv2.FONT_ITALIC, 1, 0,2)
        else:
            cv2.putText(img, "daire", (x,y), cv2.FONT_ITALIC, 1, 0,2)
        #bu yöntem kenar ve şekil bulmak için çok ,çok kötü bir yöntem        
                
    cv2.imshow("frame",frame)

    cv2.imshow("frame copy",img)
            
    cv2.imshow("result",res) 
    
    cv2.imshow("paint",paint)
    
    key = cv2.waitKey(5)
    
    if key ==ord("q"):
        break
    
    elif key==ord("e"): #e ye bastığımızda paint ekranımızı temizlesin istedik
        paint[:]=255
    
cv2.destroyAllWindows()

camera.release()


