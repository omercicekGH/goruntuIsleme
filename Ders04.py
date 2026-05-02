# %% Otsu Yöntemi , histogram kavramı

#Otsu yöntemi resmin piksel dağılımına bakar ve en ideal eşik değerini 
#matematiksel olarak kendi bulur.

#otsunun dezavantajı gürültüye(parazite) karşı çok fazla savunmasız

import cv2
import matplotlib.pyplot as plt

img1= cv2.imread("shape_noise.png",0)

#cv2.imshow("resim_gri",img1)

blur=cv2.GaussianBlur(img1, (15,15), 0)
#Gauss filtresi, bir görüntü üzerindeki gürültüyü (noise) azaltmak ve 
#görüntüyü yumuşatmak için kullanılan bir filtredir. İlerleyen derslerde 
#bu konuya ayrıntılı değineceğiz.
#burada sonuç olarak resmi blurlu alırız.

#cv2.imshow("blur",blur)

ret , th=cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
#eşik değeri olarak en düşük piksel değerini , maksvalue olarak en yüksek
#piksel değerini girdik. tür parametresine artı(+) diyerek cv2.THRESH_OTSU ekledik
#artık fonksiyon eşik değerini kendisi hesaplayıp(en uygun değeri) ret değişkenine gönderir.

#cv2.imshow("otsu_threshold",th)
print(ret)

ret2 , bl_th=cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 

#cv2.imshow("blurlu_otsu",bl_th)

#global thresholding:
ret1 , th1=cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY) 

#cv2.imshow("normal_threshold",th1)

#histogram kavramı
#bir resimdeki piksellerin hangi parlaklık değerlerinde (0 ile 255 arası) 
#yoğunlaştığını gösteren bir grafiktir

#grafik eksenleri:
    #X Ekseni (Yatay): Piksel yoğunluğunu temsil eder. Sol taraf 0 (Siyah),
    #sağ taraf ise 255 (Beyaz) değerindedir.
    #Y Ekseni (Dikey): O yoğunluk değerinden resimde toplam kaç adet 
    #piksel olduğunu gösterir.

#Resmin genel karakterini sadece histograma bakarak anlayabiliriz
#Grafik Sola Yığılmışsa: Resim çok karanlıktır (Düşük ışık).
#Grafik Sağa Yığılmışsa: Resim çok aydınlıktır veya patlama yapmıştır.
#Grafik Ortada Toplanmışsa: Resmin kontrastı düşüktür (Görüntü gridir, canlı değildir).
#Grafik Tüm Alana Yayılmışsa: Resim yüksek kontrastlı ve dengelidir.

#hist1=img1.ravel() #img1 matrisimizi tek boyutlu bir diziye çevirdik(tüm piksel değerlerini sırayla yazdık)

#plt.hist(hist1,256) #ikinci parametre kaç değerimiz olduğu (grafiğini çıkartacağımız değerler)
#resmimizin histogramını ekrana çizdirdik

# plt.title("Orjinal Resmin Histogramı")

# plt.xlabel("Piksel Değeri")

# plt.ylabel("Piksel Sayısı")

# plt.show()

plt.subplot(2,5,1),plt.imshow(img1,"gray"),plt.title("orijinal resim")

plt.subplot(2,5,2),plt.hist(img1.ravel(),256),plt.title("orijinal resmin histogramı")

plt.subplot(2,5,3),plt.imshow(blur,"gray"),plt.title("blurlu resim")

plt.subplot(2,5,4),plt.hist(blur.ravel(),256),plt.title("blurlu resmin histogramı")

plt.subplot(2,5,5),plt.imshow(th,"gray"),plt.title("otsu thresholdlu resim")

plt.subplot(2,5,6),plt.hist(th.ravel(),256),plt.title("otsu thresholdlu resmin histogramı")

plt.subplot(2,5,7),plt.imshow(bl_th,"gray"),plt.title("blurlu otsu thresholdun resmi")

plt.subplot(2,5,8),plt.hist(bl_th.ravel(),256),plt.title("blurlu otsu thresholdun histogramı")

plt.subplot(2,5,9),plt.imshow(th1,"gray"),plt.title("global thresholdun resmi")

plt.subplot(2,5,10),plt.hist(th1.ravel(),256),plt.title("global thresholdun histogramı")

plt.tight_layout()
plt.show()

# cv2.waitKey()

# cv2.destroyAllWindows()




# %% Uyarlanabilir Eşikleme (Adaptive Thresholding)


#Eğer resmin bir tarafına güneş vuruyor, diğer tarafı gölgede kalıyorsa(resimde aydınlatma homojen değil ise), 
#tek bir eşik değeri işe yaramaz; gölgeli kısımlar tamamen simsiyah olur. 
#Adaptif eşikleme, resmi küçük karelere böler ve her küçük bölge için farklı bir eşik değeri hesaplar.

import cv2
import matplotlib.pyplot as plt

img1= cv2.imread("bookpage.jpg",0)

#cv2.imshow("resim1",img1)

thresh=cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

#parametreler : src, maxValue, adaptiveMethod, thresholdType, blockSize, C
#blockSize : Eşik değerini hesaplamak için pikselin etrafında ki ne kadarlık 
#bir alanın kullanılacağını belirler (tek(odd) sayıdır,11 ise 11*11 lik bir matris)
#c : sabit çıkarım. Hesaplanan ortalamadan veya ağırlıklı ortalamadan çıkarılan sabit bir sayıdır.
#Genellikle sonucu ince ayar yaparak temizlemek için kullanılır.(görüntüde ki gürültüyü temizlemek için)
#EsikDegeri=YerelOrtalama - C    // C pozitif : resim siyah ağırlıklı(kirli grilikleri siyah yapar) , C negatif : resim beyaz ağırlıklı

#iki ana metod var
#cv2.ADAPTIVE_THRESH_MEAN_C: eşik değerini o pikselin çevresindeki komşu piksellerin 
#aritmetik ortalamasını alarak hesaplar.
#cv2.ADAPTIVE_THRESH_GAUSSIAN_C :komşu piksellerin hepsine eşit davranmaz; merkeze (asıl piksele) 
#yakın olanlara daha fazla, uzak olanlara daha az önem verir.

#adaptif eşikleme sadece görsel sonucu dönderir.ret değeri yok çünkü eşik değeri her piksel için ayrı hesaplanıp kullanılır.

#cv2.imshow("adaptive_mean",thresh)

thresh2=cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 11, 2)

#cv2.imshow("adaptive_gaussian",thresh2)

#global thresholding:
    
ret , th=cv2.threshold(img1, 10, 255, cv2.THRESH_BINARY)

plt.subplot(2,2,1),plt.imshow(img1,"gray"),plt.title("original image")
plt.subplot(2,2,2),plt.imshow(th,"gray"),plt.title("global threshold")
plt.subplot(2,2,3),plt.imshow(thresh,"gray"),plt.title("adp mean threshold")
plt.subplot(2,2,4),plt.imshow(thresh2,"gray"),plt.title("adp gaussian threshold")

plt.tight_layout()
plt.show()

#cv2.waitKey()

#cv2.destroyAllWindows()



# %% Morfolojik Kavramlar ve İşlemler


#Morfolojik İşlemler, resmi şekilsel kusurlardan arındırmak için kullandığımız araç kutusudur.
#Gürültü temizleme , nesne sınırlarını düzeltme ,küçük hataları gidermek için kullanılır.
#Genelde siyah beyaz veya gri görüntülerde kullanılır.

#Kernel (Çekirdek Matris): Resmin üzerinde gezdirilen ve işlemin boyutunu,şeklini 
#belirleyen küçük bir matristir.(Resmin üzerinde ki piksellerde bir merkez seçerek boyutu kadar gezinir 
#ve tüm resmi aynı şekilde gezer. Bir çarpma işlemi uygulamadığı için bu matrisimizi çift alabiliriz 2*2 gibi)

#Erosion : Bir görüntüdeki nesnelerin sınırlarında ki pikselleri kaldırır.
#resimdeki beyaz nesnelerin sınırlarını aşındırarak küçültür.

#Dilation (genişleme) : Bir görüntüdeki nesnelerin sınırlarına piksel ekler.
#Beyaz nesnelerin sınırlarına piksel ekleyerek onları büyütür.

#Opening: Önce erosion uygulanır ardından dilation uygulanır.
#Erozyon, o küçük istenmeyen beyaz noktaları tamamen siler. Ancak asıl nesnemizi de biraz küçültmüş olur. 
#Ardından gelen Genişleme işlemi, tamamen yok olmayan asıl nesnemizi eski orijinal boyutuna geri büyütür.
#nesnenin dışındaki gürültüleri temizlemek için kullanırız

#Closing: Önce dilation Uygulanır ardından erosion uygulanır.
#Nesnenin içindeki siyah boşlukları kapatır ve nesneyi büyütür. 
#Ardından gelen Erozyon, nesneyi orijinal boyutuna geri alır.
#nesnenin içinde ki gürültüleri temizlemek için kullanırız

import cv2
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread("2-dilation.png",0)

#cv2.imshow("original",resim)

_ , resim=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#otsu yöntemi ile resmimizi siyah beyaz yaptık

kernel = np.ones((3,3),np.uint8) #5*5 lik 1lerden oluşan pozitif 256 ya kadar değer alan bir matris oluşturduk.
#kernel matrisi büyüdükçe daha fazla aşındırır veya genişletir. (beyazları aşındırır veya genişletir)

erosion = cv2.erode(resim, kernel) #resmimize erosion uyguladık

#cv2.imshow("erosion",erosion)

dilation=cv2.dilate(resim, kernel) #resmimize dilation uyguladık

#cv2.imshow("dilation",dilation)

erosion2 = cv2.erode(resim, kernel,iterations=3)
dilation2=cv2.dilate(resim, kernel,iterations=3)
#iterations parametresi erezyon veya yayma fonksiyonunu kaç tekrar yapacağını belirlediğimiz parametredir.
#kernel matrisini büyütmek gibi sonuçlar doğurur.

kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT ,(12,12)) #1 lerden oluşan 12*12 matris oluşturmanın diğer yolu
#kernelımız elips , artı veya elmas gibi şekillerde olsun istersek bu fonksiyonu kullanırız
#shape, ksize, anchor . shape: oluşturmak istediğimiz matrisin geometrik şekli . anchor:şeklin merkez noktası(opsiyonel)

erosion3 = cv2.erode(resim, kernel2)
dilation3=cv2.dilate(resim, kernel2)

erosionorg = cv2.erode(img, kernel,iterations=3)
dilationorg=cv2.dilate(img, kernel,iterations=3)

#opening ve closing :

opening=cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel)
closing=cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel)

opening2=cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel,iterations=3)
closing2=cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel,iterations=3)

#diğer morfolojik işlemler

#Top Hat:Orijinal resim ile opening işlemi yapılmış resim arasındaki farktır
#Parlak (beyaz) nesneleri karanlık arka plandan ayıklamak için kullanılır.
#(tophat = orjinal - opening)
tophat=cv2.morphologyEx(resim, cv2.MORPH_TOPHAT, kernel)

#Black Hat:Closing işlemi yapılmış resim ile orijinal resim arasındaki farktır.
#(Closing - Orijinal).Ana nesneden daha koyu olan küçük detayları veya bölgeleri
#belirginleştirmek için kullanılır.
blackhat=cv2.morphologyEx(resim, cv2.MORPH_BLACKHAT, kernel)

# Morfolojik Gradyan: Genişleme ile Erozyon arasındaki farkı alır (Genişleme - Erozyon).
#Nesnenin sadece dış hattını bulmak için harikadır.
# Nesnenin içi boşalır, geriye sadece sınır çizgileri kalır.
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

plt.subplot(4,5,1),plt.imshow(img,"gray"),plt.title("gray scale img")
plt.subplot(4,5,2),plt.imshow(resim,"gray"),plt.title("black white img")
plt.subplot(4,5,3),plt.imshow(erosionorg,"gray"),plt.title("gray img erosion3x")
plt.subplot(4,5,4),plt.imshow(dilationorg,"gray"),plt.title("gray img dilation3x")
plt.subplot(4,5,5),plt.imshow(erosion,"gray"),plt.title("bw erosion")
plt.subplot(4,5,6),plt.imshow(dilation,"gray"),plt.title("bw dilation")
plt.subplot(4,5,7),plt.imshow(erosion2,"gray"),plt.title("bw erosion3x")
plt.subplot(4,5,8),plt.imshow(dilation2,"gray"),plt.title("bw dilation3x")
plt.subplot(4,5,9),plt.imshow(erosion3,"gray"),plt.title("bw erosion kernel4x")
plt.subplot(4,5,10),plt.imshow(dilation3,"gray"),plt.title("bw dilation kernel4x")
plt.subplot(4,5,11),plt.imshow(opening,"gray"),plt.title("opening")
plt.subplot(4,5,12),plt.imshow(closing,"gray"),plt.title("closing")
plt.subplot(4,5,13),plt.imshow(opening2,"gray"),plt.title("opening3x")
plt.subplot(4,5,14),plt.imshow(closing2,"gray"),plt.title("closing3x")
plt.subplot(4,5,15),plt.imshow(tophat,"gray"),plt.title("tophat")
plt.subplot(4,5,16),plt.imshow(blackhat,"gray"),plt.title("blackhat")
plt.subplot(4,5,17),plt.imshow(gradient,"gray"),plt.title("gradient")


plt.tight_layout()
plt.show()

#cv2.waitKey()

#cv2.destroyAllWindows()



# %% Görünmezlik Pelerini Uygulaması

#chroma keyde yeşil kullanılmasının sebebi insan vücudunda yeşil renk tonu bulunmamasıdır

import cv2
import numpy as np

cam=cv2.VideoCapture(0)

lower=np.array([52,109,45]) #bende ki yeşil sweatshirtün minumum hsv değerleri
upper=np.array([94,184,88]) #maksimum hsv değerleri (Ders 02 deki renkli nesne tespiti uygulamasından aldım)
                            #opencvde ki h değeri 180 derece üzerinden olduğu için ikiye böldük

_,background = cam.read()
#nesnenin arkasında gözükecek arka planın boş görüntüsünü background olarak okuduk

kernel=np.ones((7,7),np.uint8)  #close için yani yeşil nesnede ki gürültüler için
kernel2=np.ones((7,7),np.uint8) #open için yani dışardaki gürültüler için
kernel3=np.ones((5,5),np.uint8)  #maskeyi büyütmek için
#görüntüyü büyütmemizin sebebi yeşil nesnenin kenarlarınında kaybolmasını istememiz.(nesne büyür arka planı kırpar)

while(cam.isOpened()):
    
    _ , frame = cam.read()
    
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #kamerada ki yeşil rengi algılayabilmek için
                                              #kameradan okuduğumuz görüntüyü hsv renk uzayına çevirdik.
                                              #çünkü hsv uzayında renk tespiti çok daha kolay
        
    mask =cv2.inRange(hsv,lower,upper)  #sadece yeşil rengi okuyacak maskemizi hazırladık
    
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) #görüntü üzerinde ki parazitleri azaltmak için 
                                                          #maskemizi morfolojik işleme soktuk
    mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)  #hem close hem de open ile işleme sokarak 
                                                         #hem yeşil nesnenin içinde ki hemde arka planda
                                                         #parazitleri temizledik                                                                                        
    mask=cv2.dilate(mask, kernel3,iterations=2) #görüntümüzü büyütmek için kullandık
    
    mask_not=cv2.bitwise_not(mask)  #maskede ki beyazları siyah , siyahları beyaza çevirdik (maskeyi tersledik)
                                    #bu(mask_not) orijinal görüntü için gerekli    
    bg=cv2.bitwise_and(background,background,mask=mask) #sadece yesil sweatshirtin oldugu kısmı renkli alabilmek için
                                                        # arka planımızı maske ile filtreledik       
    fg=cv2.bitwise_and(frame,frame,mask=mask_not) #sadece arka planın renkli olduğu görüntüyü oluşturduk
    
    #bg ve fg görüntülerimizi birleştirerek işlemimizi tamamlayabiliriz.
    
    dst=cv2.addWeighted(bg, 1, fg, 1, 0)
    
    dst=np.hstack((frame,dst)) #numpy kütüphanesinde ki görüntüleri yan yana birleştiren fonksiyon
                                #vstack dikey birleştirir    
    #cv2.imshow("orijinal",frame)
    #cv2.imshow("mask",mask)
    cv2.imshow("dst",dst)
    
    if cv2.waitKey(1)& 0xFF ==ord("q"):
        cam.release()
        break

cv2.destroyAllWindows()



# %% Görüntü Bulanıklaştırma  , keskinleştirme , Filtre Oluşturma

#Görüntü filtreleme, bir resmi yumuşatmak, keskinleştirmek veya kenarlarını belirginleştirmek 
#için kullanılan temel bir tekniktir. Burada da bir Kernel (çekirdek matris) resmin üzerinde gezinir;
# ancak bu sefer pikselleri sadece "var/yok" diye kontrol etmek yerine, onları belirli 
#matematiksel katsayılarla çarparak yeni değerler hesaplar.

#Bulanıklaştırma (Smoothing / Blurring):Görüntüdeki azaltmak veya detayları gizlemek için kullanılır.

#Ortalama Bulanıklaştırma: Kernel alanındaki tüm piksellerin ortalamasını alır.
#Gaussian Bulanıklaştırma: Merkeze yakın piksellere daha fazla ağırlık verir.(komşuluk değerine dayanır)
#Doğal bir bulanıklık sağlar ve gürültü gidermede çok popülerdir.
#Median Bulanıklaştırma: Kernel alanındaki pikselleri sıralar ve ortadaki (medyan) değeri seçer. 
#"Tuz ve biber" (siyah-beyaz noktacıklar) tarzındaki gürültüleri yok etmekte en iyi yöntemdir.

#Keskinleştirme (Sharpening)
#Resimdeki kenarları ve detayları daha belirgin hale getirir. Bu işlem aslında resmin 
#ani renk değişimlerini güçlendirir.

import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("1.jpg")

#Ortalama Bulanıklaştırma için kernel matrisimiz:
ort_bulnk_kernel=np.ones((7,7),np.float32)/49 #Bir filtrenin matrisindeki tüm değerlerin toplamı 1 olmalıdır. 
#Eğer bölme işlemi yapmazsak, her pikselin değeri çevresindekilerin toplamıyla çarpılır ve 
#sonuçta bembeyaz, patlamış bir görüntü elde ederiz. Bölme işlemi, görüntünün genel parlaklığını korur.
#Bölme işlemi yapacağımız için bu veri tipi hassasiyet açısından önemlidir. bu yüzden float aldık

dst = cv2.filter2D(img, -1, ort_bulnk_kernel)
#src, ddepth, kernel . ddepth(desired depth): bu derinlik renk kanalı değil .Çıkış görüntüsünün (filtrelenmiş resmin)
#veri tipini ve derinliğini belirler. -1 =sonuç görüntüsü orijinal resimle aynı derinliğe sahip olur.(float32den uint8 e çevirmiş olduk)

#bu kendi kernelımızı olusturup isleme sokmak yerine opencvde otomatik ortalama bulanıklastıran fonksiyon var:
dst2=cv2.blur(img,(9,9)) #cv2.blur() fonksiyonunda çift sayı matris kullanabiliriz(2*2 gibi)

#gaussian blur
dst3= cv2.GaussianBlur(img, (9,9), 0)
#src , kernalsize, sigmaX , sigmaY , borderType
#Standart Sapma - X Yönü: Gaussian çan eğrisinin yatay yöndeki genişliğini (yayılımını) belirler.
#0 yazarsak, OpenCV ksize değerine bakarak bizim için otomatik bir sigma değeri hesaplar.
#sigmaY:Genellikle boş bırakılır veya 0 verilir. Bu durumda otomatik olarak sigmaX değerine eşitlenir.
#borderType:Filtreleme sırasında resmin kenarlarındaki piksellerin nasıl işleneceğini belirler. Genellikle varsayılan değerde bırakılır.

#median blur 
dst4= cv2.medianBlur(img, 9)
#src , ksize

#keskinleştirme: OpenCV'de keskinleştirme işlemi için hazır bir fonksiyon yoktur. 
#Bunun yerine kendi kernelimizi tanımlayıp cv2.filter2D() fonksiyonunu kullanırız.

filter=np.array([[-1,-1,-1],
                 [-1,9,-1],
                 [-1,-1,-1]]) #keskinleştirmede yaygın kullanılan bir kernel matrisi
#matrisin ortasındaki sayı pozitif
#ortadaki sayı küçüldükçe görüntü kararır. büyüdükçe görüntü parlaklaşır

dst5=cv2.filter2D(img, -1, filter) #-1: Çıkış görüntüsü orijinaliyle aynı derinlikte (8-bit) olsun
#ddepth : type of array .

#mexican hat filter (Laplacian of Gaussian)
#iki farklı filtrenin tek bir matriste vücut bulmuş halidir.
# Gaussian Blur (Bulanıklaştırma) ve Laplacian (Kenar Bulma)

kernel=np.array([[0,0,-1,0,0],
                 [0,-1,-2,-1,0],
                 [-1,-2,16,-2,-1],
                 [0,-1,-2,-1,0],
                 [0,0,-1,0,0]])

dst6=cv2.filter2D(img,-1,kernel)

#rastgele filtre oluşturmak:

kernel2=np.array([[-5,-1,1],
                 [-2,15,-1],
                 [-1,-6,-1]])

dst7=cv2.filter2D(img,-1,kernel2)

# #instagram sepia filtresi(eskitme görünümü)

# b,g,r=cv2.split(img) #renklerimizi ayırdık 

# r_new=r*0.393+g*0.769+b*0.189 #Kırmızı kanala, orijinal kırmızının yanı sıra yeşil ve maviden de belirli oranlarda ekleniyor.
# g_new=r*0.349+g*0.686+b*0.168
# b_new=r*0.272+g*0.534+b*0.131

# r_new = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255).astype(np.uint8)
# g_new = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255).astype(np.uint8)
# b_new = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255).astype(np.uint8)
#matplotlib ile görüntüleyebilmek için bu formatta yazmamız gerekir

# dst8=cv2.merge([b_new,g_new,r_new])#kanalları birleştirerek tekrar 3 kanallı bir resim haline getiriyor
#biz bunu kullanmadık yerine bunu matris haline getirip kullandık

kernel3=np.array([[0.131, 0.534, 0.272],  
                  [0.168, 0.686, 0.349],  
                  [0.189, 0.769, 0.393]])

dst8=cv2.transform(img,kernel3)
# cv2.transform() kernel3 matrisini her pikselin renk kanallarına ayrı ayrı uygular

#kabartma filtresi:
#görüntüdeki pikseller arasındaki yoğunluk farklarını kullanarak sanki resim bir
#yüzeye kazınmış veya kağıt üzerinde kabartılmış gibi 3D bir derinlik algısı yaratan bir tekniktir.
kernel4=np.array([[0,1,0],
                 [0,0,0],
                 [0,-1,0]])
#kernelin bir tarafındaki değerler negatif, diğer tarafındaki değerler pozitif

dst9=cv2.filter2D(img,-1,kernel4) +50 #görselin daha aydınlık olması için +30 ekledik

cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow("img",img)

resimler = [dst, dst2, dst3, dst4, dst5,dst6,dst7,dst8,dst9]
basliklar = ["Ortalama Bulaniklastirma", "Ortalama Bulaniklastirma Oto", "Gaussian Blur", "Median Blur",
             "Keskinlestirme","Mexican Hat","Rastgele Filtre","Sepya","kabartma"]

for i in range(len(resimler)):
    
    plt.subplot(3, 3, i + 1)
    
    # OpenCV'nin BGR formatını Matplotlib'in RGB formatına çeviriyoruz
    resim_rgb = cv2.cvtColor(resimler[i], cv2.COLOR_BGR2RGB)
    
    plt.imshow(resim_rgb)
    plt.title(basliklar[i])
    plt.axis('off') # Eksenleri (x ve y sayılarını) gizler, daha temiz görünür

plt.tight_layout()
plt.show()

cv2.waitKey()

cv2.destroyAllWindows()

