# %% 
#Pikseller


import cv2

import matplotlib.pyplot as plt


resim=cv2.imread("C:/Users/omerc/Desktop/goruntuIslemeKurs/r.jpg")

#bu resimde ki size = [1080,1920,3]
#sırası ile[height, width,channels]
#height = y ekseni boyunca kaç pikselden oluştuğu
#channels = Blue Green Red 
#[1080,1920] olsaydı gri skalada olucaktı


px=resim[100,100] 
print(px)
#çıktı: [146 193 255] BGR sıralaması ile

px_blue=resim[100,100,0] #resimin 100e 100 pikselinde ki 0 indeksli kanal

print(px_blue)
resim[100,100,0]=255 #0. kanalı yani maviyi 255 e eşitledik

print(px) 
#resmin 100 e 100 kordinatlarındaki renklerin piksel yoğunluğu [255 193 255] oldu

resim[100,100]=[255,255,255]
#resmin 100e 100 piksellerinde ki renk yoğunluğunu bu şekilde de değiştirebiliriz

print(px)

#bu işlemler aslında hızlı değil
#bu sebeple numpy kütüphanesini kullanırız

print(resim.item(100,100,0))
#item fonksiyonu ile 100 e 100 pikselinde ki 0. kanalın(mavinin) değerini okuduk
#item fonksiyonun dezavantajı tek bir değer döndürmesi. Hangi kanalı döndüreceğini illa ki belirtmeliyiz

#yukarıdaki ifadeleri tek piksel üzerinde işlem yapıcaksak kullanırız


print(resim.shape) 

print(resim.size) 
#size=height*width*total channel number

print(resim.dtype)


#uint8 = unsigned integer 8 bit ,2**8=256  (0dan başladığımız için 255e kadar)


#resim_kucuk=cv2.resize(resim,(300,400))
#cv2.imshow("resim",resim_kucuk) 




#%% ROI


#ROI= Region of Image
#Belirli bir alanın kırpılması/ ayrı pencereye alınması
#ROI yapılmasının önemli yanı örneğin göz tespiti yapacağız
#gözü daha kolay bulabilmek için öncelikle yüz tespiti yaparız daha sonra
#yüzün içinde gözü ararız bu hem hızlı çalışmamızı sağlar hem de programın doğrulunu artırır


resim=cv2.imread("C:/Users/omerc/Desktop/goruntuIslemeKurs/r.jpg")
#parametre olarak (dosya,0) ekler isek aldığımız resmi gri yapmış oluruz
#bu grinin üstüne matplotlib rgb modunda renklendirerek resmi verir

#plt.imshow(resim) ##görseli çiz
#burada parametre olarak(resim,"gray") der isek çıktımız tek renk uzayına sahip gri tonunda olur

#plt.show()       ##görseli render et (göster)

#matplotlib rgb şeklinde çalıştığı için mavimsi bir görüntü alıyoruz

#matplotlib i analiz için kullanırız . çerçevde istediğimiz gibi oynayabiliriz

kirpilmis_resim=resim[500:800,500:800]

#xte ve y de 500 den 800 e kadar aldık

plt.subplot(1,3,1)
#1 satır , 2 sütun , 1. resim

#1 satır 2 sütun olunca yan yana oldu 2 satır bir sütun olsaydı alt alta olucaktı
#

plt.imshow(resim)

plt.subplot(1,3,2)

plt.imshow(kirpilmis_resim)

plt.subplot(1,3,3)
resim[100:400,1400:1700]=kirpilmis_resim
plt.imshow(resim)
#kirpilmis resmi xte 100 den 400 e y de 1400 den 1700 e kadar resim'in içine yerleştir demiş olduk

#sütun sayısını 3 e artırdım ve son sütunun içine kırptığımız resmi yerleştirdim

plt.show()

#b,g,r=cv2.split(resim)
#opencvde görseli renk uzaylarına tek tek otomatik ayırmamızı sağlayan 
#fonksiyon
#print(b,g,r)

#plt.imshow(b)
#resmi mavi tonunda çıktı olarak alırız
#plt.imshow(g)
#resmi yeşil tonunda çıktı olarak alırız
#plt.imshow(r)
#resmi kırmızı tonunda çıktı olarak alırız bu resim sarı yesile ağırlıklı bir tona döndü

#b,g,r olarak ayırdığımız bir resmi tekrar 3 kanallı renk tonunda birleştirmek için:
#resim2=cv2.merge((b,g,r)) fonksiyonunu kullanabiliriz (b,g,r) yi birleştirerek resim2yi     
#oluştur demiş oluruz

#split yavaş bir işlemdir bunun yerine mavi renk uzayını şu şekilde elde edebiliriz:

#b = resim[:,:,0] #resmin tüm x ve y piksellerini 0. kanalda al demiş olduk

#resim[:,:,2]=0 der isek resmin tüm piksellerinde ki kırmızı kanalı 0 a eşitle demiş oluruz
#ve kırmızı kanalı resimden kaldırırız (kırmızıyı komple resimden kaldırmış oluruz)




# %% Çerçeve


import cv2

import matplotlib.pyplot as plt

BLUE=[255,0,0]

resim=cv2.imread("C:/Users/omerc/Desktop/goruntuIslemeKurs/indir.png")

replicate =cv2.copyMakeBorder(resim, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
#parametreler = src, top, bottom, left, right, borderType
#src = çerçeve eklemek istediğimiz değişken 
#top left bottom right çerçevenin sınır boyutları
#border type : çerçeve türleri
 
reflect =cv2.copyMakeBorder(resim, 10, 10, 10, 10, cv2.BORDER_REFLECT)
reflect101 =cv2.copyMakeBorder(resim, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
wrap =cv2.copyMakeBorder(resim, 10, 10, 10, 10, cv2.BORDER_WRAP)

#standart renkli çerçeve :
constant =cv2.copyMakeBorder(resim, 10, 10, 10, 10, cv2.BORDER_CONSTANT,value=BLUE)




plt.subplot(2,3,1),plt.imshow(resim,"gray"),plt.title("original")
plt.subplot(2,3,2),plt.imshow(replicate,"gray"),plt.title("replicate")
plt.subplot(2,3,3),plt.imshow(reflect,"gray"),plt.title("reflect")
plt.subplot(2,3,4),plt.imshow(reflect101,"gray"),plt.title("reflect_101")
plt.subplot(2,3,5),plt.imshow(wrap,"gray"),plt.title("wrap")
plt.subplot(2,3,6),plt.imshow(constant,"gray"),plt.title("constant")
#matplotlib rgb şeklinde kullandığı için son resmimizde kırmızı çerçeve olur.

# oluşturduğumuz tüm resimler aynı figurede çıkıcak şekilde ayarladık


plt.show()




# %% Goruntu toplama


import cv2
import numpy as np

x= np.uint8([[250]])
#cv2.add 1 boyutlu dizelerde bazen sorun çıkartır ([250]) yazdığımda sonuc2 yi float 260 olarak veriyordu
#Bellekte 8 bit yer kaplayan, değeri 250 olan ve bir piksel verisi olarak işlenmeye hazır bir sayı oluştur
y=np.uint8([[10]])

sonuc1=x+y

# x+y=260%256
#unsigned integer 8 bit kullandığımız için kapasite 255 te dolar 255 ten sonra sistem başa döner
#görüntü işlemede karşılığı : çok parlak bir resmin parlaklığını biraz daha artırırsak
#bembeyaz olması gereken yerler bir anda simsiyah olur


sonuc2=cv2.add(x,y)

#Görüntü işlemede piksellerin anlamsızca kararmasını istemediğimiz durumlar için bu fonksiyon önemlidir.
#Eğer toplama sonucu üst sınırı aşarsa, sonucu o sınırda sabitler.

print(sonuc1) #çıktı : 4

print(sonuc2) #çıktı :255

#sonuc olarak resim toplama cikarma işlemi yapıcaksak opencv kütüphanelerini kullanmak
#daha doğru sonuç verecektir


img1=cv2.imread("C:/Users/omerc/Desktop/goruntuIslemeKurs/cv2.png")

img2=cv2.imread("C:/Users/omerc/Desktop/goruntuIslemeKurs/d.jpg")

toplam=cv2.addWeighted(img1,0.3,img2,0.7,0)
#parametreler : src1, alpha, src2, beta, gamma
#alpha : ilk resmin yüzde olarak ne kadar kullanılacağı
#beta : ikinci resmin yüzde olarak ne kadar kullanılacağı
#gamma : her bir piksele uygulanan parlaklık katsayısı
# gamma>0 daha aydınlık goruntu
#gamma<0 görüntüyü karartır
#gamma=0 ek parlaklık müdahalesi yapmaz

#yukarıda yaptığımız işlem aslınta
#toplam = img1*0.3+img2*0.7+ 0

cv2.imshow("resim",toplam)

cv2.waitKey(0)

cv2.destroyAllWindows()




# %%bitsel islemler


#Bit düzeyinde işlemler (bitwise), sayıları ikili (binary) olarak ele alıp
#tek tek bitler üzerinde işlem yapmaktır. Görüntü işlemede 
#maskeleme, ROI seçme, logo bindirme gibi yerlerde kullanılır.

# bitwise_and : kesişim
# bitwise_or : birleşim
# bitwise_xor : fark 

import cv2
import numpy as np

#resim 1 in sadece renkli yerlerini alıp resim 2nin sol üst kısmına
#yerleştirmek istiyoruz

img1=cv2.imread("C:/Users/omerc/Desktop/goruntuIslemeKurs/cv2.png")

img2=cv2.imread("C:/Users/omerc/Desktop/goruntuIslemeKurs/r.jpg")


img1_gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

#ilk resmimizi bgr renk uzayından gri renk uzayına çevirdik

cv2.imshow("garka plan siyah gri logo",img1_gray)

ret , maske = cv2.threshold(img1_gray,10,255,cv2.THRESH_BINARY)

#oluşturduğumuz gri resimde 10 pikselden üstünü(siyah olmayanları) beyaza çevirdik
#1. resmimizi siyah beyaz yapmış olduk
cv2.imshow("siyah arka planda siyah beyaz logo",maske)

#resimde beyaz gördüğümüz her yeri 1
#siyah gördüğümüz her yeri 0 gibi düşünmeliyiz

print(img1.shape)
print(img2.shape)

#resim iki de ne kadar bir alan kırpmamız gerektiğini öğrenmek için
#resimlerin boyutlarını öğrendik
#resim2 yi kırpmamızın sebebi resim1i resim 2 de kırptığımız yere yerleştireceğiz
#daha sonra oluşan bu görüntüyü tekrardan resim2de kırptığımız yere koyacağız

x,y,z=img1.shape #resim 1 in değerlerini x y z ye eşitledik

roi=img2[0:x,0:y] #resim 2 yi sol üstten kırparak resim 1 ile aynı boyuta getirdik

cv2.imshow("resim2de kirptigimiz goruntu",roi)

#bunu yapmamzın sebebi oluşturduğumuz resim 1 ile resim 2nin kırpılmış halini
#çarparak resimleri birleştirmek (0'ı her hangi bir yoğunlukla çarparsak 0 , ama 1 i yani beyazı
#her hangi bir yoğunlukla çarparsak çarptığımız rengi alır)

mask_inv=cv2.bitwise_not(maske)

cv2.imshow("beyaz arka planda siyah beyaz logo",mask_inv)

#resim1 in renklerini ters çevirdik (siyah kısmı beyaz , beyaz kısmını siyah yaptık)
#bunu yapmamızın sebebi kırptığımız resime logoyu yerleştirdikten sonra kırptığımız resimde ki
#arka planının değişmesini istememiz sadece logoyu koyduğumuz yerin değişmesini istememiz
#sadece logo siyah arka planı beyaz oldu

img1_bg=cv2.bitwise_and(roi,roi,mask=mask_inv)
#parametre olarak aynı resmi kullanmamızın sebebi fonksiyon aslında iki farklı resmi
#karşılaştırmak için tasarlanmıştır . ama bizim işlemimizde amaç iki resmi karşılaştırmak değil
#resmi bir maske yardımıyla filtrelemektir
#Orijinal görüntüyü korumak istediğimiz için de kendisiyle işleme sok diyerek görüntüyü 
#bozmadan fonksiyona sokarız

#kısaca renkli görüntülerde bitsel olarak and işleminde yanlış sonuç elde etmemek için
#parametre olarak iki kere aynı görüntüyü alırız

#logoyu maske olarak kullandık

#logomuzu kırptığımız yerin üstüne yerleştirdiğimiz resmi oluşturduk(img_bg)

cv2.imshow("kirpilmis goruntu uzerine siyah logo", img1_bg)


#daha sonra logonun üzerinde ki renkleri kullanabilmek için
#logonun orjinal halini ve oluşturduğumuz kırpılmış resmin üzerinde ki siyah logoyu
#topladık. siyahla her hangi bir rengi topladığımızda sonuç olarak o resmi alırız
#siyah sıfır olduğu için etkisiz elemandır

toplam=cv2.add(img1_bg,img1)
#artık kırptığımız resmin üzerine renkli şekilde logomuzu yerleştirmiş olduk
#son olarak bu oluşturduğumuz görseli resim 2nin üzerine yerleştirmeliyiz

cv2.imshow("kirpilmis goruntu uzerine renkli logo",toplam)

img2[0:x,0:y]=toplam

#img2 ye toplamımızı 0 dan x e ve 0 dan y ye kadar eklemiş olduk

#cv2.namedWindow("sol ustte logo bulunan resim2",cv2.WINDOW_NORMAL)
#gorseli gösterince çerçevesinde büyültüp küçültme yapabilmek için önce çerçeve oluşturduk
#daha sonra WINDOW_NORMAL parametresi ile pencereyi kullanıcı tarafından yeniden 
#boyutlandırabilir hale getirdik

cv2.imshow("sol ustte logo bulunan resim2",img2) 


cv2.waitKey(0)

cv2.destroyAllWindows()




# %% Renk Uzayı ve Donusumleri , Renkli Nesne Tespiti


#BGR

#Grayscale : Siyah Sıfır beyaz 255

#HSV (Hue, Saturation, Value) : Renk Tespiti İçin kullanılır 
#çünkü ortamdaki ışık BGRdeki üç değeride aynı anda bozar
#Hue - Öz Renk: Rengin türüdür (Sarı, Kırmızı, Mavi). OpenCV'de 0-179 arası değer alır. (180 derece içinde işlem yaparız)
#Işık değişse de nesnenin H değeri pek değişmez.                #normalde ki h değerini 2 ye bölüp opencvde işleme sokarız
#Saturation: Rengin canlılığıdır (Soluk kırmızıdan, fosforlu kırmızıya).0-255 arası değer alır.
#Value - Parlaklık: Rengin aydınlığıdır (Karanlıktan aydınlığa). 0-255 arası değer alır.


#LAB : Algısal Uzay
#İnsan gözünün renkleri algılama biçimine en yakın ve donanımdan bağımsız uzaydır.
#L (Lightness): Parlaklık (0: Siyah, 100: Beyaz).
#a: Yeşil (-128) ile Kırmızı (+127) arasındaki eksen.
#b: Mavi (-128) ile Sarı (+127) arasındaki eksen.
#İki renk arasındaki farkı ölçmek içinidealdir. Eğer bir ürünün renginin standartlara 
#uyup uymadığını kontrol ediyorsan (delta E hesabı), LAB en güvenilir 

# YCrCb (veya YUV) - Video ve Sıkıştırma Uzayı
# Dijital video yayıncılığı ve JPEG/MPEG sıkıştırma formatlarının temelidir. İnsan 
# gözünün parlaklığa (Işık), renkten daha duyarlı olduğu gerçeğine dayanır.
# Y (Luma): Parlaklık bilgisi.
# Cr (Chroma Red): Kırmızı farkı.
# Cb (Chroma Blue): Mavi farkı.
# Görüntü sıkıştırmada, "Cr" ve "Cb" kanallarından veri atılırken "Y" kanalına dokunulmaz. 
# Böylece dosya boyutu küçülür ama insan gözü bozulmayı pek fark etmez.

import cv2
import numpy as np

# for i in dir(cv2):
#     if "COLOR_" in i:
#         print(i) #opencvde ki renk dönüşümü fonksiyonlarına göz attık


#renk uzaylarında ki dönüşümler belirli matematiksel formüller ile yapılır

#renk tespitinde hsv kullanmamızın sebebi belirli renk aralığında ki değerleri aldığımızda
#bunu rgb renk uzayında yaparsak kırmızının belirli bir tonu için tüm renklerde işem yapmamız gerekir
#hsv renk uzayında ise sadece h kısmında işlem yaparak renk tespiti yapabiliriz

def nothing(x):
    pass

camera = cv2.VideoCapture(0) #kameramızı tanımladık

#trackbar oluşturarak hangi renk nesneleri tepit edeceğimizi trackbar ile otomatik ayarlayacağız

cv2.namedWindow("frame") #trackbar için çerçevemizi oluşturduk

cv2.createTrackbar("Hlow", "frame", 0, 359, nothing)
cv2.createTrackbar("HUp", "frame", 0, 359, nothing)
cv2.createTrackbar("Slow", "frame", 0, 255, nothing)
cv2.createTrackbar("SUp", "frame", 0, 255, nothing)
cv2.createTrackbar("Vlow", "frame", 0, 255, nothing)
cv2.createTrackbar("VUp", "frame", 0, 255, nothing)




while camera.isOpened():
    _, frame =camera.read()  #ret değişkeni ile işimiz olmadığı için _ olarak tanımladık
    
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #aldığımız görüntüyü bgr den hsv ye çevirdik
    
    #rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #rgbye dönüştürülmüş görüntü
    
    hLow=int(cv2.getTrackbarPos("Hlow", "frame")/2) #bolme islemi yaptigimiz icin floata cevirdi
    hUp=int(cv2.getTrackbarPos("HUp", "frame")/2)    #int kullanildigi icin sonucu integera cevirdik
    sLow=cv2.getTrackbarPos("Slow", "frame")
    sUp=cv2.getTrackbarPos("SUp", "frame")
    vLow=cv2.getTrackbarPos("Vlow", "frame")
    vUp=cv2.getTrackbarPos("VUp", "frame")
    
    lower = np.array([hLow,sLow,vLow])
    upper=np.array([hUp,sUp,vUp])

    
    #lower = np.array([110,50,50])
    #tespit etmek istediğimiz nesnenin minumumda olmasını istediğimiz hsv değerleri
    
    #upper=np.array([130,255,255])
    #tespit etmek istediğimiz nesnenin maksimumda olmasını istediğimiz hsv değerleri
    
    #120 derece normalde mavi
    
    mask = cv2.inRange(hsv, lower, upper)
    
    #bu fonksiyonla görüntümüzü bir pikselin rengi 110-130 arasındaysa, doygunluğu 50'den fazlaysa 
    #ve parlaklığı da 50'den fazlaysa o pikseli beyaz yap. Bu şartlara uymayan her şeyi siyah 
    #olucak şekilde ayarladık.
    
    #yani mavi nesneler beyaz bir görüntü verecek geri kalan her şey siyah görüntü verecek
    
    res = cv2.bitwise_and(frame,frame,mask=mask)
    
    #burada mavi nesnelerin beyaz değil mavi görüntü vermesini ayarladık
    
       
    
    #cv2.imshow("orjinal",frame)

    #cv2.imshow("donusturulmus",hsv)
    
    #cv2.imshow("mavi tespit",mask)
    
    cv2.imshow("frame",res)
    
    if cv2.waitKey(1) ==ord("q"):
        break

cv2.destroyAllWindows()

camera.release()

#yeşil renge direkt ulaşabilmek için şu kısayolu izleyebilirdik:
    #yesil=np.unit8([[0,255,0]])
    #hsv_yesil=cv2.cvtColor(yesil,cv2.COLOR_BGR2HSV)
    #lower h ile upper h arasını kaynaklarda -10 ile +10 ile tutmamız gerektiği söylenir
    








