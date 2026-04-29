# %% resim okuma yazma
import cv2 #bazı kaynaklarda as cv diye ekleniyor

#resim okuma

resim=cv2.imread("C:/Users/omerc/Desktop/goruntuIslemeKurs/kizkulesi.jpg")

#aynı klasörde ise direkt resmin ismini yazabiliriz
#resmi matrise dönderdik

cv2.imshow("resim penceresi", resim)
#resmimizi pythonda açtık
#ilk ifade açılan pencerenin ismi ikinci ifade hangi nesneyi açacağımız

cv2.waitKey(0) #bende resim açılıp duruyordu
#ama hocada resim açılıp hemen kapanıyordu bu yüzden
#resmin açıldığında ekranda durması için 0 parametresiyle
#waitKey fonksiyonunu gönderdi. waitkeyin içi ms cinsinden

cv2.destroyWindow("resim penceresi") 
#her hangi bir tuşa bastığımızda resmin kapanması için
#ekrandayken
#cv2.destroyAllWindows() bir çok resim penceresini kapatmak için

resim2=cv2.imread("C:/Users/omerc/Desktop/goruntuIslemeKurs/kizkulesi.jpg",0)
#0 ile resmi siyah beyaz yaptık
cv2.imshow("resim penceresi 2", resim2)

k =cv2.waitKey(0) #& 0xFF #bu ifade çalışmıyorsa 64 bit bilgisayarda eklememiz gereken kod
print(k)
cv2.destroyWindow("resim penceresi 2") 

#resmi kapatmak için bastığımız tuşun bilgisini bu şekilde alabiliriz
#klavyede ki karakterin ascii tablosunda ki karşılığını bize verir
if k==27:
    print("çıkmak için esc tuşuna basıldı")
elif k==ord("q"): #q tuşuna basarak çıktığımızda buraya girer
    print("q tuşuna basıldı")
    cv2.imwrite("kizkulesigri.jpg",resim2)

#q ya basarak çıktığımızda ikinci resmimizi kaydederek çıkacak
#ord() fonksiyonu karakterin sayısal değerini anlamak için kullandığımız bir fonksiyon
#imwrite fonksiyonu ile oluşturğumuz görseli kaydederiz
#ilk parametre resmin kayıt adı ikinci parametre kaydedeceğimiz nesne
#farklı bir konuma kaydetmek isteseydik
#cv2.imwrite('C:/HedefKlasor/cikti.jpg', resim2)

from matplotlib import pyplot as plt
#görseli yaklaştırma hangi pikselde ne var görmekiçin 
#matplotlib kütüphanesini ekledik 

plt.imshow(resim2,cmap="gray")
#resmimizi figure olarak çağırdık

#opencv görüntüyü bgr olarak okuyor , matplotlib rgb olarak
#okuyor bu sebeple plt ile çağırdığımızda mavimsi bir resim görüyoruz

#renk uzayını gri yaptık
plt.colorbar() #figuredeki renklerin skalasını görmek için ben ekledim

plt.show

#cv2.namedWindow(resim) boş resim çerçevesi oluşturma
cv2.namedWindow("resim_cerceve",cv2.WINDOW_AUTOSIZE)

cv2.imshow("resim_cerceve",resim)
#resim dosyamızı çerçevede açtık
#burada çerçeveyi otomatik olarak resmin boyutuna göre belirleriz
#cv2.namedWindow("resim_cerceve",cv2.WINDOW_NORMAL)
#der isek pencereyi istediğimiz gibi büyültüp küçültebiliriz

#matriste her bir pixele karşılık bir sayı gelir
#sayı değeri renk tonuna göre değişir


# %% 0 ın siyahı 1 in beyazı referans ettiğini göstermek
#amacı ile

import numpy as np

sifir=np.zeros([300,300])

bir=np.ones([300,300])

cv2.imshow("sifir",sifir)

cv2.imshow("bir",bir)

cv2.waitKey(0)

cv2.destroyAllWindows()

# %% video okuma yazma

kamera1 = cv2.VideoCapture(0)
#kameramızı programa tanıttık . 0 parametresi bilgisayara dahili olarak
#bağlı kamerayı referans eder
#farklı bir kamera kullanıcak(harici bir kamera) isek parametreyi integer
#olarak artırarak (1,2,3..) kamerayı bulmalıyız

print(kamera1.get(3)) #genişlik
print(kamera1.get(4)) #uzunluk

#kamera1.get(parametre) ile kameranın özelliklerini öğrenebiliriz
#hangi parametrenin hangi özelliği verdiğini
#opencv.org tutorialsden videocaptureproperties yazarak get() fonksiyonuna bakabiliriz
#bunu opencvde bulunan tüm fonksiyonlar için yapabiliriz
#her bir sayının ifade olarak karşılığıda çalışıyor


kamera1.set(3,320) #genişlikle uzunluğu değiştirdik
kamera1.set(4,240)



if not kamera1.isOpened():  
    print("kamera tanınmadı")
    #program kamerayı tanımadığında aşağıda ki ret kısmında hata almamak için
    exit()


while True:
    ret, frame =kamera1.read()
    #genelde ret ve frame olarak isimlendirilir
    #ret true yada false döndürür . kameradan görüntünün okunup
    #okunamadığı bilgisini verir
    #frame ise videoda ki tek bir fotoğraf bilsini verir
    #video bir çok frameden oluşur
    #sürekli yeni görüntü gelir ve frame her loopta yenilenir
    
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert color
    #goruntuyu okurken griye çevirdik #opencv bgr okuduğu için bgrtogray 
    if not ret:
        print("kameradan goruntu okunamiyor")
        
    cv2.imshow("kamera_pencere",frame)
    
    if cv2.waitKey(1) ==ord("q"): 
        print("goruntu sonlandirildi.")
        break
    
kamera1.release() #koddan çıkınca bile kamera arka planda açık kalmaya devam edecekti
                  #bunun önüne geçmek için release fonksiyonunu kullandık

cv2.destroyAllWindows() # olusturduğumuz "kamera_pencere" penceresini kapatmak için                  
    
#q ya basmadığımız sürece videodan çıkmkıyor


# %% dosya içerisindeki videoyu okumak

cam = cv2.VideoCapture("C:/Users/omerc/Desktop/goruntuIslemeKurs/ornekvideo.mp4")

#bu sefer kamera olarak dosyayı tanıttık .
#dosya çalıştığımız klasörde olsa idi ismini yazmamız yeterli olcaktı

while cam.isOpened():    #yukarıda ki gibi uzatmadık direkt kamera 
                            #açık ise while döngüsüne soktuk
                            
    ret , frame = cam.read()

    if not ret:
        print("kameradan goruntu okunamiyor")
        break
    
    cv2.imshow("goruntu_pencere",frame)
    
    if cv2.waitKey(1)==ord("q"): #q ye basınca videodan çıksın istedik
        print("video kapatildi.")
        break
    
cam.release() 
cv2.destroyAllWindows()


# %% kameradan video kaydetmek için

cam = cv2.VideoCapture(0)

video1= cv2.VideoWriter_fourcc(*"XVID") #normalde video1 yerine fourcc değişkeni tanımlanır

#fourcc =four character code , verinin hangi formatta (codec) sıkıştırıldığını
#belirten 4 bytelık bir tanımlayıcı


#bu fonksiyon içine 4 tane parametre alıyor
#("M","J","P","G") yazsaydık görüntümüzü mp4 olarak kaydedecekti
#(*"XVID") yazarak biz avi dosyası olarak kaydettik
#hem windowsta daha tutarlı hemde memoryde daha az yer kaplıyor

bos_sablon=cv2.VideoWriter("ornek.avi",video1,30.0,(640,480)) #normalde bos_sablon yerine out kullanılır

#videoyu içine kaydedeceğimiz boş şablon oluşturduk
#birinci parametre videoyu kaydetmek istediğimiz isim
#ikinci parametre codec ,3. parametre fps değeri 
#4. parametre videonun genişliği ve uzunluğu

while cam.isOpened():
    
    ret , frame =cam.read()
    
    if not ret:
        print("kameradan goruntu alinamadi")
        
    bos_sablon.write(frame) #bos sablonun üzerine görüntüyü yazdık
    
    cv2.imshow("goruntu", frame) #kameranın görüntüsünü ekranda görebilmek için
    
    if cv2.waitKey(1)==ord("q"): 
        print("video kapatildi.")
        break

cam.release()
bos_sablon() #programdan çıkınca şablonu da kapattık . kaydetmeyi burada bıraktı
cv2.destroyAllWindows()


#videoyu farklı kaydetmek için
#dosya_yolu = r'C:\Kullanicilar\Masaustu\videolar\cikti.avi'
#out = cv2.VideoWriter(dosya_yolu, fourcc, 20.0, (640, 480))
#yapmalıydık

# %%goruntu uzerine geometrik sekil cizmek

import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)

#poizitif int sayılardan oluşan 512*512 ve 3 renk skalasından oluşan
#bir matris oluşturduk (simsiyah bir görsel)


cv2.line(img,(0,0),(511,511),(255,0,0),5)
#gorselde cizgi cizmek icin kullandığımız fonksiyon
#1. parametre hangi nesnenin uzerinde cizgi cizeceğimiz
#2. parametre baslangic x,y kordinatlari
#3. parametre cizginin bitiş kordinatları biz çizgiyi matrisin sonuna kadar goturduk
#4. parametre BGR olarak renk kodları biz tam koyu mavi yaptık
#5. parametre çizginin kalınlığı (piksel olarak) , biz 5 piksek kalınlıkta seçtik



cv2.line(img,(50,400),(400,50),(0,255,0),10)
#sol alttan capraz giden yeşil çizgi oluşturduk

cv2.rectangle(img,(50,50),(300,300),(0,0,255),5)
#diktörgen cizme fonksiyonu
cv2.rectangle(img,(500,500),(410,410),(0,0,255),-1)

#içi boyali diktdörtgen çizme ***


cv2.circle(img,(30,30),(15),(240,240,240),3)
#2. parametre merkez kordinatları
#3. parametre yarıçap

cv2.circle(img,(60,80),20,(240,240,240),-1)

#yay olusturma :
cv2.ellipse(img,(256,256),(100,50),0,0,100,(100,200,255),10)
#3. parametre eksen uzunlukları
cv2.ellipse(img,(450,350),(100,50),0,0,100,(100,200,255),-1)

#çokgen çizimi:
pts=np.array([[20,30],[100,120],[255,255],[10,400]],np.int32)

pts2=pts.reshape(-1,1,2) 
#reshape yapmamızın nedeni polylines() fonksiyonu bizden veriyi
#(nokta sayısı,1,2) formatında ister özellikle büyük verili matrislerde programın
#patlamaması için bu önemli -1 le nokta sayısını pythona hesaplatırız
#1= içinde 1 sütun barındıran 2 kanallı vektörler
#2 = x ve y kordinatı

cv2.polylines(img,[pts2],True,(255,255,255),3)

#false yapsaydik sadece cizgiler olusturcakti (bir kenarini bos birakıcaktı)

#yazi yazmak için

font=cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, "OpenCV", (0,511), font, 3, (240,220,255),2,cv2.LINE_AA)
#3. parametre yazinin sol alt kosesi
#5. parametre boyut
#7. paremetre kalınlık
#8. parametre çizgi tipi
#kalınlık girmeseydik içi boş çizgi alıcaktık
cv2.imshow("resim",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% fare hareketleri

import cv2
import numpy as np

# for i in dir(cv2):
#     if "EVENT" in i:
#         print(i)
# opencv deki event içeren fonksiyonları bulduk

cizim= False #sadece goruntu uzerine tıkladığımız zaman işlem yapabilmek
#için oluşturduğumuz değişken
mod = False #goruntu uzerinde aynı işlemle 2 farklı işlev yapabilmek için
#bu şekilde bir değişken oluşturduk

xi,yi=-1,-1
#kordinatları sabit tanımlamazsak daire çizebilmek için
#mouse sol click yaptıktan sonra fareyi kaydırmamız gerekiyor
#direkt tıkladığımızda daire çizebilmesi için değişken tanımladık
#sabit değişken tanımlamadan çembere x , y girdiğisini verir isek mouse sol tuşa
#basılı kaldığı sürece çizgi gibi çember çizer


def draw(event,x,y,flags,param):
    #print(x,y) bunu yazarsak resmin üzerinde fare ile
    #ile gezerken kordinatları gösterir
    
    # if event==cv2.EVENT_LBUTTONDBLCLK:
    #     cv2.circle(img,(x,y),25,(255,255,255),3)
    # #çift tıkladığımız yere çember çizdik

    global cizim
    global xi,yi 
    global mod
    
    if event==cv2.EVENT_LBUTTONDOWN: #sol buton basılı
        xi,yi=x,y
        cizim=True
        
    if event==cv2.EVENT_MOUSEMOVE: #sol buton basılı iken mouse hareket halinde
        if cizim==True:
            if mod:
                cv2.line(img,(x,y),(x,y),(0,255,0),10)    
            else:
                cv2.circle(img,(xi,yi),25,(100,50,0),2)
            #mod true olduğunda çizgi
            #false olduğunda çember çizecek
        else:
            pass
    if event==cv2.EVENT_LBUTTONUP: #sol buton bırakıldığında
        cizim=False
        pass

#goruntuye sol tikladiğimizda çizgi çizmeye başlayacak sol clicki bıraktığımızda
#çizgiyi bitirecek 
    

img=np.ones((512,512,3),np.uint8)*255 #255 le çarparak beyaz bir resim oluşturduk


cv2.namedWindow("paint")

cv2.setMouseCallback("paint", draw)



while(1):
    cv2.imshow("paint",img)
    if cv2.waitKey(1)==ord("q"):
        break
    if cv2.waitKey(1)==ord("m"):
        mod=not mod 
        
        #klavyede m ye bastığımızda mod tersine dönecek(m ile mod değiştireceğiz)
        
cv2.destroyAllWindows()

# %% Trackbar Kullanımı


def nothing(x): #trackbarı oynattığımızda tackbarin hangi değerde olduğu bilgisini döndüren fonksiyon
    pass



img = np.zeros((512,512,3),np.uint8)

cv2.namedWindow("cerceve")

cv2.createTrackbar("R", "cerceve", 0,255, nothing)

# parametreler = trackbarName, windowName, value, count, onChange

#value baslangic değeri count bitiş değeri
#onchange : fonksiyon

cv2.createTrackbar("G", "cerceve", 0,255, nothing)

cv2.createTrackbar("B", "cerceve", 0,255, nothing)

#RGB nin her biri için bir adet trackbar oluşturduk

cv2.createTrackbar("ON/OFF","cerceve",0,1,nothing)



while(1):
    cv2.imshow("cerceve",img) 
    #goruntumuzu donguye koyduk ki sürekli ekranda kalsın
    
    if cv2.waitKey(1) & 0xFF == 27:
        break #27 esc ye denk geliyor
    #escye basınca döngüden çık dedik

    r=cv2.getTrackbarPos("R", "cerceve")
    g=cv2.getTrackbarPos("G", "cerceve")
    b=cv2.getTrackbarPos("B", "cerceve")
    
    #trackbardan renk bilgisi çekebilmek için 
    #opencvde ki hazır fonksiyon
    #ilk parametre trackbar ismi , ikinci parametre çerçeve ismi
    
    switch=cv2.getTrackbarPos("ON/OFF", "cerceve")
    
    if switch:
        img[:]=[b,g,r]
        #img in tüm piksellerini bgr a eşitledik
        #opencv bgr şeklindeydi sıralama önemli
        #artık track bar ile ekranda ki düz resmin rengini ayarlayabiliriz
    else:
        img[:]=0
    
    #switch ekleyerek switch trackbarı 0 değerinde iken görüntü simsiyah
    #ve goruntu uzerinde oynama yapamıyoruz
    
    
    
cv2.destroyAllWindows()
