"""
Haar Cascade, özellikle gerçek zamanlı yüz tespiti (face detection) konusunda devrim yaratmış ve 
günümüzde hala yaygın olarak kullanılan makine öğrenmesi tabanlı bir nesne tespit algoritmasıdır.

Derin öğrenme (Deep Learning / YOLO vb.) yöntemleri çıkmadan önce, 
bilgisayarlı görünün tartışmasız kralıydı. Hala tercih edilmesinin sebebi çok hızlı çalışması ve 
çok düşük işlem gücü gerektirmesidir.

Algoritma 4 Temel Kavrama Dayanır
Haar Benzeri Özellikler (Haar-like Features)
Algoritma, görüntünün üzerinde siyah ve beyaz dikdörtgenlerden oluşan şablonlar gezdirir.
Bu şablonların mantığı -> Özellik Değeri = (Beyaz bölgedeki piksellerin toplamı) - (Siyah bölgedeki piksellerin toplamı).
Algoritma ,eğer özellik değeri eşik değerini geçiyorsa nesne vardır der. 

İntegral Görüntü (Integral Image)
Görüntüdeki her piksel değeri, kendisinin sol üstünde kalan tüm piksellerin toplamına eşitlenecek şekilde 
önceden hesaplanıp yeni bir matrise yazılır. Bu sayede devasa bir dikdörtgenin içindeki piksellerin toplamı,
sadece 4 köşe pikselinin değerine bakılarak 1 milisaniyeden kısa sürede bulunur.
 
AdaBoost Algoritması
Bir insan yüzünü tanımlamak için üretilebilecek 160.000'den fazla "Haar özelliği" 
(dikdörtgen şablonu) vardır. AdaBoost, bu devasa havuzun içinden insan yüzünü tespit etmede 
"en iyi iş yapan" birkaç bin özelliği seçerek sistemi hafifletir.

Sınıflandırıcı Şelalesi (Cascade Classifier)
Görüntüdeki her pencereye seçilen o birkaç bin özelliği birden uygulamak yerine, 
özellikleri aşamalara (stage) böler.
Aşama 1: Sadece 2 tane en temel özelliğe (örneğin göz-yanak kontrastı) bakar. E
ğer görüntü bu testi geçemezse, burası kesinlikle yüz değildir der ve 
diğer binlerce özelliği test etmeden o bölgeyi çöpe atar.
Eğer geçerse Aşama 2'ye, Aşama 3'e aktarılır (şelale gibi). 
Bu eleme sistemi, algoritmanın saniyede onlarca kareyi tarayabilmesini sağlar.


//kendi Haar Cascade modelimizi eğitme:
1- Veri Seti Hazırlama: Pozitif ve Negatif Görüntülerden oluşan iki ana klasöre ihtiyacımız var
Pozitif Görüntüler: tespit etmek istediğiniz nesneyi içeren resimler(en az 1000 resim)
Negatif Görüntüler: nesnenin olmadığı arka plan resimleri (pozitifin 2 katı)
(bunu yaptıktan sonra negatif pozitif dosya listesini pythonda oluşturmalıyız )
2. Eğitim Adımları (OpenCV Araçları)
OpenCV, bu işlem için geleneksel olarak terminal üzerinden çalışan yardımcı araçlar sunar:
A. Pozitiflerin Hazırlanması (opencv_annotation)
Resimlerdeki nesnenin koordinatlarını belirleyerek bir .txt dosyası oluşturmanı sağlar.
B. Vektör Dosyası Oluşturma (opencv_createsamples)
Tüm pozitif resimleri ve etiketleri, bilgisayarın anlayacağı tek bir .vec dosyasına dönüştürür.
C. Eğitim Başlatma (opencv_traincascade)
En çok zaman alan kısımdır. Bilgisayar bu aşamada "şelale" (cascade) yapısını kurmaya başlar.

"""
import cv2 #ctrl basılı iken buradaki cv2 yazısına tıklayarak cv2nin modül dosyasını açarız
            #modül dosyasının başlığına sağ tıklayıp show in external file explorer deriz
            #modülün bulunduğu klasörü açarız (opecvnin kurulu olduğu klasörü açtık)
            # burada data klasörünü açarak opencvde ki hazır haar cascade xml dosyaları ile işlem yaparız
            #ben dosyalarda haarcascade yazısını aratarak buldum
            #(opencvde ki hazır modeller) buradan kullanmak istediğimiz modeli kopyalıyoruz
            #modeli kullancağımız dosyanın klasörünü açarız(yani bu dosyanın klasörünü)
            #yine başlığa sağ tıklayıp show in external file ile açabiliriz
            #çalışma yaptığımız dosyanın bulunduğu klasöre , çalışacağımız(kopyaladığımız)
            #haarcascade modelinin xml dosyasını yapıştırırız
            #kaç tane model kullanıp algılama yapıcaksak hepsi için aynısını yaparız
            #biz bu çalışma için yüz ve göz tespiti yapacağımız için 
            #haarcascade_eye.xml ve haarcascade_frontalface_default.xml dosyalarını 
            #bu çalışmanın dosyasının olduğu klasöre kopyaladık
            #kopyaladığımız modeller halihazırda eğitilmiş tespit modelleri
            
            
#şimdi kopyaladığımız bu hazır modelleri çalışmamıza okutmamız(yüklememiz gerekiyor):
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#cv2.CascadeClassifier() bu hazır fonksiyon bize modellerimizin nesnesini oluşturur.
#kullandığımız modelin dosya yolunu bu fonksiyonun içine tanımlarız
#yüz tanımlama için bir cascade modeli tanımladık

#göz için cascade değişkeni:
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")
           
#dosyayı kopyalayıp çalıştığımız yere atmadan işlem yapmak için şunu kullanabiliriz:
#face_cascade = cv2.CascadeClassifier(
#    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#) cv2.data.haarcascades ile modellerin klasör yolunu hazır bulup direkt istediğimiz modeli seçebiliriz
            
img=cv2.imread("C:/Users/omerc/Desktop/Omer/dersler/IMG_0294.JPEG")

img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
faces = face_cascade.detectMultiScale(img_gray,1.3,5)
#kullanacagimiz modeli tanıttığımız değişkeni yazıp . koyduk
#fonksiyon parametreleri img , scaleFactor ,minNeighbors #genellikle 1.05 ile 1.2 arasında alınır
#scaleFactor : küçültme oranı / önce büyük yüz arar sonra görüntüyü küçültür tekrar arar
#minNeighbors:Bir bölgenin nesne sayılması için o bölge etrafında kaç tane aday dikdörtgen 
#bulunması gerektiğinin sayısıdır. Genellikle 3 ile 5 arasında alınır 
#diğer girilebilecek parametreler: minSize , maxSize :min max nesne boyutu (Matris formatında)

#detectMultiScale fonksiyonu, resimde bulduğu her bir nesne için bize nesnenin kordinatlarını içeren
#sırası ile x , y , w(width) , h(height) dan oluşan bir liste döndürür.

for (x, y, w, h) in faces: #for döngüsü kullanarak birden çok yüzü işaretledik
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3) #bulduğumuz yüzleri resmin uzerine diktörgen 
                                                            #olarak cizdik
    #göz tespiti için tüm resimde taramaktansa yüzün olduğu kısımda gözleri aramak daha mantıklı                                                      
    #bu sebeple yüzleri kırparak işlem yapabiliriz
    roi_gray = img_gray[y:y+h,x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray) #parametreleri girmeyerek default bıraktık
    
    for ex , ey, ew , eh in eyes:
        cv2.rectangle(img,(ex+x,ey+y),(ex+ew+x,ey+eh+y),(255,0,0) , 3)
        # Göz koordinatları roi_gray (yüz bölgesi) içinde hesaplandığı için,
        # bunları ana görüntü koordinat sistemine çevirmek amacıyla
        # yüzün başlangıç koordinatları olan x ve y eklenir.
    
cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
            


# %% vücut tespiti uygulaması
#bu kısımda biraz yorgun olduğum için eğitmenden bağımsızgittim
#eğitmende scalefactor ve min neighbours parametrelerini trackbar   
#ile ayarladığımız vücut tespit uygulaması yaptı       
            
import cv2 
import os

#os kullanma sebebim program cascade dosyalarını bulur iken hata veriyordu direkt dosya yollarını
#os kütüphanesiyle tanıttım

cam=cv2.VideoCapture("Desktop/goruntuIslemeKurs/vtest.avi")

klasor = os.path.dirname(os.path.abspath(__file__))

fullbody_cascade  = cv2.CascadeClassifier(os.path.join(klasor, "haarcascade_fullbody.xml"))
lowerbody_cascade = cv2.CascadeClassifier(os.path.join(klasor, "haarcascade_lowerbody.xml"))
upperbody_cascade = cv2.CascadeClassifier(os.path.join(klasor, "haarcascade_upperbody.xml"))

while cam.isOpened():
    
    ret,frame = cam.read()
    #cv2.imshow("goruntuss",frame)         

    if not ret:
        print("done")
        break
    frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    tum_vucutlar = fullbody_cascade.detectMultiScale(frame_gray,1.1,5)
    
    for (x, y, w, h) in tum_vucutlar: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)  
        
        roi_gray = frame_gray[y:y+h,x:x+w]
        
        alt_vucutlar = lowerbody_cascade.detectMultiScale(roi_gray)
        
        for ax , ay, aw , ah in alt_vucutlar:
            cv2.rectangle(frame,(ax+x,ay+y),(ax+aw+x,ay+ah+y),(255,0,0) , 3)
            
        ust_vucutlar = upperbody_cascade.detectMultiScale(roi_gray)
        
        for ux , uy, uw , uh in ust_vucutlar:
            cv2.rectangle(frame,(ux+x,uy+y),(ux+uw+x,uy+uh+y),(255,0,0) , 3)
    
    cv2.imshow("goruntu",frame)         
            
    if cv2.waitKey(5)==ord("q"):
        print("by")
        break
cv2.destroyAllWindows()
cam.release()



# %% K-Nearest Neighbour
"""
basit ama çok güçlü Machine Learning algoritmalarından biridir.
Yeni veri, en çok hangi komşularına benziyor? mantığına dayanır

K-NN Nasıl Çalışır?
Haritalama: Elindeki tüm eğitim verilerini bir düzleme yerleştirir.
Yeni Veri: Ne olduğu bilinmeyen yeni bir veri geldiğinde, onu da düzleme koyar.
Mesafe Ölçümü: Yeni noktanın, haritadaki diğer tüm noktalara olan uzaklığını tek tek hesaplar.
(Genellikle bildiğimiz pisagor/öklid uzaklığı kullanılır).
K Seçimi: En yakın olan "K" adet komşuyu bulur.
Oylama (Çoğunluk Kuralı): Bu K kadar komşunun sınıfına bakar. En çok oyu alan komşuya göre bu yeni 
nesnede bu komşunun sınıfındandır der

Adımları:
Öncelikle K değeri Belirlenir(en önemli adım)
Diğer nesnelerden hedef nesneye olan öklit uzaklıkları hesaplanır.
Uzaklıklar sıralanır ve minumum uzaklığa bağlı olarak komşular bulunur.
En yakın komşu kategorileri toplanır
En uygun komşu kategorisi seçilir

Avantaj dezavtaj:
Matematiksel bir model kurulmadığı için eğitim süresi sıfırdır. Yeni veri eklemek çok kolaydır.
Tahmin yaparken yeni noktanın milyonlarca eski noktaya olan mesafesini tek tek ölçmesi gerektiği için,
devasa veri setlerinde inanılmaz yavaştır. Ayrıca çok fazla RAM tüketir.(eski verileri ezberlediği için)

Çok küçük K : overfitting . Gürültüye çok duyarlı . sınırlar çok giriltili çıkıntılı olur
Çok büyük K : Underfitting . Fazla genelleme yapar. Sınırlar çok düzleşir ve küçük sınıflar yok sayılır

K değerini her zaman tek sayı seçmeliyiz ki oylamada eşitlik çıkmasın!
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Veri Seti Oluşturma:
#25 adet rastgele (x, y) koordinatı oluşturulur (0-100 arası)
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)

#Bu 25 noktaya rastgele 0 veya 1 etiketleri atanır (0=Kırmızı, 1=Mavi)
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

#Görselleştirme için verileri etiketlerine göre ayırıyoruz
red = trainData[responses.ravel() == 0]    # Etiketi 0 olanlar (Kırmızı üçgenler)
blue = trainData[responses.ravel() == 1]   # Etiketi 1 olanlar (Mavi daireler)

#Eğitim verilerini ekrana çizdirme
plt.scatter(red[:,0], red[:,1], s=80, c='r', marker='^', label="Kırmızı-0", alpha=0.4) 
plt.scatter(blue[:,0], blue[:,1], s=80, c='b', marker='o', label="Mavi-1", alpha=0.4)

#TAHMİN EDİLECEK YENİ NOKTAYI OLUŞTURDUK:
#Sınıfını bilmediğimiz yeşil bir kare nokta oluşturuyoruz
new_data = np.random.randint(0, 100, (1, 2)).astype(np.float32)
plt.scatter(new_data[:,0], new_data[:,1], s=80, c='g', marker='s', label="Yeni Nokta", alpha=1)

#K-NN Modelini Oluşturma ve Eğitme
knn = cv2.ml.KNearest_create() #ml =machine learning
#bellekte henüz eğitilmemiş, boş bir KNN nesnesi oluşturduk.

#Verileri satır bazlı (ROW_SAMPLE) olarak modele tanıtıyoruz
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses) #algoritmaya verileri ezberlettik.
#ROW_SAMPLE: Her bir veri örneğinin (noktanın) bir satır olduğunu belirtir.
#COL_SAMPLE :train data sütunda sıralansaydı bunu kullancaktık
#trainData ve responses mutlaka float32 olmalı!

#4. En Yakın komşuları bulma:
#Yeni noktanın etrafındaki en yakın 3 komşuya bakıyoruz
ret, results, neighbours, distance = knn.findNearest(new_data, 5)
#ret: Tahmin edilen sınıf. results: Tahmin sonucu (matris formatında).
#neighbours: En yakın komşuların kimlikleri. dist: Komşulara olan uzaklıklar.

#Sonuçları yazdırma:
print("*" * 40)
print("""
      ret: {}           
      results: {}       
      neighbours: {}    
      distance: {}      
      """.format(ret, results, neighbours, distance))
print("*" * 40)

plt.legend()
plt.show()



# %% Sayı Verileri Eğitim ve Test

import cv2
import numpy as np

img = cv2.imread("Desktop/goruntuIslemeKurs/digits.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#1. Veri hazırlama Aşaması
# Veriyi küçük hücrelere bölme: 50 satır, 100 sütunluk bir ızgara yapısı oluşturur.
# digits.png 2000x1000 pikseldir; her rakam 20x20 piksel olur.
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
x = np.array(cells)

# Eğitim verisi: Her rakamın ilk 90 örneğini al (Toplam 5000 rakamdan 4500'ü eğitim için)
# .reshape(-1,400) -> 20x20'lik resmi 400 piksellik düz bir diziye çevirir.
train = x[:,:90].reshape(-1,400).astype(np.float32)

# Test verisi: Her rakamın son 10 örneğini al (Toplam 500'ü test için)
test = x[:,90:100].reshape(-1,400).astype(np.float32)

# Etiketleri oluşturma (rakamlar için)
k = np.arange(10)
train_responses = np.repeat(k,450).reshape(-1,1) # Her rakamdan 450 adet etiket
test_responses = np.repeat(k,50).reshape(-1,1)   # Her rakamdan 50 adet etiket

# 2. VERİ KAYDETME VE OKUMA
# np.savez: Birden fazla diziyi sıkıştırılmış tek bir .npz dosyasında saklar.
np.savez("knn_data.npz", train_data = train, train_label = train_responses)

# Veriyi geri yükleme
with np.load("knn_data.npz") as data:
    train = data["train_data"]
    train_responses = data["train_label"]

# 3. EĞİTİM VE MODEL DOĞRULAMA
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_responses)

# Test verisiyle modelin başarısını ölçme (K=5 komşuya bakarak)
ret, results, neighbours, distance = knn.findNearest(test, 5)

# Doğruluk (Accuracy) hesaplama
matches = test_responses == results
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / results.size
print("Doğruluk Oranı: ", accuracy)

# 4. MODELİ YÜKLEME VE GERÇEK ZAMANLI TEST
# Önceden eğitilmiş bir model dosyasını (.yml) sisteme yükler.
# Not: Bu dosyanın daha önce cv2.ml.KNearest_save() ile kaydedilmiş olması gerekir.
# knn = cv2.ml.KNearest_load('KNN_Trained_Model.yml')

def test_digit(img):
    # Görüntü iyileştirme:
    img = cv2.medianBlur(img, 21)
    img = cv2.dilate(img, np.ones((15,15),np.uint8))
    
    # Çizilen resmi 20x20 boyutuna (modelin eğitildiği boyut) getir ve düzleştir
    test_img = cv2.resize(img, (20,20)).reshape(-1,400).astype(np.float32)
    
    # Tahmin yap
    ret, results, neighbours, distance = knn.findNearest(test_img, 5)
    
    # Sonucu ekrana yazdır (img2 global resmine yansıtır)
    cv2.putText(img2, str(int(ret)), (100,300), font, 10, 255, 4, cv2.LINE_AA)
    return ret

# 5. ÇİZİM PANELİ (PAINT) AYARLARI
cizim = False # Fareye basılı tutup tutmadığımızı kontrol eder
mod = False   # Çizim modunu (Daire/Dikdörtgen) değiştirir
xi, yi = -1, -1
font = cv2.FONT_HERSHEY_SIMPLEX
img = np.zeros((400,400), np.uint8) # Siyah arka planlı çizim alanı

def draw(event, x, y, flags, param):
    global cizim, xi, yi, mod
    if event == cv2.EVENT_LBUTTONDOWN:   # Sol tık basıldı
        xi, yi = x, y
        cizim = True
    elif event == cv2.EVENT_MOUSEMOVE:   # Fare hareket ediyor
        if cizim:
            if mod:
                cv2.circle(img, (x,y), 10, 255, -1) # Serbest çizim (Dairelerle)
            else:
                cv2.rectangle(img, (xi,yi), (x,y), 255, -1) # Kare kutular çizme
    elif event == cv2.EVENT_LBUTTONUP:   # Fare bırakıldı
        cizim = False
    elif event == cv2.EVENT_LBUTTONDBLCLK: # Çift tıkla ekranı temizle
        img[:,:] = 0

cv2.namedWindow("paint")
cv2.setMouseCallback("paint", draw)

# Ana Döngü
while(1):
    img2 = np.zeros((400,400), np.uint8) # Sonuçların gösterileceği alan
    key = cv2.waitKey(33) & 0xFF
    if key == ord("q"): break
    elif key == ord("m"): mod = not mod # Çizim modunu değiştir
    
    print("Tahmin:", test_digit(img))
    cv2.imshow("paint", img)
    cv2.imshow("result", img2)

cv2.destroyAllWindows()



