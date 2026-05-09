# %% Hough Transform ,Standard Hough Transform
"""
Görüntü işlemede belirli geometrik şekilleri (özellikle doğru ve çemberleri)
matematiksel olarak tespit etmek için kullanılan en güçlü tekniklerden biridir.

Normalde bir doğruyu y = mx + b olarak biliriz. Ancak dikey doğrularda eğim (m) 
sonsuz olduğu için bilgisayar bunu hesaplayamaz. Bu yüzden Hough Transform, 
doğruları kutupsal koordinat sisteminde (rho,theta) temsil eder.
rho = x*cos(theta) + y*sin(theta)
rho: Orijinden doğruya olan en kısa (dik) mesafe.
theta: Bu dikmenin x ekseniyle yaptığı açı.
Görüntüdeki her kenar pikseli, bu parametre uzayında bir eğri oluşturur. 
Birçok eğrinin kesiştiği "zirve" noktaları, resimdeki gerçek doğruları verir.

Kısaca görüntü uzayı (image space) yerine parametre uzayında (parameter space) 
arama yapar.(parametre uzayında en çok kesişen noktaları doğru olarak alır.)

Standart Hough Line (cv2.HoughLines)
Bize (r,theta) değerlerini döndürür. Doğrunun uç noktalarını vermez, 
boydan boya bir çizgi denklemi sunar. Çizmek için matematiksel hesaplama yapman gerekir.

Standart houg line nasıl uygulanır:
önce doğrularımızı buluruz. daha sonra bu doğrunun ve rhonun kesiştiği noktayı buluruz.
(doğrunun merkez referans noktası)
(doğru ile merkezden çizilen doğrunun(rho) kesişim noktası: (r*cos(theta),r*sin(theta)))
daha sonra bu kesişim noktasından doğrumuzun azaldığı ve arttığı noktalarda ki noktalarını buluruz
(referans noktamızı azaltarak ve artırarak çizgimizin başlangıç ve bitiş noktasını belirleriz)
bu noktaları kullanarak cv2.line() ile çizgimizi görüntü üzerine çizdiririz

Hough transform kullanabilmemiz için kenarları tespit edilmiş bir görüntü lazım.

"""

import cv2
import numpy as np

img=cv2.imread("Desktop/goruntuIslemeKurs/1.jpeg")

img_copy=img.copy() #kopya üzerinde işlem yapabilmek için görüntünün kopyasını aldık

gray=cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY) # kopyamızı gri skalaya çevirdik

edges=cv2.Canny(gray,30,50) #canny ile görüntümüzde ki kenarları tespit ettik
                            #neredeyse her şeyi kenar olarak aldık

lines=cv2.HoughLines(edges, 1, np.pi/180 , 165) #kenarları bulduğumuz görüntüde çizgileri elde ettik
#(image, rho, theta, threshold)
#rho yu görüntüde kaç piksel ile çalışacaksak onu alırız. Biz 1 piksel ile çalışacağız                            
#merkezden uzaklığı 1 er piksellik adımlar ile tara demiş olduk.(genelde 1 alınır)
#theta:Oylama tablosundaki açı (theta) ekseninin radyan cinsinden çözünürlüğüdür.
#bilgisayar radyan cinsinden bekler. Bu yüzden genellikle np.pi / 180 (yani tam olarak 1 derece) yazılır.
# Bu, "açıları 1'er derece döndürerek kontrol et" demektir.
#threshold :Bir çizginin gerçekten bir "doğru" olarak kabul edilmesi için kaç tane çizgi ile
# kesişmesi gerektiğini belirttiğimiz parametre.
#Genellikle 50 ile 200 arasında bir değerle deneme-yanılma yapılır.
#ne kadar düşürürsek aldığımız çizgi sayısı o kadar artar. Alabileceği minumum değer 1 dir!!!

#cv2.HoughLines() bir dizi return eder.(çizgi değerlerini(rho ve thetayı )return eder
#biz bu çizgileri genellikle bir for döngüsü ile resmin üzerine çizeriz

#if type(lines) !=type(None):

if not isinstance(lines, type(None)): #lines boş değer gönderirse hata vermesin diye kullandık.
    for line in lines:
        for rho , theta in line:
            a=np.cos(theta)  #line ile rhonun (x,y) de ki kesişim noktalarını hesaplamak için gerekli
                                #cv2.line() için gerekli.
            b=np.sin(theta) #(doğrunun yönünü belirten birim vektörleri)
            
            x0=a*rho   #line ile rho nun xteki kesişim noktası
            y0=b*rho  #cv2.line için gerekli çizgi üzerindeki merkez noktamızı oluşturduk 
            #(çizginin merkez referans noktaları)
            
            x1 = int(x0 + 1000*(-b)) #sin(theta) kadar xten çıkardık (noktamızdan sola gittik)
            #1000 yazarak noktadan çok uzaklaştık(1000*1000 lik bir resim için ideal)
            #resmin size ı ile doğru oranda artırarak kullanırız
            y1 = int(y0 + 1000*(a)) #cos(theta) nın 1000 katı kadar y yi artırdık.(noktamızdan yukarı gittik)
            #ve ilk noktamızı bulduk(x1,y1) 
            x2 = int(x0 - 1000*(-b)) #sin(theta) nın 1000 katı kadar x i artırdık.(noktamızdan sağa gittik)
            y2 = int(y0 - 1000*(a))  #cos(theta) nın 1000 katı kadar y i azalttık.(noktamızdan aşağı gittik)
            #çizgimizi oluşturacağımız başlangıç ve bitiş noktasını belirledik
            
            cv2.line(img_copy, (x1,y1), (x2,y2), (0,0,255), 2) #bulduğumuz noktalardan çizgi oluşturup kopya resmimize çizdik.

cv2.imshow("org",img)
cv2.imshow("kenar",edges)
cv2.imshow("lines",img_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()




# %% cv2.HoughLines() da ki threshold değerini trackbar ile belirlediğimiz uygulama


import cv2
import numpy as np

img=cv2.imread("Desktop/goruntuIslemeKurs/yol.jpg")

img_copy=img.copy() #kopya üzerinde işlem yapabilmek için görüntünün kopyasını aldık

gray=cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY) # kopyamızı gri skalaya çevirdik

edges=cv2.Canny(gray,50,100) #canny ile görüntümüzde ki kenarları tespit ettik
                           
#HoughLinesda ki threshholdu trackbar ile belirlemek için:

def nothing(x): #trackbar için fonksiyon oluşturduk
    pass    

cv2.namedWindow("trackbar",cv2.WINDOW_AUTOSIZE) #pencere oluşturduk
cv2.createTrackbar("threshold","trackbar",0,300, nothing) #penceremize trackbar ekledik

while(1): #thresholdu trackbar ile alacağımız için while döngüsüne aldık
    
    img_copy=img.copy() #trackbar 1 değeri ile başladığı için tüm ekranı boyar. img_copy i buraya koymaz isek
    #boyadığı görüntüde işlemi yapmaya devam eder ve trackbarımızı değiştirsek bile boyalı ekranda işlem yapacağı 
    #için(threshold değerini boyalı ekranda değiştireceği için) istediğimiz sonucu alamayız
    #dolayısıyla görüntünün kopyasını buraya da ekledik
    
    threshold=cv2.getTrackbarPos("threshold", "trackbar")+1 #trackbardan değerimizi aldık. 
    #0 değerinde program hata vermesin diye 1 ekledik
    
    lines=cv2.HoughLines(edges, 1, np.pi/180 , threshold) #kenarları bulduğumuz görüntüde çizgileri elde ettik
    
    if not isinstance(lines, type(None)): #lines boş değer gönderirse hata vermesin diye kullandık.
        for line in lines:
            for rho , theta in line:
                a=np.cos(theta)
                b=np.sin(theta) #(doğrunun yönünü belirten birim vektörleri bulduk)
                
                x0=a*rho 
                y0=b*rho   #(çizginin merkez referans noktalarını bulduk)
                               
                x1 = int(x0 + 1000*(-b)) 
                y1 = int(y0 + 1000*(a)) #(x1,y1) noktamızı bulduk
                
                x2 = int(x0 - 1000*(-b)) 
                y2 = int(y0 - 1000*(a))  #(x2,y2) noktamızı bulduk
                
                cv2.line(img_copy, (x1,y1), (x2,y2), (0,0,255), 2)  #bulduğumuz noktalardan çizgi oluşturup kopya resmimize çizdik.
    
    cv2.imshow("trackbar",img_copy) #çizgileri çizdiğimiz görüntüyü trackbar penceresine yerleştirdik
    
    if cv2.waitKey(33) & 0xFF ==ord("q"): #1000ms/33ms=30.3 FPS
        break

cv2.destroyAllWindows()




# %%

"""

Probabilistic (Olasılıksal) Hough Line (cv2.HoughLinesP) 
Daha verimlidir. Tüm pikseller yerine rastgele bir piksel alt kümesi kullanır. 
En büyük avantajı, bize direkt doğrunun başlangıç ve bitiş noktalarını vermesidir.
****çok önemli bir avantaj direkt (x1,y1),(x2,y2) noktalarını dönderir

"""

import cv2
import numpy as np

img=cv2.imread("Desktop/goruntuIslemeKurs/yol.jpg")

img_copy=img.copy() #kopya üzerinde işlem yapabilmek için görüntünün kopyasını aldık

gray=cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY) # kopyamızı gri skalaya çevirdik

edges=cv2.Canny(gray,30,50) #canny ile görüntümüzde ki kenarları tespit ettik
                           
#HoughLinesda ki threshholdu trackbar ile belirlemek için:

def nothing(x): #trackbar için fonksiyon oluşturduk
    pass    

cv2.namedWindow("trackbar",cv2.WINDOW_AUTOSIZE) #pencere oluşturduk
cv2.createTrackbar("threshold","trackbar",0,300, nothing) #penceremize trackbar ekledik

while(1): #thresholdu trackbar ile alacağımız için while döngüsüne aldık
    
    img_copy=img.copy() 
    
    threshold=cv2.getTrackbarPos("threshold", "trackbar")+1 #trackbardan değerimizi aldık. 
    
    lines=cv2.HoughLinesP(edges, 1, np.pi/180, threshold,10,2)
    #(image, rho, theta, threshold, minLineLenght,maxLineGap )
    #minLineLenght:Tespit edilen bir çizgi parçasının sahip olması gereken minimum piksel uzunluğudur.
    #Kullanım: Bu değerden daha kısa olan tüm çizgiler algoritma tarafından silinir. 
    #Küçük gürültüleri veya noktacıkları temizlemek için harika bir filtredir.
    #maxLineGap:Aynı doğrultuda olan iki çizgi parçası arasındaki maksimum boşluk mesafesidir.
    #Kullanım: Eğer iki çizgi parçası arasındaki boşluk bu değerden küçükse, 
    #algoritma bunları tek bir kesintisiz çizgi olarak birleştirir. 
    #kesik yol şeritlerini veya pikselleri tam birleşmemiş kenarları bütünleştirmek için kullanılır.
    #bu iki parametrede isteğe bağlı girilir. bu parametreleri düşürdükçe bulduğumuz çizgi sayısı artar. 
    
    if not isinstance(lines, type(None)): #lines boş değer gönderirse hata vermesin diye kullandık.
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img_copy, (x1,y1), (x2,y2), (0,255,0), 2)  #bulduğumuz noktalardan çizgi oluşturup kopya resmimize çizdik.
        
    cv2.imshow("trackbar",img_copy) #çizgileri çizdiğimiz görüntüyü trackbar penceresine yerleştirdik
    
    if cv2.waitKey(33) & 0xFF ==ord("q"): #1000ms/33ms=30.3 FPS
        break

cv2.destroyAllWindows()



# %% Hough Circle Transform

"""
Daire tespiti için kullanılır.

hough circle transform denklemi:
 r**2=(x-a)**2+(y-b)**2
x,y: Çember üzerindeki kenar noktaları.
a,b: Çemberin merkez koordinatları.
r: Çemberin yarıçapı.

cv2.HoughCircles() merkez kordinatlarını ve yarı çapı return eder
parametreleri : #(image, method, dp, minDist, param1, param2, minRadius, maxRadius)

method : genelde cv2.HOUGH_GRADIENT kullanılır. 
cv2.HOUGH_GRADIENT _ALT : çok küçük daireleri tespit etmek için kullanılır
dp: Görüntü çözünürlüğü ile oylama tablosu çözünürlüğü arasındaki ters oran. 
Genellikle 1 seçilir (aynı çözünürlük). 2 seçilirse oylama tablosu yarı boyuta iner.
(görüntü büyütülerek daha küçük daireleri daha kolay algılamak için bu değeri artırabiliriz)
minDist: Tespit edilen iki daire merkezi arasındaki minimum mesafe. 
Çok küçükse aynı daire üzerinde birçok hatalı merkez bulabilir; 
hoca minDist = image.shape[0]/8 olarak kullanıyor.
çok büyükse birbirine yakın gerçek daireleri kaçırabilir.
param1: İçsel Canny kenar dedektörüne gönderilen yüksek eşik değeri.(sadece üst eşiği parametre olarak veririz)
(alt eşiği otomatik olarak üst eşiğin yarısını alır)(bu fonksiyonun içinde dahili canny edge detection yapılır )
param2: merkez algılama eşiği. Oylama (akümülatör) eşiği. Bu değer ne kadar küçükse o kadar çok (hatalı dahil) daire bulunur.
Büyükse sadece çok net daireler seçilir.
minRadius ve maxRadius: Aranacak dairelerin minimum ve maksimum yarıçap sınırları. 0 dersek program kendisi yapar.
(hoca genelde min 5 maksimum 70 civarı alıyor) max radiusa -1 girersek sadece dairenin merkezini buluruz (daireyi çizdirmeyiz)
"""

import cv2
import numpy as np

image = cv2.imread("Desktop/goruntuIslemeKurs/ay.jpg") 

img_copy=image.copy() #birinde işlemlerimizi yapacağız, diğerinin üstüne dairelerimizi çizeceğiz

image = cv2.medianBlur(image, 5)  #5*5 lik bir matrisin medyan değerini referans alarak görüntüyü bulanıklaştırdık
#görüntüde çok keskin kenarlar olursa yanlış tespitler yapma şansımız artar bu sebeple bulanıklaştırdık.
#GaussianBlur() da kullanabilirdik . gaussian kullanırsak hoca 7*7 lik kernel öneriyor

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #bulanıklaştırdığımız görüntüyü gri skalaya çevirdik

cv2.namedWindow("aaa",cv2.WINDOW_NORMAL)

def nothing(x): #param1 ve param2 parametrelerini trackbar ile değiştirerek 
    pass        #farklı resimlerde daha doğru sonuçlara ulaşabilmek için trackbar ekleyeceğiz

cv2.createTrackbar("min_dist", "aaa", 0, 500, nothing)
cv2.createTrackbar("param1", "aaa", 0, 500, nothing)
#cv2.createTrackbar("param2", "aaa", 0, 500, nothing)

while(1):
    img=img_copy.copy()
    
    min_dist=cv2.getTrackbarPos("min_dist", "aaa") +15
    param_1=cv2.getTrackbarPos("param1", "aaa") +15
    #param_2=cv2.getTrackbarPos("param2", "aaa") +15 
    
    try: #trackbardan gelen değerleri hatalı bulursa programı patlatmasın diye kullandık
        # circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT , 1, min_dist,
        #                        param1=param_1, param2=param_2,
        #                        minRadius=0, maxRadius=0)
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT_ALT , 1.5, min_dist,
                               param1=param_1, param2=0.85,
                               minRadius=0, maxRadius=0) #daha küçük daireleri bulduğumuz fonksiyonumuz
        #bunu çözünürlüğü büyüterek kullanırız(3. parametresini) param2 side 1 in altında olmalı
    except Exception as e:
        print("hata: ",e)
        
        cv2.imshow("aaa",img)
        
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break
        continue
    
    if circles is not None:
        circles = np.uint16(np.around(circles)) #HoughCircles x , y ve r değerlerini float olarak dönderir.
        #bu yüzden yuvarlayarak integera çevirdik #256 dan büyük değerlerimiz kaybolmasın diye 16 bit yaptık.
        
        for x,y,r in circles[0, :]:
            # Dairenin dış çerçevesini çiz
            cv2.circle(img, (x, y), r, (0, 255, 0), 5)
            # Dairenin merkezini çiz
            cv2.circle(img, (x,y), 2, (0, 0, 255), 5)
            
            cv2.putText(img, "r:"+str(r),(x+10,y+10) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA, )
            
    cv2.imshow("aaa", img)
     
    if cv2.waitKey(33) & 0xFF == ord("q"):
         break
  
    
cv2.destroyAllWindows()




# %% Hough Transform ile Şerit Tespiti Uygulama

"""
cv2.waitKey() hem klavye girişini yakalar hem de programa 
milisaniye cinsinden bekleme süresi ekler. Bu bekleme, döngünün ne kadar 
hızlı döneceğini etkilediği için FPS (frame per second) ile doğrudan ilişkilidir.
cv2.waitKey(delay)
delay → milisaniye (ms)
Bu süre boyunca program bekler ve tuş dinler
FPS=1000/delay ile görüntüde kullanacağımız fps i ayarlayabiliriz.
cv2.waitKey(0): Sonsuz bekler. Bir tuşa basılana kadar ekran donar.Video akışları için uygun değildir
delay=1000/FPS

"""

import cv2
import numpy as np

cam= cv2.VideoCapture("Desktop/goruntuIslemeKurs/car2.mp4")

sapma = 100                          # sol alt ve sağ alt kenar kırpma payı
kernel = np.ones((3,3), dtype=np.uint8)  # morfolojik işlemler için kernel (filtre kernelımız)

cv2.namedWindow("img",cv2.WINDOW_NORMAL)

#sadece aracın görüş alanında işlem yapmak için aldığımız görüntüyü kırpıyoruz
#daha doğru sonuçlar almak için bunu yapıyoruz.
def crop_matris(img):
    """"Şerit tespiti yapılacak yamuk(trapez) bölgesinin köşe noktalarını hesaplar.
    kısaca kameranın gördüğü tüm açıyı değil yolun bulunduğu yamuk şeklindeki alanı hesaplar"""
    x,y=img.shape[:2]
    value= np.array([
        [(sapma,x-sapma),  # sol alt köşe
         (int((y*3.2)/8),int(x*0.6)), # sol üst köşe
         (int((y*5)/8),int(x*0.6)), # sağ üst köşe
         (y,x-sapma)]    # sağ alt köşe
        ],np.int32)
    return value
#aldığımız görüntünün üst tarafını( yaklaşık yüzde 60ını) ,
#görüntünün alt tarafını (yaklaşık yüzde beşini , araçların kaputları kameradan gözüktüğü için)
#görüntümüzden kırpmak için matrisimizi oluşturduk .üçgene benzer yamuk dörtgen gibi 
#ortadan aşağıya genişleyecek şekilde görüntümüzü kullanacak şekilde fonksiyonumuzu tanımladık. 
#(aracın kaputundaki kısmı sapma kullanarak attık, farklı araçlarda sapmayı hızlıca değiştirebilmek için)
#sonuç olarak valueyi (araç kaputunun gözükmeyeceği sol alt nokta, sol üst hedef nokta(şerit takibinin biteceği nokta),
#sağ üst hedef nokta ,araç kaputunun gözükmeyeceği sol alt nokta) olarak yazdık.

#yukarıda kırptığımız matrisi işleme sokarak görüntüyü kırpacağımız fonksiyonu tanımladık:
def crop_image(img, matris):
    """Görüntüyü yamuk(trapez) maskesiyle kırpar; ilgilenilen bölgeyi (ROI) döndürür.
    Belirlenen yamuk alanın dışındaki her yeri tamamen siyaha boyar.Böylece kendi şeridimize odaklanabiliriz"""
    x, y = img.shape[:2]
    mask = np.zeros(shape = (x,y), dtype=np.uint8)   # siyah maske oluştur
    mask = cv2.fillPoly(mask, matris, 255)     # yamuğu beyaz doldur
    mask = cv2.bitwise_and(img, img, mask=mask)     # maskeyi görüntüye uyguladık(çarparak)
    return mask
#siyah maskenin üzerine beyaz olarak yamuğu koyduk ve bunu görüntümüze uyguladık
#istediğimiz alanı kırpma işlemini yaptık (şerit takibi yapacağımız alanı)

#Probabilistic Hough Line kullanabilmek için siyah beyaz kenarları algılanmış görüntü fonksiyonumuz:
def filt(img):
    """Görüntüye şerit tespitine hazırlık filtrelerini uygular."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # griye çevir
    img = cv2.inRange(img, 150, 255)             # beyaza yakın(parlak) pikselleri seç (şeritler)
    #cv2.imshow("aaa",img) #inrange e aldığımız img yi görüntülemek için
    img = cv2.erode(img, kernel)                 # gürültüyü azalt(küçük noktaları sil)
    img = cv2.dilate(img, kernel)                # aşınan kenarları geri genişlet
    img = cv2.medianBlur(img, 9)                 # görüntüyü bulanıklaştırdık(tuz-biber gürültüsünü temizledik)
    #cv2.imshow("aaa",img) bu şekilde görüntüleyerek şeritleri küçük çizmemizin sebebinin kernelın boyutu olduğunu farkettik.
    img = cv2.Canny(img, 40, 200)                # kenar tespiti yap
    return img

#cv2.HoughLinesP ile çok fazla çizgi belirleyebilir.(kesikli şerit gibi) . Ama şerit bir çizgidir.
#bu sebeple belirlediği tüm çizgilerin ortalamasını alan bir fonksiyon oluşturduk: 
def line_mean(lines):
    """
    Tespit edilen çizgileri sağ/sol olarak ayırır ve her birinin ortalamasını döndürür.
    Eğime göre: negatif eğim :sağ şerit, pozitif eğim : sol şerit.
    """
    left = []
    right = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            #çizginin eğime göre sağ şerit mi sol şerit mi olduğunu anlamak için:
            m = (y2 - y1) / (x2 - x1)  # çizginin eğimi
    
            if m < -0.2:                 # sağ şerit (negatif eğim)
                right.append((x1, y1, x2, y2))
            elif m > 0.2:                # sol şerit (pozitif eğim)
                left.append((x1, y1, x2, y2))
    
            right_mean = np.mean(right, axis=0)  # sağ şerit çizgilerinin ortalaması ,axisi değişmesin diye 0 yazdık
            left_mean  = np.mean(left,  axis=0)  # sol şerit çizgilerinin ortalaması
            # NOT: liste boşsa np.mean nan döndürür -> 0'a bölme uyarısı gelebilir

    # nan kontrolü: geçerli ortalama varsa döndür, yoksa None ver
    #çizgi ortalamaları boş değer dönderirse programda hatayı önlemek için yaptık.
    if not isinstance(right_mean, type(np.nan)): #sağ çizgi ortalaması varsa buraya girer
        if not isinstance(left_mean, type(np.nan)):
            return right_mean, left_mean #sağ ve sol çizgi ortalaması varsa dönder
        else: 
            return right_mean, None #sağ çizgi ortalaması varsa sol yoksa sadece sağı dönder solu none dönder
    else:#sağ çizgi ortalaması yoksa buraya girer
        if not isinstance(left_mean, type(np.nan)):
            return None, left_mean #sol çizgi ortalaması varsa sağ yoksa sadece solu dönder sağı none dönder
        else:
            return None, None #ikiside yoksa ikisini de none olarak dönder


def draw_line(img, line):
    """Verilen çizgiyi görüntü üzerine kırmızı renkte çizer."""
    line = np.int32(np.around(line))        # float koordinatları int'e çevir
    x1, y1, x2, y2 = line
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 10)  # kırmızı, kalınlık 10
    return img


def draw_polylines(img, matris):
    """crop_matris'te belirlediğimiz yamuk ROI muzu (İlgilenilen Alan)
    ekranda görebilmemiz için sarı renkli ince bir çerçeve olarak çizdiğimiz fonksiyon."""
    dst = np.array([
        [matris[0][1, 0], matris[0][1, 1]],  # sol üst
        [matris[0][0, 0], matris[0][0, 1]],  # sol alt
        [matris[0][3, 0], matris[0][3, 1]],  # sağ alt
        [matris[0][2, 0], matris[0][2, 1]]   # sağ üst
    ], np.int32)
    cv2.polylines(img, [dst], True, (0, 255, 255), 2)  # sarı, kapalı çokgen
    return img


def pers(img, matris, resize_x=300, resize_y=200):
    """
    Perspektif dönüşümü (Kuş Gözü Görünümü ) uygular.
    Yamuk bölgesini düzleştirilmiş yukarıdan bakış görüntüsüne çevirir.
    """
    x, y = img.shape[:2]

    # kaynak noktalar: orijinal görüntüdeki yamuk köşeleri
    src = np.float32([ #perspective fonksiyonlarında noktaların sıralaması bu şekildeydi.
        [matris[0][1, 0], matris[0][1, 1]],  # sol üst
        [matris[0][2, 0], matris[0][2, 1]],  # sağ üst
        [matris[0][0, 0], matris[0][0, 1]],  # sol alt
        [matris[0][3, 0], matris[0][3, 1]]   # sağ alt
    ])

    # hedef noktalar: düzleştirilmiş görüntünün köşeleri
    dst = np.float32([
        [0,   0  ],
        [y-1, 0  ],
        [0,   x-1],
        [y-1, x-1]
    ])

    M = cv2.getPerspectiveTransform(src, dst)          # dönüşüm matrisi hesapla
    img_output = cv2.warpPerspective(img, M, (y, x))   # perspektif uygula
    img_output = cv2.resize(img_output, (resize_x, resize_y))  # boyutlandır
    return img_output



while cam.isOpened(): #kamera açıksa döngüye girdik 
    ret , image =cam.read()
    if not ret : #kameradan görüntü okunmuyorsa döngüyü kırdık
        print("bitti")    
        break
    
    img_org=image.copy() 
    
    matris = crop_matris(image)          # yamuk köşe noktalarını hesapla
    img    = crop_image(image, matris)   # ROI bölgesini maskele

    img_org[:200, 300:600] = cv2.resize(img, (300, 200))  # kırpılmış görüntüyü sağ üste yerleştir

    img = filt(img)   # kırpılmış görüntüyü filtreye sokup kenar tespiti yaptık

    # Olası şerit çizgilerini tespit et(kenarlarları algıladığımız görüntüde çizgi tespiti yaptık):
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 20,   
                            minLineLength=5, maxLineGap=200)

    img_org[:200, :300] = pers(img_org, matris)  # kuş gözü görünümünü sol üste yerleştir
    #img_org[0:200,300:600] = cv2.cvtColor(cv2.resize(img,(300,200)),cv2.COLOR_GRAY2BGR)
    image = draw_polylines(img_org, matris)       # ROI yamuğunu çiz

    if lines is not None: #çizgi tespiti yapabildiysek çizgi ortalamasını alan fonksiyona gönderdik
        right_line, left_line = line_mean(lines)  # sağ/sol şeritleri ayır ve ortala
        if right_line is not None: #boş dönmemişse çizme draw_line fonksiyonuna git
            image = draw_line(image, right_line)  # sağ şeridi ana görüntüye çiz
        if left_line is not None:
            image = draw_line(image, left_line)   # sol şeridi ana görüntüye çiz
    
    cv2.imshow("image",image)
    #cv2.imshow("img",img)
    
    key=cv2.waitKey(33) % 0xFF #1000/30= 33
    if key ==ord("q"): 
        print("kapatıldı")
        break

cam.release()
cv2.destroyAllWindows()








