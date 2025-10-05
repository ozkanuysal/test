Case Study
Bir online iş bulma platformu için ilan öneri sistemi kurulacaktır. Bu sistem için 2 ana
veriseti kullanılacaktır:
a) Kullanıcı Hareket Verileri: Veriye buradan ulaşabilirsiniz. Bu veride kullanıcıların
sitede tıkladığı ve başvurduğu ilanların logları yer alıyor. Veri önizlemesi (Fig 1) ve
kolon açıklamaları aşağıdaki gibidir.
b) İlan Verileri: Veriye buradan ulaşabilirsiniz. Bu veride ilanların id ’si (item_id),
başlık (pozisyon_adi) ve açıklama ( item_id_aciklama ) bilgilerini bulabilirsiniz.
Fig 1. Kullanıcı Veriseti Önizlemesi
event_type: Kullanıcı hareketinin tipini belirtir. click tıklama, purchase başvurma anlamına gelir.
client _id: Anonimleştirilmiş kullanıcı id’si
item_id : İlan id'si
ds_search_id : Kullanıcının site içinde yaptığı arama id’si. Bu arama sonrasında o satırdaki ilanı
görmüş anlamına gelir.
timestamp : İlana başvurulma zamanı
----------- ------------ ------------ ------------ ------------ ------------ ------------ ------------ -----------
Kurulacak öneri sistemi için beklenen adımlar aşağıdaki gibidir:
Part 1 - Graph Kurulumu ve Analizi
a) Kullanıcı verilerinden yola çıkarak bir Graph kurmanızı istiyoruz. Kurduğunuz graph’ta
kenar bağlantılarını nasıl oluşturuduğunuzu sebepleriyle birlikte açıklayınız.
b) Kurduğunuz graph ı analiz etmenizi ve yorumlamanızı istiyoruz.

Part 2 - Kullanıcı Verisi ile İlan Vektörü Üretme
a) Part 1’de kurduğunuz g raph’ı kullanarak her bir ilan için ilan vektörleri oluşturulmalı.
Bunun için için uygun bulduğunuz bir node embedding metodunu kullanabilirsiniz.
Seçtiğiniz embedding metodunu sebepleriyle birlikte açıklayınız.
b) Oluşturulan embeddinglerin kalitesini anlayabilmek i çin en az 1 metrik belirleyip
node embedding adımını optimize etmeye çalışınız. (Bunun için veriden benzer
olmasını ve olmamasını beklediğiniz ilanlar ile bir test kümesi oluşturabilirsiniz)
Part 3 - İlan Bilgileri ile İlan Vektörü Üretme
a) Verilen ilan bilgilerini (başlık, açıklama) kullanarak her ilan için vektör oluşturmanızı
bekliyoruz. Bunun için, önce gerekli temizleme işlemlerini yapmanızı daha sonra bir
pre-trained model kullanarak vektörler oluşturulmalı. Kullandığınız yöntem ve
modelleri seçme sebebinizi açıklamanızı bekliyoruz.
Part 4 - Öneri Sistemi
a) Bu adımda öneri sistemi geliştirmenizi istiyoruz. Bunun için 2 ilan vektörünü ((Part 2
ve Part 3) de kullanan bir sistem kurmanız gerekmektedir. Uyguladığınız sistemi
sebepleriyle birlikte açıklayınız.
b) Ürettiğiniz ilan vektörlerini bir vector database ’e kaydetmeniz i bekliyoruz
(istediğiniz bir vector database seçebilirsiniz).
c) Son olarak, input olarak ilan id’si alıp, en yakın N tane ilan önerisi yapan bir
fonksiyon yazmanızı bekliyoruz. Bu fonksiyon vector database’e bağlanıp önerileri
getirebilir olmalı.