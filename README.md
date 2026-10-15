# ProiectPI
Proiect PI 
Masurarea pulsului fara contact prin videocamera 
(Remote PPG/rPPG)
<br><br />

![alt text](https://github.com/Nikita-Babiuc/ProiectPI/blob/main/images/SchemaBloc.png "Schema Bloc implementare POS")
<br><br />
<br><br />
<br><br />

rPPG (remote photoplethysmography) extrage pulsul dintr-un video al feței, pe baza unui efect optic.
<br><br />
Principiul de funcționare: 
<br><br />
Pielea feței conține o rețea densă de capilare. La fiecare sistolă, volumul de sânge din aceste vase crește ușor, iar între bătăi scade. Hemoglobina absoarbe puternic lumina din zona verde a spectrului (în jur de 540-580 nm), deci cantitatea de lumină reflectată de piele variază periodic, sincron cu ciclul cardiac. Amplitudinea este de ordinul a 0,1-1% din intensitatea pixelilor, sub pragul vizibil cu ochiul liber, dar măsurabilă cu un senzor de cameră.
<br><br />
<br><br />

Aplicații/proiecte relevante:

<br><br />
| Nr. | Autor(i) / An | Titlul articolului / proiectului | Aplicație / Domeniu¹ | Tehnologii utilizate² | Metodologie / Abordare³ | Rezultate⁴ | Limitări⁵ | Comentarii suplimentare⁶ |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 |W. Chen, D. McDuff(2018) |DeepPhys |Telemedicină|TensorFlow/PyTorch, OpenCV|Rețea duală: o ramură detectează fața, iar cealaltă măsoară diferențele dintre cadre. |MAE = 1.52 bpm |Sensibil la schimbări neuniforme de lumină. | |
| 2 |X. Liu(2023) |rPPG-Toolbox |Benchmarking / Evaluare rPPG|Python, PyTorch |Cadru software ce unifică detecția feței, metodele clasice (POS) și AI. |Evaluare standardizată pe 7 seturi de date |Nu include interfață grafică pentru transmisie live.| |
| 3 |X. Liu, Brian L. Hill, Z. Jiang, S. Patel, D. McDuff(2021) |EfficientPhys |Dispozitive mobile / Edge |PyTorch, Tensor-Shift Modules, OpenCV |Rețea neuronală rapidă bazată pe diferențe de culoare cadru-cu-cadru. | MAE = 2.40 bpm |Performanță scăzută în lumină naturală schimbătoare.| |
| 4 |Z Yu, Y Shen, J Shi, H. Zhao, P. Torr, G. Zhao(2021) |PhysFormer |Telemedicină |PyTorch, Vision Transformers |Transformer spațio-temporal ce extrage variațiile periodice de culoare ale pielii. |MAE = 2.84 bpm |Reține resurse mari de calcul; sensibil la mișcări bruște. | |
| 5 |Wang, W.,Brinker, A. C., Stuijk,Haan(2017)|Plane Orthogonal-to-Skin (POS) algorithm |Telemedicină|Python,NumPy|Combină matematic canalele RGB pe un plan ortogonal pe culoarea pielii pentru a elimina mișcarea, fără AI. | MAE = 2.35 bmp | Sensibil la schimbări rapide de lumină și umbre| Implementare mai ușoară |

*MAE = mean absolute error
