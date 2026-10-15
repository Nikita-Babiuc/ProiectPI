# ProiectPI
Proiect PI 
Masurarea pulsului fara contact prin videocamera 
(Remote PPG/rPPG)


| Nr. | Autor(i) / An | Titlul articolului / proiectului | Aplicație / Domeniu¹ | Tehnologii utilizate² | Metodologie / Abordare³ | Rezultate⁴ | Limitări⁵ | Comentarii suplimentare⁶ |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 |W. Chen, D. McDuff(2018) |DeepPhys |Telemedicină|TensorFlow/PyTorch, OpenCV| | |Sensibil la schimbări neuniforme de lumină. | |
| 2 |X. Liu(2023) |rPPG-Toolbox |Benchmarking / Evaluare rPPG|Python, PyTorch | |Evaluare standardizată pe 7 seturi de date |Nu include interfață grafică pentru transmisie live.| |
| 3 |X. Liu, Brian L. Hill, Z. Jiang, S. Patel, D. McDuff(2021) |EfficientPhys |Dispozitive mobile / Edge |PyTorch, Tensor-Shift Modules, OpenCV |Rețea neuronală rapidă bazată pe diferențe de culoare cadru-cu-cadru. | |Performanță scăzută în lumină naturală schimbătoare.| |
| 4 |Z Yu, Y Shen, J Shi, H. Zhao, P. Torr, G. Zhao(2021) |PhysFormer |Telemedicină |PyTorch, Vision Transformers |Transformer spațio-temporal ce extrage variațiile periodice de culoare ale pielii. |MAE = 2.84 bpm |Reține resurse mari de calcul; sensibil la mișcări bruște. | |
| 5 |Wang, W.,Brinker, A. C., Stuijk,Haan(2017)|Plane Orthogonal-to-Skin (POS) algorithm |Telemedicină, monitorizare vitală non-contact|Python,NumPy|Combină matematic canalele RGB pe un plan ortogonal pe culoarea pielii pentru a elimina mișcarea, fără AI. | | Sensibil la schimbări rapide de lumină și umbre| Implementare mai ușoară |


![alt text](https://github.com/Nikita-Babiuc/ProiectPI/blob/main/images/SchemaBloc.png "Schema Bloc implementare POS")
