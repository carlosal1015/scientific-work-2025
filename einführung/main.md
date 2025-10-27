


Das Gewicht entspricht demnach dem Normalvektor und der Bias ist das Skalarprodukt
mit dem ausgewählten Punkt. Ein Neuron trennt somit den Raum in die beiden Hälften
jenseits der Ebene und dies ist eine der grundlegenden Aufgaben, für die künstliche
neuronale Netze eingesetzt werden können. Gegeben seien x1, . . . , xN, denen jeweils
eine Eigenschaft yi, hier einfach yi ∈{0, 1} zugeordnet ist. Wir suchen die Gewichte
w und den Bias b des künstlichen Neurons, so dass f [w, b](xi) = yi gilt. D.h., wir
suchen ein Modell, welches die Klassifizierung der Daten in Bezug auf die Eigenschaften
vornimmt. Die Gewichte verstärken oder schwächen den Einfluß des Inputs und somit
den Übergang von Nicht-Aktivierung eines Neurons zur Aktivierung. Der Bias sorgt für
eine Verschiebung der Aktivierung.
Ein Modell bestehend aus einem einzelnen Neuron ist sehr eingeschränkt und es muss
davon ausgegangen werden, dass die Aufgabe nicht exakt zu lösen ist. Stattdessen ver-
suchen wir das beste Modell zu finden, d.h. wir betrachten analog zur Regression in den
Abschn.4.5 und dem Exkurs 11.4 die Minimierungsaufgabe
Künstliche neuronale Netzwerke (KNN) werden aus einzelnen künstlichen Neuronen
zusammengesetzt. Die einfachsten KNN sind sogenannte Ein-Layer-Netzwerke:

Definition11.13(Ein-SchichtNetzwerk) Essei W ∈Rn×m undb ∈Rn sowieσ : R →R
eine Aktivierungsfunktion. Dann heißt die Funktion
ein künstliches neuronales Netzwerk mit einer Schicht und n künstlichen Neuronen. Dabei ist
die Anwendung der Aktivierungsfunktion komponentenweise zu verstehen, d.h. für y ∈Rn
ist

Von dieser Definition kommen wir schnell zu allgemeineren künstlichen neuronalen Netzen,
den sogenannten tiefen neuronalen Netzen oder eben deep neural networks. Hierzu schalten
wir mehrere Ein-Schicht-Netzwerke in Reihe. Die Ausgabe von Schicht l ist Eingabe von
Schicht l + 1.

Definition 11.14 (Tiefes neuronales Netz)
Es sei L ∈N die Anzahl der Schichten.
Für l = 1, . . . , L seien Wl ∈Nnl×nl−1 sowie bl ∈Rnl, und durch σ : R →R sei eine
Aktivierungsfunktion gegeben. Dann ist durch f : Rn0 →RnL gemäß

ein deep neural network gegeben. Die erste Schicht f 1(·) heißt Eingabeschicht, die letzte
Schicht f L(·) die Ausgabeschicht und hier wird im Allgemeinen keine Aktivierungsfunk-
tion hinzugefügt. Die inneren Schichten heißen verborgene Schichten (hidden Layer). Wir
benennen die Architektur eines tiefen neuronalen Netzwerkes mit N(σ; n0, n1, . . . , nL),
bzw. kurz mit N(L, n) mit n = maxl nl. Den zugehörigen Parameterraum bezeichnen wir
mit

Für eine Parameterwahl (W, b) ∈P bezeichnen wir mit f [W, b] : Rn0 →RnL die
resultierende Funktion gemäß (11.18). Die Menge der Realisierungen, d.h., die Menge der
in der Architektur darstellbaren Funktionen benennen wir mit

Das Abzählen der freien Parameter ergibt:
Korollar 11.15 (Anzahl freier Parameter) Es sei N(σ; n0, . . . , nL) ein künstliches neu-
ronales Netz mit L Schichten und jeweils nl künstlichen Neuronen. Dann hat der Parame-
terraum PN insgesamt
freie Parameter, davon �L
l=1 nlnl−1 Gewichte und �L
l=1 nl Biaswerten.

Eine Grundoperation im maschinellen Lernen ist das Auswerten eines gegebenen Netzes
N(σ; n0, . . . , nL) bei gegebenen Parametern (W, b) ∈PN für einen Datenpunkt x ∈Rn0
oder, meistens für eine ganze Liste von (eventuell sehr vielen) Datenpunkten xi ∈Rn0 für
i = 1, . . . , Ntr.

Bemerkung 11.16 (Netzwerk-Architekturen) Die obige Definition führt nur die ein-
fachste Art von künstlichen neuronalen Netzwerken ein: alle Ausgaben von Schichtl−1 sind
Eingabe für alle künstlichen Neuronen der Schicht l. Diese Netze werden auch Multilayer
perceptron (mehrlagiges Perzeptron) genannt. Sie gehören zur Kategorie der Feedforward
neural networks, d.h., die Information geht nur in eine Richtung, von Schicht 1 zu Schicht
2, und so weiter. Neuronale Netze, in denen die Ausgabe der Schicht l auch als Eingabe
von früheren Schichten l′ < l dienen kann, heißen recurrent neural networks (rekurrente
neuronale Netze).
Prinzipiell kann in jeder Schicht eine andere Aktivierungsfunktion verwendet werden.
Wird in der letzten Schicht eine Aktivierungsfunktion angewendet, so schränkt dies den
Bildraum ein, bei der Sigmoid-Aktivierung auf (0, 1), bei der ReLU-Aktivierung auf die
nicht-negativen Zahlen.
Sogenannte skip connections lassen einzelne Schichten aus. D.h., die Ausgabe von
Schicht l kann direkt als Eingabe von z.B. Schicht l + 2 verwendet werden. Eine andere
einfache Modifikation sind residual networks, hier bildet sich eine Schicht in der Form
d.h., das Ergebnis des künstlichen Neurons wird zur Eingabe addiert.
In der Bildverarbeitung spielen die convolutional neural networks eine große Rolle. Hier
ist jede Schicht eine Faltung (engl. convolution). Das besondere an convolution networks
ist, dass die Größe der Eingabe x nicht unbedingt an die Anzahl von Gewichten gekoppelt
sein muss. Mit den Faltungs-Gewichten W ∈R f definieren wir
wobei x0 = x−1 = · · · = x1−f = 0 gesetzt wird. Die gleichen Gewichte werden immer nur
auf einen kleinen Teil der Eingaben angewendet. Auf diese Weise können mit nur wenigen
freien Parametern (den Gewichten) Funktionen in beliebig hochdimensionalen Räumen
dargestellt werden.
Moderne Architekturen, wie sie zum Beispiel in Sprachmodellen wie ChatGPT2 verwen-
det werden sind weitaus komplexer, nutzen und kombinieren aber auch die hier eingeführ-
ten Basismodelle. In der Praxis eingesetzte künstliche neuronale Netze haben eine enorme
Größe, z.B. hat ChatGPT-3 etwa 200 Milliarden, also 2 · 1011 freie Parameter.
In Abb.11.11 stellen wir einige künstliche neuronale Netze graphisch dar. Einen ausführ-
lichen Überblick gibt die Literatur [11, 17, 43, 51, 66].

## Approximation mit neuronalen Netzen

Wir betrachten ausschließlich tiefe künstliche neuronale Netze vom Typ der Definition 11.14
und fassen ein Netz mit L Schichten als Funktion f : Rn0 →RnL mit den Gewichtsmatrizen
Wl und Bias-Vektoren bl als freie Parameter. Zunächst stellen wir einen einfachen aber
wichtigen Zusammenhang fest:

Satz 11.17 Es sei N = N(σ, n0, . . . , nL) eine Netzwerkarchitektur mit L Schichten und
Parametern Wl ∈Rnl×nl−1 sowie bl ∈Rnl für l = 1, . . . , L. Die Menge der Realisierungen

ist im allgemeinen kein Vektorraum. Insbesondere ist FN nicht abgeschlossen bzgl. der
Addition oder skalaren Multiplikation

Beweis Wir konstruieren ein einfaches Gegenbeispiel:

2 Chat Generative Pre-trained Transformer, ein Sprachmodell basierend auf neuronalen Netzen, siehe
https://en.wikipedia.org/wiki/ChatGPT.

Abb.11.11 Beispiele für künstliche neuronale Netze. Oben links: einfaches mehrschichtiges Netz-
werk (multilayer perceptron) N(σ; 2, 3, 3, 2) mit Sigmoid-Aktivierung (bis auf die Ausgabeschicht).
Oben rechts: dieses Netzwerk stellt Funktionen f : R2 →R2 dar. Tiefes neuronales Netzwerk 2
Eingaben und Ausgaben und L −1 verborgenen Schichten mit jeweils n Neuronen. Unten links:
rekurrentes Neuronales Netzwerk, bei dem Ausgaben auch Eingaben für frühere Schichten sein kön-
nen. Unten rechts: Künstliches neuronales residual network. Die Eingabe wird zur Ausgabe addiert.
Dieses Netz stellt Funktionen f : R →R dar mit dem Aufbau f (x) = x + f [W, b](x)

und wählen die zwei Realisierungen f1, f2 ∈FN

Es gilt

da die Betragsfunktion nicht in der Form ReLU(wx + b) zu schreiben ist.
Der Zusammenhang mag klar sein, hat aber wesentliche Konsequenzen: Es gibt keine Basis
von FN und wir können auch keinerlei vertraute Struktureigenschaften wie Skalarprodukte
oder die Linearität nutzen. Diese Hilfsmittel waren entscheidend für die effiziente Umset-
zung von Bestapproximationen oder Lagrange-Interpolationen. Zur Suche der optimalen
Parameter und Koeffizienten müssen wir also neue Wege einschlagen.

Bereits in den 1980er Jahren wurde nachgewiesen, dass neuronale Netze sehr gute Appro-
ximationseigenschaften haben, genauer gesagt, dass die Menge FN (L,n) für L →∞oder
n →∞dicht in der Menge der stetigen Funktionen liegt.

Theorem 11.1 (Universelles Approximationstheorem) Es sei I = [a, b] ⊂R, und σ sei
eine stetige Aktivierungsfunktion mit der Eigenschaft

Dann ist die Menge der neuronalen Netzen mit einer verborgenen Schicht FN (σ;N,1), also

dicht in C(I).

Beweis Beweise sind z.B. bei Cybenko [20] oder Hornik [52] gegeben. Es werden zum
Teil tiefliegende Resultate der Funktionalanalysis benötigt.

DerSatzbedeutet,dasseszujederstetigenFunktion f ∈C(I)undjedempositivenϵ > 0 ein
neuronales Netzwerk mit Architektur N(σ; 1, N, 1) gibt und eine Realisierung f [W, b] ∈
FN , so dass maxx∈I ∥f (x) −f [W, b](x)∥≤ϵ gilt. Das Resultat gibt jedoch noch keine
quantitativen Abschätzungen, d.h., es sagt nichts darüber aus, wie schnell N wachsen muss,
wenn ϵ →0 kleiner wird.
Das ursprüngliche Resultat wurde für Aktivierungsfunktionen vom Sigmoid-Typ gezeigt,
kann jedoch auf fast beliebige Funktionen erweitert werden und die Eigenschaft bleibt
erhalten, solange die Aktivierungsfunktion kein Polynom ist. Die Dichtheit überträgt sich
auch auf Funktionen C(X) für X ⊂Rn und ebenso auf vektorwertige Funktionen.
Einfacher zu verstehen sind ReLU-Netze. Es gilt:

Satz 11.18 (ReLU-Netze) Es sei N eine neuronale Netzwerkarchitektur mit ReLU-Akti-
vierung. Dann ist jedes f [W, b] ∈FN eine stückweise lineare Funktion.

Beweis Das Netz ist aufgebaut als Verkettung von einschichtigen Netzen
Jedes fl(x) = ReLU(Wlxl +bl) ist als Komposition der affin linearen Abbildung Wlxl +bl
und der stückweise linearen Abbildung ReLU(·) stückweise linear. Also ist auch ganz f
stückweise linear.

Im einfachen skalaren Fall können wir zu jedem Polygonzug ein entsprechendes neuronales
Netz konstruieren:

Satz 11.19 Es sei I = [a, b]. Jede stückweise lineare Funktion mit N ∈N paarweise
verschiedenen Stützstellen lässt sich als künstliches neuronales ReLU-Netzwerk mit einer
Schicht und N künstlichen Neuronen darstellen.

Beweis Es seien (xi, yi) ∈[a, b] × R für i = 1, . . . , N Stützstellenpaare der stückweise
linearen Funktion ph, mit ph(xi) = yi. Ohne Einschränkung nehmen wir an, dass xi−1 < xi
gilt. Dann konstruieren wir das Netzwerk als

Auf jedem Teilintervall (xk−1, xk) ist f (x) eine lineare Funktion, denn es gilt

d.h., ReLU(x −xi−1) ist stetig und stückweise linear. Damit ist auch die Summe stückweise
linear.

Es sei x ∈[xk−1, xk]. Dann gilt ReLU(x −xi−1) = x −xi−1 für x ≥xi−1, also i ≤k
und ReLU(x −xi−1) = 0 für x ≤xi−1 also für k < i. Hiermit folgt

Dies entspricht gerade der stückweise linearen Funktion auf [xk−1, xk]. Der Ausdruck
(11.19) lässt sich schreiben als

mit den Gewichten N −1 Gewichten
und den N Biaswerten

Hiermit können wir ein einfaches quantitatives Konvergenzresultat für neuronale Netze
herleiten:

Korollar 11.20 Die einschichtigen ReLU Netze sind gerade die stückweise linearen Funk-
tionen. Zu einer Funktion f ∈C2[a, b] existiert eine Folge von neuronalen Netzwerkfunk-
tionen fN ∈FN (ReLU,N) mit

mit einer Konstanten C > 0.

Beweis Satz 11.19 zeigt, dass fN eine stückweise lineare Funktion mit N Intervallen ist.
Bei gleichmäßiger Zerlegung von [a, b] ist h = (b −a)/N die Intervallgröße. Mit der
Fehlerabschätzung für stückweise lineare Funktionen (9.7) folgt dann sofort die Fehlerab-
schätzung.

Bemerkung 11.21 (Approximation hochdimensionaler Funktionen) Künstliche neuro-
nale Netze spielen ihre Stärke gerade bei der Approximation hochdimensionaler Funktionen
f : Rn →R mit n ≫1 aus. Wollen wir so eine Funktion mit klassischen Mitteln, z.B.
mit stückweise linearen Funktionen approximieren, so ist der erste Schritt stets eine Zerle-
gung des Definitionsbereiches X ⊂Rn in ein Gitter. Als einfaches Beispiel wählen wir den
n-dimensionalen Einheitswürfel

Ein regelmäßiges Punktgitter von Wn mit Gitterweite h = 1/M besteht aus den Punkten
Insgesamt hat das Gitter somit mit (M + 1)n Punkten eine exponentiell steigende Komple-
xität. Dies bedeutet, dass der Aufwand zum Erreichen einer gewissen Genauigkeit expo-
nentiell mit der Dimension steigt. Dieser Zusammenhang ist üblich bei der Approximation
aller hochdimensionalen Probleme und wird der Fluch der Dimension genannt.
In bestimmten Situationen können künstliche neuronale Netze solche Funktionen viel
effizienter approximieren. Das ist zumeist dann der Fall, wenn die zu approximierende
Funktion eine inhärente niederdimensionale Struktur hat. Hierzu diskutieren wir ein einfa-
ches Beispiel. Die zehndimensionale Funktion

lässt sich über eine Koordinatentransformation
als eindimensionale Funktion darstellen. Hier haben wir die symmetrische Struktur der
Funktion ausgenutzt und sie einfacher dargestellt. Der Aufwand würde von O(M10) auf
O(M) sinken. Solche niederdimensionalen Strukturen finden sich in vielen Aufgaben, meist
sind diese aber nicht so offensichtlich.
Künstliche neuronale Netze sind unter Umständen in der Lage, diese Zusammenhänge
automatisch zu erlernen und können dann sehr effiziente Approximationen liefern. Wir
verweisen auf die Literatur [12].

## Trainieren von künstlichen neuronalen Netzen

In diesem Abschnitt beschreiben wir die Suche nach Parametern (W, b) ∈PN , die in einer
gegebenen Netzwerkarchitektur N(σ; n0, . . . , nL) die optimale Realisierung f [W, b] ∈
FN darstellen, um eine gegebene Funktion f : Rn0 →RnL auf einer Menge X ⊂Rn0 zu
approximieren, d.h., f [W, b] ≈f . Hierzu wählen wir eine Menge von Punkten xi ∈X
sowie Werten yi = f (xi) für i = 1, . . . , Ntr. Diese Menge nennen wir die Trainingsdaten

Die Interpolationsaufgabe

werden wir nicht lösen können. Zunächst ist Ntr in der Regel sehr groß. Der große Unter-
schied zur Polynominterpolation ist, dass die unbekannten Koeffizienten, die Gewichte und
der Bias in nichtlinearer Form vorkommen und das es keine Basis der Menge der zur Inter-
polation zur Verfügung stehenden Funktionen gibt. Im Sinne der Bestapproximation (siehe
Abschn.4.5) werden wir daher ein Funktional (also eine Abbildung mit Bildraum R) defi-
nieren, welches das Erfüllen der Interpolationsaufgabe (11.20) misst

und versuchen, das Minimum von diesem Funktional zu bestimmen, also

Das Funktional J(·) wird oft die loss-Funktion also Verlust-Funktion genannt.

Dieses Optimierungsproblem können wir im Allgemeinen nicht exakt lösen. Stattdessen
müssen wir uns einem Mininum mit einem approximativen Verfahren, wie dem Gradienten-
verfahren aus Abschn.11.1 annähern. Wir geben hier Algorithmus 11.1 im Kontext der
künstlichen neuronalen Netze noch einmal an

Die Schrittweite sk wird im Bereich der KNN üblicherweise mit learning rate bezeichnet.
Die Anwendung des Verfahrens auf künstliche neuronale Netze bringt einige Probleme mit
sich. Je nach gewählter Aktivierungsfunktion ist die Zielfunktion J(·) gar nicht überall dif-
ferenzierbar, dies ist z.B. bei der ReLU Funktion der Fall. Künstliche neuronale Netze sind
oft stark überparametrisiert, d.h., die Anzahl der freien Parameter kann sehr groß sein. Die
Zielfunktion ist darüber hinaus nicht konvex und hat üblicherweise sehr viele lokale Minima.
Wir müssen also davon ausgehen, dass es kaum möglich sein wird, ein globales Minimum
zu finden. Für eine leistungsfähige Optimierung müssen zahlreiche Heuristiken angewendet
werden, um robust ein gutes Minimum zu finden und nicht im erstbesten lokalen Mini-
mum zu verharren. Schließlich kommt hinzu, dass die Auswertung von Zielfunktion und
deren Gradient sehr aufwändig ist, wenn die Anzahl der Datenpunkte groß ist. Daher wird
das sogenannte stochastische Gradientenverfahren angewandt, wo aus nicht der komplette
Gradientenvektor ausgewertet wird, sondern nur einzelne – zufällig gewählte – Komponen-
ten. Dies reduziert den Rechenaufwand erheblich und führt trotzdem oft zu zufriedenstellen-
den Ergebnissen. Die Arbeit [37] gibt einen guten Überblick über die Forschung auf diesem
Gebiet.

### Backpropagation - Die Ableitung eines künstlichen neuronalen Netzes

Wie wir bereits wissen, ist der wichtigste Bestandteil im Gradientenverfahren die Bestim-
mung der Abstiegsrichtung, also des Gradienten der Verlustfunktion J(p) = J(W, b).
Hierzu müssen alle Ableitungen von J(·) in Bezug auf alle Parameter des künstlichen
neuronalen Netzes bestimmt werden, d.h. in Richtung der Gewichtsmatrizen Wl und der
Bias-Vektoren bl, jeweils für l = 1, . . . , L. Korollar 11.15 hat gezeigt, dass die Parame
terzahl sehr schnell ansteigen kann. Hinzu kommt, dass die Verlustfunktion J(·) natürlich
vom Ergebnis des neuronalen Netzes f [W, b](·) abhängt und dieses nur in algorithmischer
Schreibweise gegeben ist, vergleiche Definition 11.14. Die Auswertung der Ableitung der
Verlustfunktion ist der aufwändigste Teil beim Trainieren des neuronalen Netzes.
Im Folgenden benennen wir mit „X“ eine beliebige Ableitungsrichtung, also z.B. den
Eintrag einer Gewichtsmatrix X = Wl
rs in Schicht l. Dann gilt mit (11.21) zunächst

mit dem euklidischen Skalarprodukt (·, ·). Diese äußere Ableitung ist einfach aufzustel-
len, allerdings ist die Berechnung der Ableitungen von der Netzwerkfunktion f [W, b](x)
involvierter. Wir werden diese Ableitung in den folgenden Schritten entwickeln.
Schematisch lässt sich ein künstliches neuronales Netzwerk als eine verschachtelte
Anwendung von Aktivierungsfunktionen und affin linearen Funktionen verstehen, d.h.,

Ableitungen werden nach allen Gewichtsmatrizen Wl und Bias-Vektoren bl und eventu-
ell nach der Eingabe x benötigt. Das Grundwerkzeug zur Berechnung ist ganz einfach die
Kettenregel, die wir aber effizient für genau diese Anwendung formulieren werden. In Defi-
nition 11.14 haben wir die Größen

eingeführt, wobei x0 = x ist und x L = zL = f [W, b](x) die gesuchte Ausgabe ist. Dann
können wir das gesamte Netzwerk kompakt in der Form

schreiben. Fassen wir weiter xl und zl zusammen, d.h. wir nutzen die Notation

so ist das Netzwerk einfach als

gegeben. Mit der wiederholten Anwendung der Kettenregel schreiben wir die Ableitung
nach einer abstrakten Variable „X“, die in Schicht l oder tiefer vorkommt, als

Bevor wir konkreter werden, vereinfachen wir die Notation weiter. Die Ableitung dxl+1/dxl
entspricht gerade dem Gradienten der Funktion xl(x). D.h., wir können (11.27) schreiben
als

wobei „·“ das Matrix-Matrix Produkt ist. Noch kürzer schreiben wir
wobei ∇xl x L der Gradient der zusammengesetzten Funktion x L ◦· · · ◦xl+1(xl) ist.
Wir beginnen nun mit der Berechnung des Gradienten einer Schicht in Bezug auf ihre
Eingabe.

Satz 11.22 (Ableitung künstlicher neuronaler Netze nach der Eingabe) Es sei N(σ;
n0, . . . , nL) ein künstliches neuronales Netz mit differenzierbarer Aktivierungsfunktion
σ(·). Für x0 := x ∈Rn0 seien xl und zl für l = 1, . . . , L die Zwischenergebnisse aus
Definition 11.14. Dann gilt für l = 1, . . . , L −1

Dabei ist für einen Vektor a ∈Rn und eine Matrix A ∈Rn×m das komponentenweise
Produkt a⋆A definiert als

Es gilt

Beweis Das Ergebnis kann komponentenweise nachgerechnet werden. Es sei l < L. Dann
ist für i = 1, . . . , nl und j = 1, . . . , nl−1 mit der Kettenregel angewendet auf xl
i = σ(zl
i)

Wir betrachten die Aktivierungsfunktion und ihre Ableitung weiter als eine skalare Funktion,
die auf jeden Eintrag des Vektors separat angewendet wird, d.h.,

Das Ergebnis kann bei Verwendung des komponentenweisen Produktes (11.30) abgelesen
werden. Auch folgt sofort, dass ∇xl ∈Rnl×nl−1. Für das letzte Element x L gilt x L = zL,
d.h., der Gradient ist die innere Ableitung, also ∇x L = W L.

Bemerkung 11.23 (Sonderbehandlung der Ausgabeschicht) Bei der Berechnung des
Gradienten haben wir die letzte Schicht x L separat behandelt. Dies liegt daran, dass wir im
Allgemeinen auf die Ausgabe nicht mehr die Aktivierungsfunktion anwenden, siehe auch
Bemerkung 11.16. Es gibt durchaus auch Anwendungen, wo es wünschenswert ist, dass das
Ergebnis z.B. im Intervall [0, 1] liegt und dann würde die Aktivierungsfunktion auch in der
Ausgabeschicht zum Einsatz kommen, d.h.

Der erste Faktor der Ableitung wäre dann entsprechend σ ′(zL)⋆W L anstelle von W L.
Mit dem mehrlagigen Perzeptron, Definition 11.14, betrachten wir hier nur einen sehr ein-
fachen Fall von künstlichen neuronalen Netzen. Bei komplexer aufgebauten Architekturen
können aber die Bausteine zusammengefügt werden, die wir hier entwickeln.
♦

Die Ableitungen der Stufen hängen von den Ableitungen der Aktivierungsfunktion ab. Es
gilt:

Satz 11.24 (Ableitung der Aktivierungsfunktionen) Für die Ableitungen der Sigmoid
Funktion sowie des hyperbolischen Tangens gilt

Die Ableitung der ReLU Aktivierung ist nur stückweise definiert:

Beweis Nachrechnen.

Mit dem bisher gezeigten können wir bereits die Ableitung des neuronalen Netzes nach der
Eingabe berechnen.

Korollar 11.25 (Ableitung des neuronalen Netzes nach seiner Eingabe) Mit den Vor-
raussetzungen von Satz 11.22 gilt

sowie für den Gradienten in Bezug auf die Ausgabe von Schicht l:
Es ist ∇xl f [W, b](x) ∈RnL×nl.

Der nächste Schritt ist die Berechnung der Ableitungen in Richtung der Gewichte Wl
und bl in Schicht l, siehe die abstrakte Form der Ableitung (11.27). Hierzu genügt es, wenn
wir nur l-te Schicht des Netzes betrachten:

Satz 11.26 (Ableitung einer Schicht nach den Gewichten) Für l = 1, . . . , L, sei
mit Wl ∈Rnl×nl−1, bl ∈Rnl und einer differenzierbaren Aktivierungsfunktion σ(·). Dann
gilt für l = 1, . . . , L −1

sowie

Beweis Wir rechnen die Ableitung komponentenweise nach. Für i,r = 1, . . . , nl und
s = 1, . . . , nl−1 gilt

Die meisten Ableitungen, genauer gesagt alle Ableitungen für i ̸= r, verschwinden. Anstelle
von n2
l nl−1 müssen nur nlnl−1 Ableitungen berechnet werden. Entsprechend gilt
Auch hier verbleiben nur nl der eigentlich n2
l Ableitungen, die nicht verschwinden.

Hiermit können wir nun die Ableitungen des künstlichen neuronalen Netzes in Richtung
aller Parameter angeben.

Korollar 11.27 (Ableitung des neuronalen Netzes nach den Parametern) Mit den Vor-
raussetzungen von Satz 11.22 gilt für l = 1, . . . , L −1, dass

sowie

Beweis Einsetzen der Beziehungen aus Satz 11.26 in (11.29) gibt

d.h., es bleibt in der Summe jeweils nur der Term für j = r übrig. Für l = L vereinfacht
sich die Rechnung entsprechend.

Die Ableitung nach dem Bias kann kompakt als Gradient geschrieben werden, d.h.
∇bl f [W, b](x) = ∇xl x L⋆σ ′(zl), wobei wir entsprechend zu (11.30) für A ∈Rn×m und
a ∈Rm die Notation

verwenden. Die Ableitungen nach den Gewichten Wl
rs bilden hingegen einen Tensor dritter
Stufe. Wir benötigen die Ableitungen jedoch ausschließlich zur Berechnung des Gradienten
der Verlustfunktion und brauchen hierfür keine weitere kompakte Schreibweise.

Satz 11.28 (Gradient der Verlustfunktion) Mit den Voraussetzungen von Satz 11.26 gilt
für l = 1, . . . , L −1

sowie

Dabei ist

der Defekt der Trainingsdaten und A ◦B ∈Rn×m das Hadamard-Produkt zweier Matrizen
A, B ∈Rn×m (bzw. Vektoren), gegeben durch

Beweis Wir kombinieren (11.23) mit den Ergebnissen aus Korollar 11.27 und fassen die
auftretenden Summen in kompakter Schreibweise zusammen

und entsprechend

Mit Satz 11.28 haben wir die Herleitung des Gradienten der Verlustfunktion nach allen
Parametern abgeschlossen. Es bleibt noch die effiziente algorithmische Umsetzung. Der
maßgebliche Teil des Aufwandes liegt in der Berechnung des Gradienten nach der l-ten
Schicht, d.h. ∇xl x L, der für jedes Element der Trainingsdaten berechnet werden muss. In
Korollar 11.25 haben wir hierfür Beziehung

hergeleitet. Die Ableitung nach Schicht l benötigt das Produkt der Gewichtsmatrizen
W L, W L−1, . . . , Wl+1 sowie die linearen Zwischenergebnisse zL−1, zL−2, . . . , zl+1. Um
diese zu berechnen, muss das Netzwerk allerdings von der Eingabe x = x0 an durchlaufen
werden. Damit diese Werte nicht für jede Ableitung neu berechnet werden müssen, erfolgt
das Aufstellen des Gradienten in zwei Schritten: zunächst wird das Netzwerk für eine Ein-
gabe x = xi ausgewertet und alle Zwischenergebnisse z1
i , . . . , zL−1
i
werden gespeichert.
Dieser Schritt wird der Vorwärts-Modus genannt. Im Anschluss werden die Gradienten,
beginnend bei Schicht L berechnet. Dieser zweite Schritt heißt entsprechend der Rückwärts-
Modus. Der gesamte Algorithmus wird Backpropagation genannt.

Algorithmus 11.8: Backpropagation: Auswertung der Ableitungen des künstlicher
neuronalen Netzes und der Verlustfunktion

Satz 11.29 (Backpropagation)Seien Ntr dieMengederDatenpunkte,n := max n0, . . . , nL
und L die Anzahl der Schichten. Der Backpropagation-Algorithmus berechnet die Gradien-
ten der Verlustfunktion in Bezug auf die Gewichte. Für die Architektur N(σ; n0, . . . , nL)
können die Gradienten mit dem Aufwand

berechnet werden.

Beweis Die Berechnung der Gradienten der Verlustfunktion folgt aus den Sätzen 11.22,
11.24, 11.26 und 11.28. Die Aufwandsabschätzung folgt durch Abzählen der Operationen,
insbesondere der Matrix-Matrix Multiplikationen.

Bemerkung 11.30 (Effiziente Implementierungen) Effiziente Implementierungen von
neuronalen Netzen wie PyTorch [69] oder Tensorflow [1] erreichen ihre Leistungsfähigkeit
durch extreme Optimierung der grundlegenden Matrixoperationen. Wir haben Algorith-
mus 11.8 als Schleife über alle Trainingsdatenpaare formuliert. Viel effizienter können die
Netzwerke und Ableitungen jedoch ausgerechnet werden, wenn jeweils alle Daten gleich-
zeitig ausgewertet werden. Dies erfordert jedoch den Umgang mit Tensoren höherer Stufe,
d.h. mit multilinearen Abbildungen Xi, j,k mit drei oder noch mehr Indizes.
Darüber hinaus nutzen Bibliotheken wie PyTorch oder Tensorflow spezielle Hardware
wie Grafikkarten (GPU’s) oder Beschleunigerkarten wie TPU’s (Tensor-Processing-Units),
die genau für diese Aufgaben optimiert sind.

Mit den Gradienten der Verlustfunktion nach den Gewichten und dem Bias kann das künst-
liche neuronale Netzwerk trainiert werden, d.h., wir können mit dem Gradientenverfahren
die optimalen Parameter (W, b) ∈P suchen, welche die Verlustfunktion minimieren.

Beispiel 11.31 (Funktionsapproximation mit künstlichen neuronalen Netzen)

Wir betrachten die Funktion

die wir bereits bei der Suche nach Nullstellen analysiert haben und versuchen, diese
durch ein einfaches künstliches neuronales Netzwerk darzustellen. Wir starten mit der
Architektur N(σ; 1, 8, 1), d.h. die Menge der Funktionen

Aus dem Intervall I = [−10, 2] wählen wir Ntr = 20 zufällige Punkte x1, . . . , xNtr und
zugehörige Werte yi = f (xi). In Abb.11.12 zeigen wir links den Verlauf Verlustfunktion
in 10000 Schritten der Optimierung. Rechts geben wir den Verlauf der Funktion f (x)

sowie das trainierte neuronale Netzwerk f [W, b](x) an. Die meisten Trainingspunkte
werden vom Netzwerk sehr gut approximiert. Jenseits der Trainingsdaten, vor allem im
Bereich [1, 2] stellt das Netzwerk jedoch keinerlei Approximation an die Funktion f (x)
dar. Dies ist auch nicht zu erwarten, da das Netzwerk beim Trainieren ja nicht die Funktion
als solche kennengelernt hat, sondern nur einige diskrete Werte.
Verteilen wir mehr Trainingspunkte im Intervall, so verbessert sich die Approximation.
In Abb.11.13 zeigen wir links die Approximation für Ntr = 100 zufällige Punkte. Das
von uns gewählte Netzwerk ist sehr klein und verfügt nur über 25 freie Parameter. In
der rechten Abb.11.13 zeigen wir die Approximation für die gleichen Ntr = 100 zufäl-
ligen Trainingspunkte, jedoch einem weitaus größeren Netzwerk mit der Architektur
N(tanh; 1, 16, 16, 16, 1). Dieses Netzwerk kann die Funktion in allen Punkten sehr gut
darstellen.

Abb. 11.13 Approximation einer Funktion mit künstlichen neuronalen Netzen. Links: klei-
nes Netzwerk N(tanh; 1, 8, 1) mit Ntr
= 100 Trainingsdaten und rechts: großes Netzwerk
N(tanh; 1, 16, 16, 16, 1) wieder mit Ntr = 100 Trainingsdaten

### Überanpassung,Testen und Validieren und Generalisierung

Künstliche Neuronale Netze werden oft mit einer sehr großen Zahl von freien Parametern
entworfen. Dies macht sie sehr leistungsfähig und flexibel, die Überparametrisierung birgt
jedoch die Gefahr einer Überanpassung, overfitting genannt: Die Datenpaare werden gut
abgebildet yi ≈f [Wopt, bopt](xi), in anderen Punkten hat das Netzwerk jedoch nicht den
Funktionsverlauf gelernt (siehe Abb.11.12 rechts). Hierzu betrachten wir die Approxima-
tion der Funktion f (x) = 0, gehen aber aus, dass zu Trainings-Stützstellen xi ∈[−10, 2]
die Funktion nur Fehlerhaft bekannt ist yi = 0 + N(0.1). Dabei ist N(0.1) eine normalver-
teilte Zufallszahl mit Standardabweichung 0.1. Abb.11.14 zeigt die Trainingsdaten und die
Approximation mit einem (sehr reichhaltig parametrisierten) Netzwerk.

Es gibt verschiedene Strategien, overfitting zu minimieren. Das übliche Vorgehen ist
es, die Daten aufzuteilen in Trainingsdaten und Validierungsdaten: Die Trainingsdaten
(vielleicht 70% der Gesamtdaten) dienen dazu, dass künstliche neuronale Netzwerk zu
trainieren. D.h., sie sind die Basis zum Auswerten des Gradienten bei der Optimierung.
Während der Optimierung wird die Verlustfunktion nicht nur auf Basis der Trainingsda-
ten ausgerechnet sondern auch auf Basis der Validierungsdaten. Diese werden jedoch nie
zur Berechnung des Gradientens beim Trainieren verwendet, so dass das neuronale Netz die
Daten nicht kennt. Wir erwarten, dass das neuronale Netzwerk auch auf diesen Validierungs-
daten gute Vorhersagen liefert und dass die Verlustfunktion der Validierungsdaten im Laufe
der Optimierung abnimmt. Kommt es zu einer Diskrepanz, d.h. erreichen wir einen Punkt,
an dem die Verlustfunktion der Trainingsdaten weiter reduziert wird, die Verlustfunktion
der Validierungsdaten aber stagniert oder sogar wieder zunimmt, so ist dies ein Anzeichen
für overfitting und die Optimierung sollte beendet werden.

In Abb.11.15 zeigen wir links den Verlauf der Verlustfunktion angewendet auf Ntr = 30
Trainingsdaten und auf Nva = 10 Validierungsdaten. Auf der rechten Seite zeigen wir
einmal das Netzwerk, welches mit 10 000 Schritten trainiert wurde und dann das Netzwerk
nach Abbruch der Minimierung bei 2500 Schritten, wo Trainings- und Validierungs-Loss
noch gut übereinstimmen. Die Trainingspunkte werden nicht mehr so gut approximiert, aber
der gesamte Verlauf des Netzes ist näher an der zugrundeliegenden Funktion f (x) = 0.

Abb.11.14 Overfitting: das
Netzwerk approximiert die
Ntr = 10 Trainingsdaten sehr
gut, gibt ansonsten aber nicht
den Verlauf der Funktion
f (x) = 0 wieder

Abb. 11.15 Links: Optimierung und Verlauf der Verlustfunktionen angewendet auf Trainings- und
Validierungsdaten. Rechts: Ergebnis der künstlichen Neuronalen Netzwerke nach 10000 Optimie-
rungsschritten und bei vorzeitigem Abbruch nach 2500 Schritten, bevor Trainings- und Validierungs-
Loss auseinanderlaufen

Beispiel 11.32 (Approximation hochdimensionaler Funktionen)

Wir betrachten die Funktion f : Rd →R, gegeben als

auf dem Hyperwürfel Q = [−1, 1]d. Wir approximieren die Funktion mit einem künst-
lichen neuronalen Netz N durch Minimierung der Verlustfunktion

für Ntr gleichverteilt zufällige Trainingswerte x1, . . . , xNtr
∈
Q. Die Netzwerk-
Architektur ist sehr einfach gehalten und besteht aus L Schichten mit jeweils n künstli-
chen Neuronen, also

Das Netzwerk hat somit nd + (L −1)n2 + n = O(Ln2) Gewichte.

Eine weitere Technik zum Vermeiden von overfitting und zur Verbesserung des Trainings-
Prozesses ist die Regularisierung der Verlustfunktion. Im einfachsten Fall bedeutet dies,
dass die Verlustfunktion (11.21) um einen quadratischen Term erweitert wird

mit dem Regularisierungsparameter γ > 0. In der mathematischen Optimierung wird diese
Methode Tikhonov Regularisierung genannt.

## Exkurs: Eigenwertbestimmung mit künstlichen neuronalen Netzen

Wir werden ein künstliches neuronales Netzwerk entwickeln, welches als Eingabe eine
symmetrische Matrix erhält und als Ausgabe alle Eigenwerte der Matrix zurückgeliefert. Als
Grundmenge betrachten wir symmetrische Matrizen, die nach folgendem Prinzip aufgebaut
sind

wobei U(0, 1) gleichverteilte Zahlen im Intervall [0, 1] sind. In Kap.5 haben wir die Kon-
ditionierung des Eigenwertproblems untersucht und schon festgestellt, dass die Eigenwerte
stetig von den Matrixeinträgen abhängen, siehe Bemerkung 5.6. Das Problem sollte sich
daher gut zur Approximation mit neuronalen Netzen eignen. Auf der anderen Seite ist die
Eigenwertbestimmung hinreichend schwer. Wir haben in Kap.5 verschiedene Algorithmen
kennengelernt. Direkte Verfahren eignen sich aufgrund der mangelnden Stabilität nicht und
die iterativen Methoden bringen alle einen hohen Aufwand mit sich.

Zunächst erstellen wir die Trainings- und Validierungsdaten. Hierzu wählen wir Ntr ∈N
und Nva ∈N zufällige Matrizen Atr
1 , . . . , Atr
Ntr ∈Xd sowie Ava
1 , . . . , Ava
Nva ∈Xd und
bestimmen jeweils die Eigenwerte �tr
i , �va
i
∈Rd. Da die Matrizen symmetrisch sind, sind
alle Eigenwerte reell und wir ordnen sie jeweils der Größe nach sortiert an, d.h.
sowie

Das künstliche neuronale Netzwerk erhält die Matrixelemente als Eingaben und die Eigen-
werte als Ausgabe. Somit gilt n0 = d2 und nL = d. In den inneren Schichten wählen wir
jeweils eine feste Zahl von nl = n Neuronen. Als Aktivierungsfunktion wird der Tangens
hyperbolicus verwendet. Bei zunächst noch freier Anzahl von Schichten ist die Architektur
gegeben als

Das neuronale Netzwerk hat also Ln + d künstliche Neuronen und die Anzahl der freien
Gewichte und Bias beträgt

Angenommen, durch A = (A1, . . . , ANtr ) und �= (�1, . . . , �Ntr ) seien die Trainingsda-
ten gegeben, dann wählen wir die Verlustfunktion

### Training des Netzes

Wir haben uns für einen grundsätzlichen Aufbau des neuronalen Netzes entschieden, müssen
nun jedoch noch die Wahl konkretisieren. Insbesondere gilt es, die Anzahl der Schichten
L sowie deren Breite n zu bestimmen. Diese Wahl kann oft nur experimentell erfolgen.
Wir starten dazu mit einem einfachen Beispiel und d = 8, d.h. symmetrischen Matrizen
A ∈R8×8. Weiter wählen wir L = 4 Schichten mit jeweils n = 64 künstlichen Neuronen.
Zum Training werden Ntr = 5000 zufällige Matrizen bestimmt. Zur Validierung wählen
wir Nva = 1250 weitere zufällige Matrizen. In Abb.11.16 links zeigen wir den Verlauf
der Verlustfunktion für Trainings- und Validierungsdaten. Hier zeigt sich klar overfitting:
das Netzwerk gibt auf den Trainingsdaten weitaus bessere Vorhersagen als auf den Validie-
rungsdaten. Gleichzeitig sieht man, dass die Optimierung noch nicht abgeschlossen ist, die
Kurve der Trainingsdaten ist nach wie vor fallend.

Obwohl wir nicht davon ausgehen, dass das resultierende Netzwerk gute Vorhersagen
liefert, geben wir in Tab.11.4 für zwei zufällig gewählte Matrizen (die nicht in den Trainings-
daten enthalten sind) die Eigenwerte �und die vom Netzwerk approximierten Eigenwerte
�N jeweils auf drei Stellen gerundet an.

Das Netzwerk trainiert zu spezifisch auf den gegebenen Daten. Eine Möglichkeit, over-
fitting zu vermeiden ist die Hinzunahme eines Regularisierungsterms in die Verlustfunk-
tion (11.36). Die einfachste Form der Regularisierung ist die Tikhonov Regularisierung,
welche die zu bestimmenden Parameter als quadratische Terme hinzufügt, d.h.

Abb.11.16 Links: Verlauf der Verlustfunktion für Trainings- und Validierungsdaten zur Eigenwert-
bestimmung von symmetrischen 8×8 - Matrizen. Es ist klar overfitting zu erkennen. Rechts: Training
desgleichenNetzesmitRegularisierung.DieVerlustfunktionwirdumdenTermγ �
l ∥Wl∥2
F ergänzt

Tab. 11.4 Eigenwerte und Approximation des neuronalen Netzwerks für zwei zufällig gewählte
Matrizen, die nicht in den Trainingsdaten enthalten sind. Der relative Fehler ist zwischen Eigenwerten
λi und Netzwerkapproximation λN
i
ist gegeben als (λi −λN
i )/ maxi |λi|

wobei γ > 0 ein kleiner Parameter ist. Wir wählen γ = 10−4 und in Abb.11.16 rechts ist
der Verlauf der beiden Verlustfunktionen gezeigt. Overfitting kann so vermieden werden. Es
zeigt sich jedoch auch, dass die Verlustfunktion nicht mehr entsprechend reduziert werden
kann. Die Wahl des Parameters γ ist nicht einfach. Bei γ = 10−5 verliert die Regulari-
sierung bei diesem Beispiel ihre Wirkung. Die Wahl γ = 10−3 führt zu einem Verlust der
Approximationseigenschaft.
Tab.11.5 zeigt die Ergebnisse für zwei Testmatrizen unter Verwendung von Regulari-
sierung. Alle Eigenwerte können mit einem Fehler von maximal 5 % vorhergesagt werden.
Dabei bezieht sich die Normierung im relativen Fehler jeweils auf den betragsgrößten Eigen-
wert der Matrix.

### Einfluss der Netzwerkgröße

Die theoretische Approximationsgüte eines neuronalen Netzes ist der grundlegende Faktor
der bestimmt, wie gut ein Problem überhaupt gelöst werden kann. Man nennt dies die
Expressivität des Netzes. Wir untersuchen zunächst den Einfluss der Anzahl der Neuronen
n pro Schicht. D.h., wir bleiben bei der Architektur N(tanh; d2, n, n, n, n, d) und variieren

Tab.11.5 Training mit Regularisierung. Eigenwerte und Approximation des neuronalen Netzwerks
für zwei zufällig gewählte Matrizen, die nicht in den Trainingsdaten enthalten sind

die Neuronen pro Schicht. Für n ∈{2, 4, 8, 16, 32, 64, 128, 256} trainieren wir jeweils ein
Netzwerk und wenden es im Anschluss auf 1000 zufällige Matrizen an. Für diese bestimmen
wir jeweils die relativen Fehler und bilden die Ergebnisse in Abb.11.17 links ab. Wir zeigen
jeweils den Mittelwert sowie die Standardabweichung der Fehler.
Wie erwartet zeigt sich, dass größere Netzwerke besser in der Lage sind, die Eigen-
werte zu approximieren. Die Konvergenz ist allerdings nur langsam. Dies kann an weiteren
Einflüssen, wie der Anzahl der Schichten oder der Wahl des Regularisierungsparameters
γ liegen, die wir hier nicht angepasst haben. Schließlich halten wir auch die Anzahl der
Trainingsdaten fest. Es gilt für alle Fälle Ntr = 10 000.

Entsprechend zu diesem Test halten halten wir nun n = 64 fest und variieren die
Zahl der Schichten L von L = 1 bis L = 8. D.h., das kleinste Netz hat die Architek-
tur N(tanh; d2, n, d) und das größte hat 8 innere Schichten. Wir zeigen die Ergebnisse in
Abb.11.17 rechts. Tiefere Netze haben bessere Approximationseigenschaften. Aber auch
hier zeigt sich, dass die Verbesserung ab einer gewissen Tiefe stagniert. Dies bedeutet nicht,
dass das Netzwerk dann keine bessere Approximationseigenschaft hat, sondern eher, dass
wir entweder die Anzahl der Trainingsdaten erhöhen müssen, evtl. den Regularisierungs-
parameter γ anpassen müssen oder auf andere Weise in den Optimierungsprozess eingreifen
sollten. Die Verbesserung des ganzen Aufbaus, also die Wahl des Netzes, der Daten, der
Methode zur Optimierung und der Regularisierung wird die Optimierung der Hyperpara-
meter genannt. Hier ist meist viel Probieren und Handarbeit gefragt.

### Einfluss der Trainingsdaten

Wir haben gesehen, dass es nur bis zu einer bestimmten Grenze Sinn macht, die Komplexität
des neuronalen Netzwerkes zu steigern. Ab einer bestimmten Grenze führt dies nicht zu

Abb. 11.17 Links: Mittlerer Fehler und Standardabweichung bei wachsender Anzahl von künstli-
chen Neuronen n pro Schicht. Es gilt stets L = 4. Rechts: Mittlerer Fehler und Standardabweichung
bei Variation der Anzahl der Schichten L. Hier gilt immer n = 64

einer Verbesserung der Anwendung auf Testdaten. Dies liegt daran, dass wir die Anzahl der
Trainingsdaten immer gleich belassen haben. In Abb.11.18 zeigen wir nun die Abhängigkeit
der Netzwerkvorhersage für eine wachsende Anzahl von Trainingsdaten. Das Netz selbst hat
L = 4 Schichten mit jeweils n = 64 künstlichen Neuronen (linke Abbildung) und n = 128
in der rechten Abbildung.
Es zeigt sich, dass die Approximationsgüte mit steigender Anzahl von künstlichen Neuro-
nen steigt. Aber wieder ist schnell eine Grenze erreicht. Von Ntr = 1000 bis zu Ntr = 4000
wird der Fehler schnell kleiner, danach stagnieren die Werte jedoch. Ein weiterer Effekt ist
jedoch doch stark verringerte Varianz der Ergebnisse. Dies liegt natürlich daran, dass der
Generalisierungsfehler geringer ist. Dieser ist durch den Abstand zwischen einer Validie-
rungsmatrix A und den Testdaten gegeben,

In der rechten Abbildung wiederholen wir das Experiment mit doppelter Anzahl von künst-
lichen Neuronen.

### Generalisierung

Bisher haben wir die Netzwerke nur mit Matrizen getestet, die der gleichen Xd entstammen.
Insbesondere waren die Einträge der Matrizen bisher alle normalverteilte Zahlen mit Ai j ∈
[0, 2], vergleiche (11.35) (die Einträge von B liegen in Bi j ∈[0, 1]). Wir wenden ein bereits
trainiertes Netzwerk zunächst auf Matrizen an, deren Einträge normalverteilt sind

Abb. 11.18 Validierungsfehler (Mittelwert und Standardabweichung) bei steigender Anzahl von
Trainingsdaten. Links zeigen wir das Ergebnis für das Netzwerk N(tanh; d2, 64, 64, 64, 64, d)
und recht für ein Netzwerk mit doppelter Anzahl von Neuronen pro Schicht, also N(tanh; d2,
128, 128, 128, 128, d)

wobei N(0.5) eine normalverteilte Zahl mit Standardabweichung 1 und Mittelwert 0.5 ist.
Man nennt dies die Generalisierung eines künstlichen neuronalen Netzes: Wie gut ist die
Methode auf Daten anzuwenden die beim Training überhaupt nicht vorgekommen sind?
Wir trainieren hierzu ein Netzwerk mit der Architektur

mit Ntrain = 16 000 Trainingsmatrizen. Das Training wird nach etwa 40 000 Iterationen
abgebrochen, wenn Trainings- und Validierungsfehler beide noch nahe beieinander liegen.
In Tab.11.6 zeigen wir den mittleren Fehler für die Vorhersage der Eigenwerte von
jeweils 1000 zufälligen Matrizen. Zusätzlich geben wie die Standardabweichung der 1000
zufälligen Matrizen an als Maß für die Streuung der Fehler. Wir vergleichen Elemente der
Trainingsdaten mit Elementen der Validierungsdaten (also zufälligen Matrizen, die zwar
gleichverteilte Einträge haben, aber nicht Teil des Trainings waren) und Generalisierungs-
daten aus Yd. Auf diesem Datensatz liefert das Netzwerk keine brauchbaren Vorhersagen,
siehe Tab.11.6. Die Eigenwerte der Generalisierungsdaten haben einen Vorhersagefehler
von durchschnittlich 50 %. Dies ist nicht weiter verwunderlich, da die Matrizen A ∈Yd
Einträge haben, die nicht im Intervall [0, 2] liegen müssen, wie es bei allen Trainingsdaten
der Fall ist. Normalverteilte Zahlen können größer oder kleiner sein und Einträge dieser
Größenordnung hat unser Netzwerk noch nie gesehen. Daher ist auch nicht zu erwarten,
dass die Methode gut generalisiert. Wir modifizieren das Beispiel daher etwas: für eine
Matrix A ∈Yd bestimmen wir zunächst die betragsgrößte Abweichung vom Mittelwert 1.

Mit diesem Element wird die Matrix skaliert

Tab. 11.6 Generalisierung des künstlichen neuronalen Netzes zur Vorhersage der Eigenwerte einer
Matrix. Für 1000 zufällige Matrizen der Trainings- und Testdaten, sowohl für 1000 weitere, diesmal
normalverteilte Matrizen, geben wir den mittleren Vorhersagefehler sowie die Standardabweichung
der Fehler an

und im Anschluss wird ˜A wieder zum Mittelwert 1 verschoben. Für einen Eigenvektor x
und Eigenwert λ von A gilt

Das Netzwerk kann auf ˜A angewendet werden und der resultierende Eigenwert ˜λ werden
gemäß

skaliert zu Eigenwerten von A. Die Ergebnisse in der letzten Zeile von Tab.11.6 zeigen,
dass so eine sehr gute Generalisierung auf Matrizen mit normalverteilten Werten erreicht
werden kann.
Schließlich untersuchen wir was passiert, wenn wir die Methode auf Matrizen anwenden,
die überhaupt nicht mehr symmetrisch sind. Hierzu wählen wir Matrizen der Art

wobei α ein kleiner Parameter ist. Die eigentlich symmetrische Matrix Ax ∈Xd wird
mit normalverteilten Zahlen gestört. Die Ergebnisse sind in Abb.11.19 gezeigt für α =
0, 0.05, 0.1, . . . , 0.5. In Tab.11.7 gegeben wir für α = 0, 0.2 und α = 0.4 die 8 Eigenwerte
für jeweils eine einzelne zufällig gewählte Matrix an.