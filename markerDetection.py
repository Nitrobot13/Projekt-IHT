from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from PIL import ImageShow
import numpy as np
import math
from scipy.signal import convolve2d

"""
Hilfsklasse "Edges". Darin wird Position und Ausrichtung einer Ecke gespeichert. Man kann evtl besser damit arbeiten,
#bzw bessere Boundaries setzen für die Kantenerkennung (siehe mark Edges und Bresenham-Algo)
"""
class Edge:
    def __init__(self, posX, posY, alignmentX, alignmentY):
        self.posX = posX
        self.posY = posY
        self.alignmentX = alignmentX
        self.alignmentY = alignmentY

"""
Testfunktion für mich, zur Ausgabe eines Bildes
"""
def showImage():
    #Dateipfad mit Fotodatei
    #Verwendung von 'r' vor dem String um "rohen String zu benutzen, dadurch werden keine Sonderzeichen ausgewertet
    image_path = r'C:\Users\Constantin Reinert\Documents\Studium\Institutsprojekt\Analyse\Testbild.jpg'

    #Bild öffnen
    img = Image.open(image_path)

    #Bild anzeigen
    img.show()

"""
Formatieren des Bildes mithilfe des Sobel-Filters. 
Problem:
- Erkennbarkeit ist noch nicht exakt genug. Ich werde daran nochmal weiterarbeiten. Ich habe nochmal eine Rauschunterdrückung hinter Sobel gepackt, aber tbh,
das gibt dem ganzen nicht wirklich viel, also whatever. 
"""

def edgeDetection(Bild):
    img = Image.open(Bild)

    #Bildformat konvertieren um Zugriff auf die Pixel zu bekommen
    img = img.convert('RGB')

    #Höhe und Breite des Bildes
    width, length = img.size

    #Padding für den Filter
    img = ImageOps.expand(img, border=1,)

    #Kontrast erhöhen und SW-Filter, damit die Kanten besser erkannt werden können.
    constrastEnhancer = ImageEnhance.Contrast(img)
    imageEnhanced = constrastEnhancer.enhance(4.0)
    imgBW = ImageOps.grayscale(imageEnhanced)

    #Neues Bild für die Rauschunterdrückung und Faltung
    convolutedImg = Image.new('L', (width, length))

    #Rauschunterdrückung
    for x in range(width):
        for y in range(length):
            pixValue = [
                imgBW.getpixel((x, y)),
                imgBW.getpixel((x + 1, y)),
                imgBW.getpixel((x - 1, y)),
                imgBW.getpixel((x - 1, y - 1)),
                imgBW.getpixel((x, y - 1)),
                imgBW.getpixel((x + 1, y - 1)),
                imgBW.getpixel((x - 1, y + 1)),
                imgBW.getpixel((x, y + 1)),
                imgBW.getpixel((x + 1, y + 1))
            ]
            pixValue = sum(pixValue) // 9
            convolutedImg.putpixel((x, y), pixValue)


    #2D-Faltung (Sobel-Operator)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    #Konvertieren des Bildes in ein NumPy-Array
    img_array = np.array(convolutedImg)

    #Anwenden der Faltung für Kantenerkennung in x- und y-Richtung (Sobel-Filter)
    edges_x = convolve2d(img_array, sobel_x, mode='same', boundary='symm')
    edges_y = convolve2d(img_array, sobel_y, mode='same', boundary='symm')

    #Gesamtkantenerkennung (Betrag der Gradienten)
    edges = np.sqrt(edges_x ** 2 + edges_y ** 2).astype(np.uint8)  # Konvertierung in uint8 für PIL-Image

    convolutedImg2 = Image.new('L', (width, length))

    #Konvertieren des NumPy-Arrays zurück in ein PIL-Bild
    convolutedImg = Image.fromarray(edges)

    #Rauschunterdrückung 2
    for x in range(width):
        for y in range(length):
            pixValue = [
                convolutedImg.getpixel((x, y)),
                convolutedImg.getpixel((x + 1, y)),
                convolutedImg.getpixel((x - 1, y)),
                convolutedImg.getpixel((x - 1, y - 1)),
                convolutedImg.getpixel((x, y - 1)),
                convolutedImg.getpixel((x + 1, y - 1)),
                convolutedImg.getpixel((x - 1, y + 1)),
                convolutedImg.getpixel((x, y + 1)),
                convolutedImg.getpixel((x + 1, y + 1))
            ]
            pixValue = sum(pixValue) // 9
            convolutedImg2.putpixel((x, y), pixValue)

    #Ordner in dem die Fotodatei gespeichert wird
    output_path = r'C:\Users\Constantin Reinert\Documents\Studium\Institutsprojekt\Analyse\GefaltetesBild.jpg'
    convolutedImg2.save(output_path)

    convolutedImg2.show()  




"""
Die markEdges(Bild)-Funktion ist noch nicht fertig, die Idee ist grundsätzlich ziemlich einfach. Man nimmt sich die Ecken raus und verbindet jeweils 2 mit einer Geraden.
Sind die Eckpunkte richtig gewählt,sollte eine gute Approximation einer Kante sehr gut möglich sein. Im nächsten Schritt könnte man dann den Schnittwinkel zwischen den
zwei erzeugten Geraden berechnen (über deren Steigung, findet man im Internet falls nicht sowieso schon vorhanden).
Noch vorhandene Probleme:

- Die Erkennung der Ecken hat ein strukturelles Problem, alle nach unten geöffnete Kanten werden nicht erkannt. (Ich werde mir das nochmal anschauen, ich hab irgendwo 
gepennt, die Ausrichtung des Koordinatensystems ist anders, dass Bild liegt untypischerweise nicht im ersten Quadranten)
-> Wenn das besser funktioniert, könnte man mithilfie von OpenCV die Eckpunkte auslesen und damit weiterarbeiten.

- Die Bedingung, nach der zwei Kanten ausgewählt werden, ist aktuell nur beschränkt auf zwei aufeineranderfolgende Ecken in meiner Liste. Sinnvolle Punkte auszuwählen 
könnte schwierig sein. Ich habe deswegen die Hilfsklasse Edges implementiert. Die noch unausgereifte Idee war, Eckpunkte nach ihrere Ausrichtung zu wählen.
Wenn jmd eine gute Idee diesbezüglich hat, einfach mal ausprobieren
"""

def markEdges(Bild):
    img = Image.open(Bild)

    #Liste zum speichern der Eckenpositionen
    cornerSave = []
    cornerSave.clear()

    # Bildformat konvertieren um Zugriff auf die Pixel zu bekommen
    img = img.convert('RGB')

    # Höhe und Breite des Bildes
    width, length = img.size

    #Vergleich
    threshold = (30, 30, 30)

    #Kantenbedingung
    condition = 20

    #Markerfarbe:
    rot = (255, 0, 0)

    #Sichtbarkeitsfarbe:
    rot1 = (254, 0, 0)

    #Raster
    for x in range(width):
        for y in range(length):
            currentColor = img.getpixel((x,y))

            if currentColor > threshold:
                #Prüfe Ausrichtung
                right = 0
                left = 0
                up = 0
                down = 0
                for n in range(condition):
                    if x + n < width:
                        if  img.getpixel((x+n, y)) > threshold:
                            right+=1
                    if x - n >= 0:
                        if img.getpixel((x-n, y)) > threshold:
                            left+=1
                    if y + n < length:
                        if img.getpixel((x, y+n)) > threshold:
                            up+=1
                    if y - n >= 0:
                        if img.getpixel((x, y-n)) > threshold:
                            down+=1
                        else:
                            break
                if up == condition and right == condition and down < condition and left < condition:
                    img.putpixel((x, y), rot)
                    cornerSave.append((x, y, "right", "up"))
                if up == condition and right < condition and down < condition and left == condition:
                    img.putpixel((x, y), rot)
                    cornerSave.append((x, y, "left", "up"))
                if up < condition and right == condition and down == condition and left < condition:
                    img.putpixel((x, y), rot)
                    cornerSave.append((x, y, "right", "down"))
                if up < condition and right < condition and down == condition and left == condition:
                    img.putpixel((x, y), rot)
                    cornerSave.append((x, y, "left", "down"))

    #Größere Marker für bessere Sichtbarkeit
    for x in range(width):
        for y in range(length):
            if img.getpixel((x,y)) == rot:
                for i in range(12):
                    if x + i < width:
                        img.putpixel((x + i, y), rot1)  # rechts
                    if x - i >= 0:
                        img.putpixel((x - i, y), rot1)  # links
                    if y + i < length:
                        img.putpixel((x, y + i), rot1)  # unten
                    if y - i >= 0:
                        img.putpixel((x, y - i), rot1)  # oben

    #Punkte werden zu Kanten verbunden durch "Bresenham"-Algo
    def bresenham_line(x1, y1, x2, y2):
        #Berechnet die Pixel auf einer Linie zwischen (x1, y1) und (x2, y2) mithilfe des Bresenham-Algorithmus.
        pixels = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            pixels.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return pixels

    #Zeichnen der Kanten ins Bild
    for i in range(len(cornerSave) - 1):
        p1 = (cornerSave[i][0], cornerSave[i][1])
        p2 = (cornerSave[i + 1][0], cornerSave[i + 1][1])
        x1, y1 = p1
        x2, y2 = p2
        line_pixels = bresenham_line(x1, y1, x2, y2)
        for pixel in line_pixels:
            img.putpixel(pixel, (0, 255, 0))

    output_path = r'C:\Users\Constantin Reinert\Documents\Studium\Institutsprojekt\Analyse\GeecktesBild.jpg'
    img.save(output_path)

    img.show()








