from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from PIL import ImageShow
import numpy as np
from scipy.signal import convolve2d


def showImage():
    #Dateipfad mit Fotodatei
    #Verwendung von 'r' vor dem String um "rohen String zu benutzen, dadurch werden keine Sonderzeichen ausgewertet
    image_path = ''

    #Bild öffnen
    img = Image.open(image_path)

    #Bild anzeigen
    img.show()

def edgeDetection(Bild):
    img = Image.open(Bild)

    #Bildformat konvertieren um Zugriff auf die Pixel zu bekommen
    img = img.convert('RGB')

    #Höhe und Breite des Bildes
    width, length = img.size

    #Markerfarbe:
    rot = (255, 0, 0)

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

    #Konvertieren des NumPy-Arrays zurück in ein PIL-Bild
    convolutedImg = Image.fromarray(edges)

    #Ordner in dem die Fotodatei gespeichert wird
    output_path = ''
    convolutedImg.save(output_path)

    convolutedImg.show()








