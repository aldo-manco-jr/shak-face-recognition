import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


"""
il percorso all'interno del workspace dove poter trovare delle cartelle
dove ogni cartella rappresenta una persona
ogni cartella contiene una lista di foto della persona che rappresenta
"""

PATH_PEOPLE_FOLDERS_LIST = 'images-attendance'


"""
inizializzazione dell'array che conterrà la lista di tutte le immagini
presenti all'interno della cartella dedicata alle foto (images-attendance)
foto da cui l'algoritmo estrae le features appartenenti a tutte le foto di quella persona
"""

imagesList = []


"""
inizializza l'elenco delle etichette collegate a ciascuna foto
ogni etichetta è il nome della persona raffigurata nella foto
ad essa viene assegnato il nome della cartella a cui viene dato il nome della persona 
esempio di etichetta: "Aldo Manco"
"""

namesFacesList = []


"""
lista contenente i nomi delle cartelle all'interno della cartella dedicata alle foto (images-attendance)
dove ogni cartella rappresenta una persona
ogni nome di una cartella è il nome della persona
ogni cartella contiene tutte le foto relative a quella persona
"""

peopleFoldersList = os.listdir(PATH_PEOPLE_FOLDERS_LIST)
peopleFoldersList.pop()


"""
per ogni cartella che rappresenta una persona all'interno di images-attendance
creiamo un array contenente la lista di tutti filename delle foto all'interno di quella cartella
che raffigurano la persona rappresentata dalla cartella

per ogni filename all'interno di questo array
inseriamo il filename dell'immagine all'interno di un array definito in precedenza senza l'estensione (.jpg)
che conterrà tutte le immagini di tutte le persone
di cui l'algoritmo deve fare una mappatura delle features
"""

for personFolder in peopleFoldersList:
    pathPersonPhotoFilenameList = PATH_PEOPLE_FOLDERS_LIST + "/" + personFolder
    personPhotoFilenameList = os.listdir(pathPersonPhotoFilenameList)

    for personPhotoFilename in personPhotoFilenameList:
        currentImage = cv2.imread(f'{pathPersonPhotoFilenameList}/{personPhotoFilename}')
        print(pathPersonPhotoFilenameList + "/" +personPhotoFilename)
        imagesList.append(currentImage)
        namesFacesList.append(os.path.splitext(personFolder))


"""
funzione che calcola le features di ogni faccia presente all'interno di foto
per ogni foto presente all'interno di una lista di foto

facesListEncodingsList[]
- matrice MxN
    M -> numero delle foto
    N -> numero delle features per ogni faccia (128)
- contiene la lista delle features per ogni faccia

ogni foto all'interno della lista di tutte le foto del data set
- viene convertita da BGR a RGB
- vengono calcolate le features per ogni faccia all'interno dell'immagine
- vengono aggiunte le features di ogni faccia all'interno di un array che viene restituito 
"""

def findFacesEncodings(imagesList):

    facesListEncodingsList = []

    for image in imagesList:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faceEncodingsList = face_recognition.face_encodings(image)[0]
        facesListEncodingsList.append(faceEncodingsList)
    return facesListEncodingsList


"""
funzione che segna:
- nome della persona
- data del rilevamento
la prima volta che quella persona viene rilevata davanti la webcam in un file excel

File CSV (.csv), file strutturato dove:
- le colonne sono separate da virgole sulla stessa riga
- le righe sono separate andando a capo  

passaggi:
- apre il file CSV (ledgerAttendance.csv)
- legge il contenuto del file CSV
- salva tutti i nomi registrati nel file CSV in un array
- in ogni frame registrato dalla webcam
    ogni persona non presente nell'array che viene identificata
    viene registrata come un nuovo record nel file CSV
    dove viene assegnata la data e l'ora attuali oltre al nome
"""

def markPersonAttendance(namePersonFound):
    with open('ledgerAttendance.csv', 'r+') as file:
        ledgerAttendance = file.readlines()
        ledgerNamesList = []
        print(ledgerAttendance)
        for line in ledgerAttendance:
            entry = line.split(',')
            ledgerNamesList.append(entry[0])
        if namePersonFound not in ledgerNamesList:
            now = datetime.now()
            datetimeString = now.strftime('%d/%m/%g %H:%M:%S')
            file.writelines(f'\n{namePersonFound},{datetimeString}')


"""
tramite la funzione findFacesEncodings() definita in precedenza
otteniamo facesListEncodingsList[]
- matrice MxN
    M -> numero delle foto
    N -> numero delle features per ogni faccia (128)
- contiene la lista delle features per ogni faccia rilevata nelle foto del data set
"""

facesListEncodingsList = findFacesEncodings(imagesList)
print('Encoding Complete')


"""
OpenCV inizializza la webcam con ID=0
"""

webcam = cv2.VideoCapture(0)


#per ogni frame catturato dalla webcam finché il programma è aperto
while True:

    # ricaviamo l'immagine originale senza compressione dalla webcam
    # riceviamo un messaggio che avverte se l'operazione è andata a buon fine o meno
    success, image = webcam.read()

    # compressione dell'immagine di 1/4 rispetto alla scala originale
    # secondo e terzo parametro indicano che non vogliamo specificare una risoluzione forzata a cui l'immagine si deve adattare
    # ma vogliamo lavorare sul ridimensionamento mantenendo la stessa proporzione
    compressedImage = cv2.resize(image, (0, 0), None, 0.25, 0.25)

    # converte l'immagine compressa del frame catturato dalla webcam da BGR a RGB
    compressedImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ricaviamo la lista delle coordinate per ciascuna faccia rilevata
    # dall'algoritmo di Face Detection Landmarks nel frame catturato dalla webcam
    frameFaceLocationsList = face_recognition.face_locations(image)

    # ricaviamo la lista delle 128 features per ogni faccia
    # presente in ciascuna delle coordinate trovate in precedenza nel frame catturato dalla webcam
    frameFaceEncodingsList = face_recognition.face_encodings(image, frameFaceLocationsList)

    # funzione zip() unisce 0 o più elementi iterabili (array, liste, set, ...)
    # creando un'unica lista in cui gli elementi sono tuple
    # ogni tupla contiene i campi di ogni elemento iterabile ad un indice specifico
    # qui stiamo creando un iterabile che contiene per ogni faccia nel frame catturato dalla webcam
    # una tupla con questa struttura:
    # - coordinate della posizione in cui la faccia è presente nel frame
    # - lista di features che rappresentano delle caratteristiche che prese insieme rendono univoca quella particolare faccia
    frameFacesLocationsEncodings = zip(frameFaceEncodingsList, frameFaceLocationsList)

    # per ogni tupla contenente:
    # - posizione
    # - features
    # di ogni faccia nel frame
    for currentFrameFaceEncodings, currentFrameFaceLocation in frameFacesLocationsEncodings:

        # confronta la faccia del frame con tutte le facce conosciute presenti all'interno del data set
        # restituisce un array di valori boolean
        # che indicano se c'è corrispondenza per ognuna delle 128 features calcolate dall'algoritmo
        facesMatches = face_recognition.compare_faces(facesListEncodingsList, currentFrameFaceEncodings)

        # calcola la distanza tra la faccia del frame con tutte le facce conosciute presenti all'interno del data set
        # restituisce un array di numeri razionali in [0, 1]
        # che indicano la distanza per ognuna delle 128 features calcolate dall'algoritmo
        # minore è il numero, maggiore è la compatibilità
        facesDistances = face_recognition.face_distance(facesListEncodingsList, currentFrameFaceEncodings)

        # stampa i risultati del confronto e la relativa distanza
        print(facesMatches)
        print(f"{facesDistances}\n")

        # trova il valore minimo nella lista delle distanze
        # la faccia con quella distanza dalla faccia del frame
        # rappresenta la faccia più simile nella lista dei volti noti
        bestMatchIndex = np.argmin(facesDistances)

        # se le 2 facce vengono interpretate dalla libreria face_recognition come stessa persona
        if facesMatches[bestMatchIndex]:

            # ricavo il nome della cartella che contiene la foto del data set
            # che indica il nome di quella persona
            bestMatchPersonName = namesFacesList[bestMatchIndex][0]
            print(bestMatchPersonName)

            # ricavo le coordinate per identificare il volto nel frame
            Y1, X2, Y2, X1 = currentFrameFaceLocation
            # Y1, X2, Y2, X1 = Y1*4, X2*4, Y2*4, X1*4

            # stampa un riquadro di delimitazione tramite OpenCV
            # sulle coordinate che indicano la posizione del volto
            cv2.rectangle(image,
                          (X1, Y1),
                          (X2, Y2),
                          (0, 67, 23),
                          2)

            # stampa un rettangolo colorato sotto il riquadro di delimitazione
            # che conterrà il nome della persona che corrisponde al volto
            cv2.rectangle(image,
                          (X1, Y2-35),
                          (X2, Y2),
                          (0, 67, 23),
                          cv2.FILLED)

            print(bestMatchPersonName)

            # stampa nel riquadro colorato il nome della persona con il volto presente nel frame
            cv2.putText(image,
                        bestMatchPersonName,
                        (X1+6, Y2-6),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        2)

            # chiamo la funzione definita in precedenza markPersonAttendance()
            # per registrare la persona all'interno del registro CSV
            # solo nel caso essa non sia stata già registrata
            markPersonAttendance(bestMatchPersonName)

    # apre una finestra sul sistema operativo chiamata "Shakkam"
    # che mostra il fotogramma corrente ripreso dalla webcam
    cv2.imshow('Shakkam', image)

    # waitKey(0) mostra la finestra finché viene rilevata la pressione di un qualsiasi tasto (è adatto per la visualizzazione dell'immagine)
    # - vedi un'immagine fissa finché non premi un tasto

    # waitKey(1) mostra un frame per 1 ms, dopodiché il display verrà chiuso automaticamente
    # - la funzione mostrerà un frame solo per 1 ms
    cv2.waitKey(1)
