from tkinter import * #Interfaz
import cv2 #PROCESAMIENTO DE IMAGENES
import sounddevice as sd #SONIDO
import soundfile as sf
import threading as th
import time
import serial

# Configura la conexión serial con Arduino
puerto = 'COM14'  # Reemplaza 'COMX' con el nombre de tu puerto COM
baud_rate = 9600

arduino = serial.Serial(puerto, baud_rate)


#-----------FUNCION PARA MOVER SERVOMOTOR-----------
def move_servo(angulo):
    arduino.write(str(angulo).encode())
    print(f"Moviendo el servo a {angulo} grados.")




#--------------JUEGO------------------
def jugar():
    print("Jugamos")
    #Parametros
    global cap, mov, contador,rostro,jugadores

    #CANTIDAD DE JUGADORES
    contador=0;

    #SACAMOS EL NUMERO DE JUGADORES
    jugadores = entrada.get()
    jugadores = int(jugadores)

    print(jugadores)

    # Deteccion de rostros
    rostro = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #Video captura
    cap= cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)

    #LLAMAMOS AL METODO DE DETECCION DE MOVIMIENTO
    mov=cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=2500,detectShadows=False)

    #DESABILITAMOS OPEN CL
    cv2.ocl.setUseOpenCL(False)

    #FUNCION DE AUDIO
    def audio(archivo):
        global hilo, inicio
        inicio=time.time()
        #leemos el audio
        data,fs=sf.read(archivo)
        #REPRODUCIMOS SONIDO
        sd.play(data,fs)

    #FUNCION PARA PREGUNTAR SI TERMINO EL AUDIO

    def check2(hilo):
        fin= time.time()
        tiempo= int(fin-inicio)
        #print (tiempo)
        if tiempo>6:
            Verde()

    def check(hilo):
        fin= time.time()
        tiempo= int(fin-inicio)
        #print(tiempo)
        if tiempo>6:
            Roja()

    #FUNCION LUZ VERDE
    def Verde():


        #ASIGNAMOS HILO
        archivo='luzVerde.wav'
        hilo= th.Thread(target=audio,args=(archivo,))
        hilo.start()
        move_servo(180)

        #Creamos nuestro while true

        while True:
            check(hilo)
            #lectura de la video captura
            ret,frame=cap.read()

            #filtro gaussiano
            filtro=cv2.GaussianBlur(frame,(31,31),0)

            #metodo de deteccion de movimiento
            mascara=mov.apply(filtro)

            #Creamos una copia
            copy=mascara.copy()

            #BUSCAMOS CONTORNOS
            contornos,jerarquia=cv2.findContours(copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            #MOSTRAMOS  LOS JUGADORES vivos

            cv2.putText(frame,f"JUGADORES VIVOS: {str(jugadores)}", (400,50), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

            #DIBUJAMOS LOS CONTORNOS
            for con in contornos:

                #BORRAMOS CONTORNOS PEQUEÑOS
                if cv2.contourArea(con)< 5000:
                    continue
                #coordenadas del contorno
                (x,y,an,al)=cv2.boundingRect(con)


                #dibujamos rectangulo
                cv2.rectangle(frame,(x,y),(x+an,y+al),(0,255,0),2)
            #Mostramos los frames
            cv2.imshow("LUZ VERDE LUZ ROJA",frame)

            #ROMPER WHILE CONDICION
            t=cv2.waitKey(1)
            if t==27:
                cerrar()

    def Roja():
        move_servo(0)
        #Declaracion
        global jugadores, contador,dis

        #variable para disminuir jugador
        dis=0

        #reproducir sonido
        archivo= "tenso.wav"
        hilo=th.Thread(target=audio, args=(archivo,))
        hilo.start()
        move_servo(0)

        #creamos while true
        while True:
            check2(hilo)

            #video captura lectura
            ret,frame=cap.read()

            #filtro gaussiano
            filtro= cv2.GaussianBlur(frame,(31,31),0)

            #Aplicamos metodo de deteccion de movimiento
            mascara = mov.apply(filtro)

            #Creamos copia
            copy = mascara.copy()


            #buscamos contornos
            contornos, jerarquia=cv2.findContours(copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            #mostrmoas jugadores vivos
            cv2.putText(frame, f"JUGADORES VIVOS: {str(jugadores)} ", (400, 50), cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)

            #Dibujamos los contornos
            for con in contornos:
                #borramos  los contornos pequeños
                if cv2.contourArea(con)<5000:
                    continue

                #coordenadas del contorno
                (x,y,an,al)=cv2.boundingRect(con)

                #detecto rostro
                copia= frame.copy()
                gris=cv2.cvtColor(copia,cv2.COLOR_BGR2GRAY)
                caras= rostro.detectMultiScale(gris,1.3,5)

                #detecto rostro de jugador perdedor

                for (x2,y2,an,al) in caras:

                    #dibujamos rectangulo en el rostro
                    cv2.rectangle(frame,(x2,y2),(x2+an,y2+al),(0,0,255),2)

                    #Escribimos jugador eliminado
                    cv2.putText(frame, f"JUGADOR{str(contador)} ELIMINADO ", (x2-70,y2-5), cv2.FONT_HERSHEY_PLAIN, 2,
                                (0, 0, 255), 2)
                    #NUMERO DE JUGADORES MUERTOS
                    muerte= len(caras)

                    #disminuimos contados
                    if dis==0:
                        # reproducir sonido
                        archivo3 = "disparos.wav"
                        # leemos el audio
                        data3, fs3 = sf.read(archivo3)
                        # REPRODUCIMOS SONIDO
                        sd.play(data3, fs3)


                        contador=contador+muerte

                        #disminuimos el numero de jugadores vivos
                        jugadores=jugadores-muerte

                        #cambio la llave
                        dis=1

                        if jugadores==0:
                            #cerramos juego
                            cerrar()
            #mostramos los frames
            cv2.imshow("LUZ VERDE LUZ ROJA",frame)

            #Condicion para romper while
            t=cv2.waitKey(1)
            if t==27:
                cerrar()

    #Llamamos a la funcion
    Verde()
#---------------------- CERRAR JUEGO---------------------
def cerrar():
    print("Cerramos")
    #cerramos ventanas
    cv2.destroyAllWindows()
    cap.release()

    #Mostramos pantalla final
    global pantalla2
    pantalla2=Toplevel()
    pantalla2.title("EL JUEGO DEL CALAMAR")
    pantalla2.geometry("1280x720")
    imagen2=PhotoImage(file="final.png")


    #PLANTILLA
    plantilla2=Canvas(pantalla2,width=1280,height=720)

    fondo=Label(pantalla2,image=imagen2)
    fondo.place(x=0,y=0,relwidth=1,relheight=1)

    plantilla2.pack()
    plantilla2.mainloop()

#---------------PANTALLA PRINCIPAL------------------
def pantalla_principal():
    global pantalla,entrada
    pantalla = Tk()
    pantalla.title("EL JUEGO DEL CALAMAR")
    imagen = PhotoImage(file="principal.png")
    pantalla.geometry("%dx%d" % (imagen.width(), imagen.height()))



    #PLANTILLA DE DISEÑO
    plantilla1= Canvas(pantalla,width=1280, height=720)
    plantilla1.pack(fill="both",expand=True)
    plantilla1.create_image(0,0,image=imagen,anchor = "nw")

    #Imagen boton1
    img1=PhotoImage(file="jugar.png")
    # Imagen boton2
    img2 = PhotoImage(file="fin.png")

    #BOTONES
    #Boton1
    boton1=Button(pantalla,text="JUGAR",height="40",width="300",command=jugar,image=img1)
    boton1pla=plantilla1.create_window(310,580,anchor="nw",window=boton1)

    #BOTON2
    boton2 = Button(pantalla, text="CERRAR", height="40", width="300", command=cerrar, image=img2)
    boton2pla = plantilla1.create_window(705, 580, anchor="nw", window=boton2)

    #ENTRADA DE JUGADORES
    jugadores = StringVar()
    entrada= Entry(pantalla,textvariable=jugadores)
    entradapla=plantilla1.create_window(595,550,anchor="nw",window=entrada)

    # reproducir sonido
    archivo2 = "intrucciones.wav"
    # leemos el audio
    data2, fs2 = sf.read(archivo2)
    # REPRODUCIMOS SONIDO
    sd.play(data2, fs2)

    pantalla.mainloop()

pantalla_principal()