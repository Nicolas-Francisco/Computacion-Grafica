import scene_graph as sg
import basic_shapes as bs
import easy_shaders as es
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import transformations as tr
import sys


SIZE_IN_BYTES = 4

def applyTransform(transform, vertices):
    # Creating an array to store the transformed vertices
    # Since we will replace its content, the initial data is just garbage memory.
    transformedVertices = np.ndarray((len(vertices), 2), dtype=float)
    for i in range(len(vertices)):
        vertex2d = vertices[i]
        # input vertex only has x,y
        # expresing it in homogeneous coordinates
        homogeneusVertex = np.array([vertex2d[0], vertex2d[1], 0.0, 1.0])
        transformedVertex = np.matmul(transform, homogeneusVertex)

        # we are not prepared to handle 3d in this example
        # converting the vertex back to 2d
        transformedVertices[i,0] = transformedVertex[0] / transformedVertex[3]
        transformedVertices[i,1] = transformedVertex[1] / transformedVertex[3]
    return transformedVertices


class Ship:
    def __init__(self):                # Clase Ship, para nuestra nave y las enemigas
        self.Enemy = False             # Es enemigo o no?
        self.x = 0.0
        self.y = 0.0
        self.xray = 0.0
        self.yray = 0.0
        self.xray2 = 0.0
        self.yray2 = 0.0               # Hasta aquí solo son posiciones de la nave y sus rayos
        self.ray = False
        self.ray2 = False              # Está activado el rayo?
        self.life = 3
        self.Impacto = False           # Verificador de impacto
        self.Hitbox = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])     # Matriz hitbox
                               
        
    def Ray(self, dt):       # Rayo enemigo
        if self.ray:         # Si está activado...
            deltat = 0
            self.yray -= dt      # Baja el rayo en dt
            deltat += dt         # Guarda cuanto bajó
            if (self.Enemy and self.yray <= -1):    # Si ya cruzó el limite
                self.yray += deltat                 # Devuelvelo a la pos original
                self.ray = False                    # Rayo apagate
                deltat = 0
        else:               # Si no está activado...
            self.yray = self.y
            self.xray = self.x    # Muevete con la nave
            self.ray = True       # Dispara de nuevo
                
    def Rayv1(self, dt):    # Rayo 1 de nuestra nave
        if self.ray:        # Funciona igual que la del enemigo
            deltat = 0      # Pero mantiene apagado el rayo si está False
            self.yray += dt
            deltat += dt
            if (not self.Enemy and self.yray >= 1):
                self.yray -= deltat
                self.ray = False
                deltat = 0
        elif not self.ray:
            self.yray = self.y
            self.xray = self.x
    
    def Rayv2(self, dt):     # Segundo Rayo
        if self.ray2:
            deltat = 0
            self.yray2 += dt
            deltat += dt
            if (not self.Enemy and self.yray2 >= 1):
                self.yray2 -= deltat
                self.ray2 = False
                deltat = 0
        elif not self.ray2:
            self.yray2 = self.y
            self.xray2 = self.x

        
    def Movimiento(self, dt):    # Movimiento en loop de las naves enemigas
        if self.Enemy:           # Si es enemigo, ejecuta
            if (self.x >= -1.4 and self.x <= 0.8) and (self.y >= 0.78 and self.y <= 0.82): # A la derecha
                self.x += dt/2                                         # Muevete en dt/2
                transform=tr.matmul([tr.translate(+dt/2, 0, 0)]) 
                self.Hitbox = applyTransform(transform,  self.Hitbox)  # Mueve tu hitbox en dt/2
            if (self.x >= 0.78 and self.x <= 0.82) and (self.y >= 0.0 and self.y <= 0.82): # Abajo
                self.y -= dt/2
                transform=tr.matmul([tr.translate(0, -dt/2, 0)])
                self.Hitbox = applyTransform(transform,  self.Hitbox)
            if (self.x >= -0.8 and self.x <= 0.82) and (self.y >= -0.1 and self.y <= 0.1): # A la izquierda
                self.x -= dt/2
                transform=tr.matmul([tr.translate(-dt/2, 0, 0)])
                self.Hitbox = applyTransform(transform,  self.Hitbox)
            if (self.x <= -0.78 and self.x >= -0.82) and (self.y >= -0.1 and self.y <= 0.8): # Arriba
                self.y += dt/2
                transform=tr.matmul([tr.translate(0, +dt/2, 0)])
                self.Hitbox = applyTransform(transform,  self.Hitbox)
                
                
    def Colision(self, ListaEnemy):    # Función que detecta los impactos
        for enemigo in ListaEnemy:
            if not self.Impacto:       # Si no ha habido impacto
                if enemigo.xray > self.Hitbox[0][0] and enemigo.xray < self.Hitbox[1][0]:   # Entra el rayo 1
                    if enemigo.yray > self.Hitbox[0][1] and enemigo.yray < self.Hitbox[2][1]: 
                        self.Impacto = True   # Hay impacto y apaga el rayo que impactó
                        self.life -= 1        # Pierde vida
                        enemigo.ray = False
                elif (enemigo.xray2 > self.Hitbox[0][0] and enemigo.xray2 < self.Hitbox[1][0] and not enemigo.Enemy): 
                    if enemigo.yray2 > self.Hitbox[0][1] and enemigo.yray2 < self.Hitbox[2][1] and not enemigo.Enemy:
                        self.Impacto = True                          # Entra el rayo 2 y no es enemigo
                        self.life -= 1                               # Enemigos no tienen segundo disparo
                        enemigo.ray2 = False
            elif  self.Impacto:    # Si hubo impacto
                if enemigo.xray > self.Hitbox[0][0] and enemigo.xray < self.Hitbox[1][0]:    # Si sigue adentro
                    if enemigo.yray > self.Hitbox[0][1] and enemigo.yray < self.Hitbox[2][1]:
                        self.Impacto = False   # El impacto pasó y apagamos
                elif enemigo.xray2 > self.Hitbox[0][0] and enemigo.xray2 < self.Hitbox[1][0] and not enemigo.Enemy:
                    if enemigo.yray2 > self.Hitbox[0][1] and enemigo.yray2 < self.Hitbox[2][1] and not enemigo.Enemy:
                        self.Impacto = False
                        
                        
    def Restart(self):    # Función para las naves enemigas
        deltax = -self.x - 1.2     # Guardo su distancia para mover el hitbox
        deltay = -self.y + 0.8
        self.life = 1     # Restauro su vida
        self.x = -1.2     # Posicion inicial
        self.y = 0.8
        transform=tr.matmul([tr.translate(deltax, deltay,0)])   # Muevo el hitbox
        self.Hitbox = applyTransform(transform,  self.Hitbox)
            
            

# we will use the global controller as communication with the callback function

controller = Ship()   # Creo nuestra nave
controller.y = -0.6
transform=tr.matmul([tr.translate(0,-0.67,0),tr.scale(1/7,1/8,1)])
controller.Hitbox = applyTransform(transform,  controller.Hitbox)

# Creo nuestros enemigos
enemy1 = Ship()
enemy1.life = 1
enemy1.x = -1.2
enemy1.y = 0.8
enemy1.Enemy = True
transform=tr.matmul([tr.translate(-1.2, 0.89,0),tr.scale(1/7,1/8,1)])
enemy1.Hitbox = applyTransform(transform,  enemy1.Hitbox)
enemy1.xray2 = 1  # Saco el rayo fuera de escena
enemy1.yray2 = 1   

enemy2 = Ship()
enemy2.life = 1
enemy2.x = -1.2
enemy2.y = 0.8
enemy2.Enemy = True
transform=tr.matmul([tr.translate(-1.2, 0.89,0),tr.scale(1/7,1/8,1)])
enemy2.Hitbox = applyTransform(transform,  enemy2.Hitbox)
enemy2.xray2 = 1  # Saco el rayo fuera de escena
enemy2.yray2 = 1  

enemy3 = Ship()
enemy3.life = 1
enemy3.x = -1.2
enemy3.y = 0.8
enemy3.Enemy = True
transform=tr.matmul([tr.translate(-1.2, 0.89,0),tr.scale(1/7,1/8,1)])
enemy3.Hitbox = applyTransform(transform,  enemy3.Hitbox)
enemy3.xray2 = 1  # Saco el rayo fuera de escena
enemy3.yray2 = 1  

Enemy = [enemy1, enemy2, enemy3]   # Enemigos nuestros

def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        if not controller.ray:  # Activa el rayo 1 y el 2 dependiendo si están activados o no
            controller.ray = not controller.ray
        elif controller.ray and not controller.ray2:
            controller.ray2 = not controller.ray2

    elif key == glfw.KEY_A:
        if not controller.x<=-0.76: # Movimiento del controller limitado por la ventana
            controller.x -= 0.1
            transform=tr.matmul([tr.translate(-0.1, 0, 0)])
            controller.Hitbox = applyTransform(transform,  controller.Hitbox)

    elif key == glfw.KEY_D:
        if not controller.x>=0.76:
            controller.x += 0.1
            transform=tr.matmul([tr.translate(0.1, 0, 0)])
            controller.Hitbox = applyTransform(transform,  controller.Hitbox)

    elif key == glfw.KEY_W:
        if not controller.y>=0.6:
            controller.y += 0.1
            transform=tr.matmul([tr.translate(0, 0.1, 0)])
            controller.Hitbox = applyTransform(transform,  controller.Hitbox)

    elif key == glfw.KEY_S:
        if not controller.y<=-0.76:
            controller.y -= 0.1
            transform=tr.matmul([tr.translate(0, -0.1, 0)])
            controller.Hitbox = applyTransform(transform,  controller.Hitbox)

    elif key == glfw.KEY_ESCAPE:
        sys.exit()

    else:
        print('Unknown key')


# A simple class container to reference a shape on GPU memory

class GPUShape:
    vao = 0
    vbo = 0
    ebo = 0
    size = 0


def createQuad(c):

    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    
    vertexData = np.array([
    #   positions        colors
        -0.125, -0.125, 0.0,  c[0], c[1], c[2],
         0.125, -0.125, 0.0,  c[0], c[1], c[2],
         0.125,  0.125, 0.0,  c[0], c[1], c[2],
        -0.125,  0.125, 0.0,  c[0], c[1], c[2]
    # It is important to use 32 bits data
        ], dtype = np.float32)

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array(
        [0, 1, 2,
         2, 3, 0], dtype= np.uint32)

    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * SIZE_IN_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * SIZE_IN_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

def createTriangle():

    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining the location and colors of each vertex  of the shape
    vertexData = np.array(
    #     positions       colors
        [-0.1, -0.1, 0.0, 0.6, 0.6, 0.6,
          0.1, -0.1, 0.0, 0.6, 0.6, 0.6,
          0.0,  0.1, 0.0, 0.6, 0.6, 0.6],
          dtype = np.float32) # It is important to use 32 bits data

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array(
        [0, 1, 2], dtype= np.uint32)
        
    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * SIZE_IN_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * SIZE_IN_BYTES, indices, GL_STATIC_DRAW)


    return gpuShape

def Ship(pos=tr.identity(), tamaño=tr.identity()):    # Modelo de la nave principal con transformaciones
    transforms=[]
    transforms.append(tr.matmul([pos, tamaño, tr.translate(-1/11,-1.5/11,0), tr.scale(0.5,4,0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(1/11,-1.5/11,0), tr.scale(0.5,4,0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(0,-1/11,0), tr.scale(0.5,1.5,0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(0,3/11,0), tr.scale(0.5,0.5,0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(3.5/11,-3/11,0), tr.scale(1,1.5,0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(-3.5/11,-3/11,0), tr.scale(1,1.5,0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(-5/11,-3.5/11,0), tr.scale(0.5,1,0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(5/11,-3.5/11,0), tr.scale(0.5,1,0)]))
    
    
    
    transforms.append(tr.matmul([pos, tamaño, tr.translate(-2/11, -2.5/11,0), tr.scale(0.5,2,0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(2/11, -2.5/11,0), tr.scale(0.5,2,0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(0,-4/11,0), tr.scale(0.5,1.5,0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(0,1.5/11,0), tr.scale(0.5,1,0)]))
    
    
    
    transforms.append(tr.matmul([pos, tamaño, tr.translate(3/11, -0.5/11, 0), tr.scale(0.5,1,0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(-3/11, -0.5/11, 0), tr.scale(0.5,1,0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(0, 4.5/11, 0), tr.scale(0.5,1,0)]))
    
    
    return transforms

def Enemy(pos=tr.identity(), tamaño=tr.identity()):   # Modelo de las naves enemigas
    transforms=[]
    transforms.append(tr.matmul([pos, tamaño, tr.translate(0, 1.5/11, 0), tr.scale(0.5, 1, 0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(0, -3/11, 0), tr.scale(0.75, 1.5, 0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(1/11, -0.5/11, 0), tr.scale(0.5, 2, 0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(-1/11, -0.5/11, 0), tr.scale(0.5, 2, 0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(-2/11, 1.5/11, 0), tr.scale(0.5, 2, 0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(2/11, 1.5/11, 0), tr.scale(0.5, 2, 0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(-3/11, 1.5/11, 0), tr.scale(0.5, 1, 0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(3/11, 1.5/11, 0), tr.scale(0.5, 1, 0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(-4/11, 3/11, 0), tr.scale(0.5, 1.5, 0)]))
    transforms.append(tr.matmul([pos, tamaño, tr.translate(4/11, 3/11, 0), tr.scale(0.5, 1.5, 0)]))
    
    
    transforms.append(tr.matmul([pos, tamaño, tr.translate(0, -0.5/11, 0), tr.scale(0.5,1,0)]))
    
    return transforms

def Ray(pos=tr.identity(), tamaño=tr.identity()):    # Rayos
    transforms=[(tr.matmul([pos, tamaño, tr.translate(0, 0, 0), tr.scale(0.5,1,0)]))]
    return transforms


def createShip(color1, color2, color3):   # Función que recopila las transformaciones y crea el gpuShape
    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    V1=[-1/11, -1/11]
    V2=[1/11, -1/11]
    V3=[1/11, 1/11]
    V4=[-1/11, 1/11]
    original=np.array([V1, V2, V3, V4])
    transforms=Ship() 
    t1 = [transforms[0], transforms[1], transforms[2], transforms[3], transforms[4], transforms[5],
          transforms[6], transforms[7]]
    t2 = [transforms[8], transforms[9], transforms[10], transforms[11]]
    t3 = [transforms[12], transforms[13], transforms[14]]
    Vertex=[]
    for transform in t1:   # Transformo el cuadrado dependiendo del color que quiera
        V=applyTransform(transform, original)
        for i in range(len(V)):
            for j in range(0,2):
                Vertex.append(V[i,j])
            Vertex.append(0)
            Vertex.append(color1[0])
            Vertex.append(color1[1])
            Vertex.append(color1[2])
    for transform in t2:
        V=applyTransform(transform, original)
        for i in range(len(V)):
            for j in range(0,2):
                Vertex.append(V[i,j])
            Vertex.append(0)
            Vertex.append(color2[0])
            Vertex.append(color2[1])
            Vertex.append(color2[2])
    for transform in t3:
        V=applyTransform(transform, original)
        for i in range(len(V)):
            for j in range(0,2):
                Vertex.append(V[i,j])
            Vertex.append(0)
            Vertex.append(color3[0])
            Vertex.append(color3[1])
            Vertex.append(color3[2])

        
    vertexData = np.array(np.array(Vertex), dtype=np.float32)

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array(
        [0, 1, 2,
         2, 3, 0,
         4, 5, 6,
         6, 7, 4,
         8, 9, 10,
         10,11,8,
         12 ,13,14,
         14, 15,12,
         16, 17,18,
         18, 19,16,
         20, 21,22,
         22, 23, 20,
         24, 25, 26, 
         26, 27, 24,
         28, 29, 30,
         30, 31, 28,
         32, 33, 34,
         34, 35, 32,
         36, 37, 38, #10
         38, 39, 36,
         40, 41, 42,
         42, 43, 40,
         44, 45, 46,
         46, 47, 44,
         48, 49, 50,
         50, 51, 48, # 13
         52, 53, 54,
         54, 55, 52,
         56, 57, 58,
         58, 59, 56], dtype=np.uint32)

    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * SIZE_IN_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * SIZE_IN_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

def createEnemy(color1, color2):   # Lo mismo que lo anterior, pero con el modelo del enemigo
    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    V1=[-1/11, -1/11]
    V2=[1/11, -1/11]
    V3=[1/11, 1/11]
    V4=[-1/11, 1/11]
    original=np.array([V1, V2, V3, V4])
    transforms=Enemy() # traigo las transformaciones necesarias
    t1 = [transforms[0], transforms[1], transforms[2], transforms[3], transforms[4], transforms[5],
          transforms[6], transforms[7], transforms[8], transforms[9]]
    t2 = [transforms[10]]
    Vertex=[]
    for transform in t1:
        V=applyTransform(transform, original)
        for i in range(len(V)):
            for j in range(0,2):
                Vertex.append(V[i,j])
            Vertex.append(0)
            Vertex.append(color1[0])
            Vertex.append(color1[1])
            Vertex.append(color1[2])
    for transform in t2:
        V=applyTransform(transform, original)
        for i in range(len(V)):
            for j in range(0,2):
                Vertex.append(V[i,j])
            Vertex.append(0)
            Vertex.append(color2[0])
            Vertex.append(color2[1])
            Vertex.append(color2[2])
        
    vertexData = np.array(np.array(Vertex), dtype=np.float32)

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array(
        [0, 1, 2,
         2, 3, 0,
         4, 5, 6,
         6, 7, 4,
         8, 9, 10,
         10,11,8,
         12 ,13,14,
         14, 15,12,
         16, 17,18,
         18, 19,16,
         20, 21,22,
         22, 23, 20,
         24, 25, 26, 
         26, 27, 24,
         28, 29, 30,
         30, 31, 28,
         32, 33, 34,
         34, 35, 32,
         36, 37, 38,
         38, 39, 36,
         40, 41, 42,
         42, 43, 40], dtype=np.uint32)

    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * SIZE_IN_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * SIZE_IN_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

def createRay(color):   # Lo mismo, pero con el rayo
    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    V1=[-1/11, -1/11]
    V2=[1/11, -1/11]
    V3=[1/11, 1/11]
    V4=[-1/11, 1/11]
    original=np.array([V1, V2, V3, V4])
    transforms=Ray()
    Vertex=[]
    for transform in transforms:
        V=applyTransform(transform, original)
        for i in range(len(V)):
            for j in range(0,2):
                Vertex.append(V[i,j])
            Vertex.append(0)
            Vertex.append(color[0])
            Vertex.append(color[1])
            Vertex.append(color[2])
    
    vertexData = np.array(np.array(Vertex), dtype=np.float32)

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array(
        [0, 1, 2,
         2, 3, 0], dtype=np.uint32)

    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * SIZE_IN_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * SIZE_IN_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

def drawShape(shaderProgram, shape, transform):

    # Binding the proper buffers
    glBindVertexArray(shape.vao)
    glBindBuffer(GL_ARRAY_BUFFER, shape.vbo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.ebo)

    # updating the new transform attribute
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "transform"), 1, GL_TRUE, transform)

    # Describing how the data is stored in the VBO
    position = glGetAttribLocation(shaderProgram, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)
    
    color = glGetAttribLocation(shaderProgram, "color")
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)

    # This line tells the active shader program to render the active element buffer with the given size
    glDrawElements(GL_TRIANGLES, shape.size, GL_UNSIGNED_INT, None)



if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Space Wars", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)
    vertex_shader = """
    #version 130
    in vec3 position;
    in vec3 color;

    out vec3 fragColor;

    uniform mat4 transform;

    void main()
    {
        fragColor = color;
        gl_Position = transform * vec4(position, 1.0f);
    }
    """

    fragment_shader = """
    #version 130

    in vec3 fragColor;
    out vec4 outColor;

    void main()
    {
        outColor = vec4(fragColor, 1.0f);
    }
    """

    # Assembling the shader program (pipeline) with both shaders
    shaderProgram = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))
    
    pipelineTexture = es.SimpleTextureTransformShaderProgram()
    
    # Telling OpenGL to use our shader program

    
    # Setting up the clear screen color
    glClearColor(0.0, 0.0, 0.0, 1.0)


    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Creating shapes on GPU memory
    gpuQuadWhite = createQuad([1,1,1])
    gpuTriangle = createTriangle()
    gpuRay = createRay([0.6, 1, 1])
    gpuRayE = createRay([1, 0.1, 0])
    gpuShip = createShip([1, 1, 1], [0.6, 0.6, 0.6], [1, 0, 0])
    gpuEnemy = createEnemy([0.6, 0.6, 0.6], [1, 1, 1])
    
    
    # Creo las texturas para el fonfo y animaciones
    gpuSpace = es.toGPUShape(bs.createTextureQuad("Space.jpg"), GL_REPEAT, GL_LINEAR)
    
    gpuDied = es.toGPUShape(bs.createTextureQuad("Over.jpg"), GL_REPEAT, GL_LINEAR)
    
    gpuVictory = es.toGPUShape(bs.createTextureQuad("Victory.png"), GL_REPEAT, GL_LINEAR)
    
    t0 = glfw.get_time()
    tiempo = 0
    
    N = sys.argv[1] # N recibido del usuario, en este ejemplo, quiero 10 naves
    try:
        n = N - 3    # N que utilizaremos por las naves a reiniciar
    except ValueError:
        print("Dato ingresado debe ser tipo int")

    while not glfw.window_should_close(window):
        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1
        
        tiempo += dt   # Defino mi tiempo para mover el fondo

        # Using GLFW to check for input events
        glfw.poll_events()

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Trabajamos con texturas
        glUseProgram(pipelineTexture.shaderProgram)
        
        
        # Creo las primeras dos imagenes
        transform = tr.matmul([tr.translate(0, -tiempo/4, 0), tr.scale(2, 2, 0)])
        glUniformMatrix4fv(glGetUniformLocation(pipelineTexture.shaderProgram, "transform"), 1, GL_TRUE, transform)
        pipelineTexture.drawShape(gpuSpace)
  
       
        transform = tr.matmul([tr.translate(0, 2-tiempo/4, 0),tr.scale(1,-1,1), tr.scale(2, 2, 0)])
        glUniformMatrix4fv(glGetUniformLocation(pipelineTexture.shaderProgram, "transform"), 1, GL_TRUE, transform)
        pipelineTexture.drawShape(gpuSpace)

        # Si ya avancé en una figura
        if tiempo/4 > 1:
            transform = tr.matmul([tr.translate(0, 4-tiempo/4, 0), tr.scale(2, 2, 0)])
            glUniformMatrix4fv(glGetUniformLocation(pipelineTexture.shaderProgram, "transform"), 1, GL_TRUE, transform)
            pipelineTexture.drawShape(gpuSpace)
        
            transform = tr.matmul([tr.translate(0, 6-tiempo/4, 0), tr.scale(1, -1, 1), tr.scale(2, 2, 0)])
            glUniformMatrix4fv(glGetUniformLocation(pipelineTexture.shaderProgram, "transform"), 1, GL_TRUE, transform)
            pipelineTexture.drawShape(gpuSpace)
            
            transform = tr.matmul([tr.translate(0, 8-tiempo/4, 0), tr.scale(2, 2, 0)])
            glUniformMatrix4fv(glGetUniformLocation(pipelineTexture.shaderProgram, "transform"), 1, GL_TRUE, transform)
            pipelineTexture.drawShape(gpuSpace)
            
            # Relleno el fondo negro con las mismas tranformaciones
        
        if tiempo/4 >= 8:  # Si ya llegó al límite, reinicia
            tiempo = 0
            
            
        if controller.life == 0:   # Si muero, animación de Game Over
            transform = tr.matmul([tr.translate(0, 0, 0), tr.scale(0.8, 0.6, 0)])
            glUniformMatrix4fv(glGetUniformLocation(pipelineTexture.shaderProgram, "transform"), 1, GL_TRUE, transform)
            pipelineTexture.drawShape(gpuDied)
            
            
        if (((enemy1.life == 0) and (enemy2.life == 0) and (enemy3.life == 0) and n == 0 and (not controller.life == 0)) or
            ((enemy1.life == 0) and (enemy2.life == 0) and n <= 0 and (not controller.life == 0) and N == 2) or
            ((enemy1.life == 0) and n <= 0 and (not controller.life == 0) and N == 1)): 
            # Si todos se mueren Ó mueren los dos primeros y pedí solo dos Ó se muere el primero y pedí uno
            # Animación de victoria
            transform = tr.matmul([tr.translate(0.1, 0, 0), tr.scale(2, 0.4, 0)])
            glUniformMatrix4fv(glGetUniformLocation(pipelineTexture.shaderProgram, "transform"), 1, GL_TRUE, transform)
            pipelineTexture.drawShape(gpuVictory)
            transform = tr.matmul([tr.translate(-1.9, 0, 0), tr.scale(2, 0.4, 0)])
            glUniformMatrix4fv(glGetUniformLocation(pipelineTexture.shaderProgram, "transform"), 1, GL_TRUE, transform)
            pipelineTexture.drawShape(gpuVictory)
            
        
        glUseProgram(shaderProgram)      # Ahora usamos los shaders y no texturas
        
        if controller.life > 0:      # Mientras la nave siga viva
            controller.Rayv1(dt)
            controller.Rayv2(dt)    
            controller.Colision([enemy1, enemy2, enemy3])   # Activo funciones de rayos y Colision
            transform1 = tr.matmul([tr.translate(controller.xray, controller.yray, 0), tr.scale(1/3, 1/3, 0)])
            drawShape(shaderProgram, gpuRay, transform1)
            transform1 = tr.matmul([tr.translate(controller.xray2, controller.yray2, 0), tr.scale(1/3, 1/3, 0)])
            drawShape(shaderProgram, gpuRay, transform1)
            transform2 = tr.matmul([tr.translate(controller.x, controller.y, 0), tr.scale(1/3, 1/3, 0)])
            drawShape(shaderProgram, gpuShip, transform2)  # Dibujo mis dos rayos y nave
            
                
########################################## Enemigos            
            
        if enemy1.life > 0:    # Mientras siga vivo, hago lo mismo que antes
            enemy1.Ray(dt)
            enemy1.Movimiento(dt)
            transform=tr.matmul([tr.translate(enemy1.xray, enemy1.yray, 0), tr.scale(1/3, 1/3, 0)])
            drawShape(shaderProgram, gpuRayE, transform)
            transform2=tr.matmul([tr.translate(enemy1.x, enemy1.y, 0), tr.scale(1/3, 1/3, 0)])
            drawShape(shaderProgram, gpuEnemy, transform2)
            enemy1.Colision([controller])
            
            
        if (t1 > 3.5 and enemy2.life > 0) and N >= 2:   # Espero unos segundos a que avance la primera nave
            enemy2.Ray(dt)
            enemy2.Movimiento(dt)
            transform=tr.matmul([tr.translate(enemy2.xray, enemy2.yray, 0), tr.scale(1/3, 1/3, 0)])
            drawShape(shaderProgram, gpuRayE, transform)
            transform2=tr.matmul([tr.translate(enemy2.x, enemy2.y, 0), tr.scale(1/3, 1/3, 0)])
            drawShape(shaderProgram, gpuEnemy, transform2)
            enemy2.Colision([controller])
            
        if (t1 > 7 and enemy3.life > 0) and N >= 3:   # Espero a que salgan las primeras naves
            enemy3.Ray(dt)
            enemy3.Movimiento(dt)
            transform=tr.matmul([tr.translate(enemy3.xray, enemy3.yray, 0), tr.scale(1/3, 1/3, 0)])
            drawShape(shaderProgram, gpuRayE, transform)
            transform2=tr.matmul([tr.translate(enemy3.x, enemy3.y, 0), tr.scale(1/3, 1/3, 0)])
            drawShape(shaderProgram, gpuEnemy, transform2)
            enemy3.Colision([controller])
            
        if enemy1.life == 0 and n > 0:   # Si el enemigo se muere y quedan naves que revivir
            enemy1.Restart()             # La revivo
            n -= 1                       # Bajo el contador
            
        if enemy2.life == 0 and n > 0:
            enemy2.Restart()
            n -= 1
            
        if enemy3.life == 0 and n > 0:
            enemy3.Restart()
            n -= 1

        glfw.swap_buffers(window)

    glfw.terminate()
