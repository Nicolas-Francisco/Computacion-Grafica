#!/usr/bin/env python
# coding: utf-8

# In[80]:


import glfw
import math
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import scene_graph as sg 
import lighting_shaders as l
import local_shapes2 as ls
import triangle_mesh as tm

PROJECTION_ORTHOGRAPHIC = 0
PROJECTION_FRUSTUM = 1
PROJECTION_PERSPECTIVE = 2

class Controller:
    def __init__(self):
        self.fillPolygon = True
    
# We will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)


def generateT(t):
    return np.array([[1, t, t**2, t**3]]).T


def hermiteMatrix(P1, P2, T1, T2):
    
    # Generate a matrix concatenating the columns
    G = np.concatenate((P1, P2, T1, T2), axis=1)
    
    # Hermite base matrix is a constant
    Mh = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])    
    
    return np.matmul(G, Mh)

# M is the cubic curve matrix, N is the number of samples between 0 and 1
def evalCurve(M, N):
    # The parameter t should move between 0 and 1
    ts = np.linspace(0.0, 1.0, N)
    
    # The computed value in R3 for each sample will be stored here
    curve = np.ndarray(shape=(N, 3), dtype=float)
    
    for i in range(len(ts)):
        T = generateT(ts[i])
        curve[i, 0:3] = np.matmul(M, T).T
        
    return curve

def dist(P1,P2):
    dist=np.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2+(P1[2]-P2[2])**2)
    return dist

############################################# Vehículo

def createCar(color1, color2):
    
    gpuBlackCube = es.toGPUShape(bs.createColorNormalsCube(0,0,0))
    gpuCube1 = es.toGPUShape(bs.createColorNormalsCube(color1[0], color1[1], color1[2]))
    gpuCube2 = es.toGPUShape(bs.createColorNormalsCube(color2[0], color2[1], color2[2]))
    gpuCylinder = es.toGPUShape(ls.generateTextureNormalsCylinder(15, "neumatico.jpeg" , 3, 2, 1),GL_REPEAT,GL_NEAREST)
    gpuRueda = es.toGPUShape(bs.createTextureNormalsCube("rueda.jpeg"),GL_REPEAT,GL_NEAREST)
    gpuLogo1 = es.toGPUShape(bs.createTextureNormalsCube("logo1.jpg"),GL_REPEAT,GL_NEAREST)
    gpuLogo2 = es.toGPUShape(bs.createTextureNormalsCube("logo2.png"),GL_REPEAT,GL_NEAREST)
    
    
    # Cheating a single wheel
    wheel = sg.SceneGraphNode("wheel")
    wheel.transform = tr.matmul([tr.scale(0.04, 0.04, 0.1)])
    wheel.childs += [gpuCylinder]
    
    rueda1 = sg.SceneGraphNode("rueda1")
    rueda1.transform = tr.matmul([tr.scale(0.16, 0.16, 0.01), tr.translate(0, 0, 20)])
    rueda1.childs += [gpuRueda]

    wheelRotation = sg.SceneGraphNode("wheelRotation")
    wheelRotation.childs += [wheel]
    wheelRotation.childs += [rueda1]

    ############################## Ruedas
    
    ###### Cilindros
    
    frontRightWheel = sg.SceneGraphNode("frontRightWheel")
    frontRightWheel.transform = tr.matmul([tr.translate(0.36, 0.18, -0.25), tr.rotationX(-math.pi/2)])
    frontRightWheel.childs += [wheelRotation]
    
    frontLeftWheel = sg.SceneGraphNode("frontLeftWheel")
    frontLeftWheel.transform = tr.matmul([tr.translate(0.36, -0.18, -0.25), tr.rotationX(math.pi/2)])
    frontLeftWheel.childs += [wheelRotation]

    backRightWheel = sg.SceneGraphNode("backRightWheel")
    backRightWheel.transform = tr.matmul([tr.translate(-0.36, 0.18, -0.25), tr.rotationX(-math.pi/2)])
    backRightWheel.childs += [wheelRotation]
    
    backLeftWheel = sg.SceneGraphNode("backLeftWheel")
    backLeftWheel.transform = tr.matmul([tr.translate(-0.36, -0.18, -0.25), tr.rotationX(math.pi/2)])
    backLeftWheel.childs += [wheelRotation]
    
    
    ############################### Base
    
    Cube1 = sg.SceneGraphNode("chasis1")
    Cube1.transform = tr.matmul([tr.translate(0, 0, -0.25), tr.scale(1.2, 0.5, 0.1)])
    Cube1.childs += [gpuCube1]
    
    Cube2 = sg.SceneGraphNode("chasis2")
    Cube2.transform = tr.matmul([tr.translate(0.69, -0.29, -0.25), tr.scale(0.1, 0.1, 0.1)])
    Cube2.childs += [gpuCube2]
    
    Cube3 = sg.SceneGraphNode("chasis3")
    Cube3.transform = tr.matmul([tr.translate(0.69, 0.29, -0.25), tr.scale(0.1, 0.1, 0.1)])
    Cube3.childs += [gpuCube2]
    
    Cube8 = sg.SceneGraphNode("chasis8")
    Cube8.transform = tr.matmul([tr.translate(0.69, 0, -0.25), tr.scale(0.1, 0.5, 0.1)])
    Cube8.childs += [gpuCube1]
    
    Cube4 = sg.SceneGraphNode("chasis4")
    Cube4.transform = tr.matmul([tr.translate(0.6, 0, -0.25), tr.scale(0.1, 0.7, 0.1)])
    Cube4.childs += [gpuCube1]
    
    Cube5 = sg.SceneGraphNode("chasis5")
    Cube5.transform = tr.matmul([tr.translate(-0.69, -0.29, -0.25), tr.scale(0.1, 0.1, 0.1)])
    Cube5.childs += [gpuCube2]
    
    Cube6 = sg.SceneGraphNode("chasis6")
    Cube6.transform = tr.matmul([tr.translate(-0.69, 0.29, -0.25), tr.scale(0.1, 0.1, 0.1)])
    Cube6.childs += [gpuCube2]
    
    Cube7 = sg.SceneGraphNode("chasis7")
    Cube7.transform = tr.matmul([tr.translate(-0.6, 0, -0.25), tr.scale(0.1, 0.7, 0.1)])
    Cube7.childs += [gpuCube1]
    
    Cube9 = sg.SceneGraphNode("chasis9")
    Cube9.transform = tr.matmul([tr.translate(-0.69, 0, -0.25), tr.scale(0.1, 0.5, 0.1)])
    Cube9.childs += [gpuBlackCube]
    
    Cube10 = sg.SceneGraphNode("chasis10")
    Cube10.transform = tr.matmul([tr.translate(0, 0, -0.25), tr.scale(0.4, 0.7, 0.1)])
    Cube10.childs += [gpuCube1]
    
    ##################################### Parte superior
    
    ################## Asiento
    Cube11 = sg.SceneGraphNode("chasis11")
    Cube11.transform = tr.matmul([tr.translate(-0.1, 0, -0.175), tr.scale(0.3, 0.2, 0.05)])
    Cube11.childs += [gpuBlackCube]
    
    Cube12 = sg.SceneGraphNode("chasis12")
    Cube12.transform = tr.matmul([tr.translate(-0.255, 0, -0.125), tr.scale(0.05, 0.2, 0.15)])
    Cube12.childs += [gpuBlackCube]
    
    ############## Resto del cuerpo
    
    Cube13 = sg.SceneGraphNode("chasis12")
    Cube13.transform = tr.matmul([tr.translate(-0.1, -0.15, -0.15), tr.scale(0.4, 0.05, 0.1)])
    Cube13.childs += [gpuCube1]
    
    Cube14 = sg.SceneGraphNode("chasis12")
    Cube14.transform = tr.matmul([tr.translate(-0.1, 0.15, -0.15), tr.scale(0.4, 0.05, 0.1)])
    Cube14.childs += [gpuCube1]
    
    Cube15 = sg.SceneGraphNode("chasis12")
    Cube15.transform = tr.matmul([tr.translate(-0.325, 0, -0.125), tr.scale(0.1, 0.4, 0.15)])
    Cube15.childs += [gpuCube1]
    
    Cube16 = sg.SceneGraphNode("chasis12")
    Cube16.transform = tr.matmul([tr.translate(0.3, 0, -0.175), tr.scale(0.5, 0.4, 0.05)])
    Cube16.childs += [gpuCube1]
    
    ################# Fondo
    
    Cube17 = sg.SceneGraphNode("chasis12")
    Cube17.transform = tr.matmul([tr.translate(-0.5, -0.19, -0.125), tr.scale(0.02, 0.02, 0.15)])
    Cube17.childs += [gpuBlackCube]
    
    Cube18 = sg.SceneGraphNode("chasis12")
    Cube18.transform = tr.matmul([tr.translate(-0.5, 0.19, -0.125), tr.scale(0.02, 0.02, 0.15)])
    Cube18.childs += [gpuBlackCube]
    
    Cube19 = sg.SceneGraphNode("chasis12")
    Cube19.transform = tr.matmul([tr.translate(-0.5, 0, -0.05), tr.scale(0.1, 0.5, 0.005)])
    Cube19.childs += [gpuBlackCube]
    
    ##################### Stickers
    
    deco1 = sg.SceneGraphNode("deco1")
    deco1.transform = tr.matmul([tr.scale(0.01, 0.1, 0.1), tr.translate(-75, 0, -2.4)])
    deco1.childs += [gpuLogo1]
    
    deco2 = sg.SceneGraphNode("deco2")
    deco2.transform = tr.matmul([tr.scale(0.34, 0.34, 0.01), tr.rotationZ(math.pi/2), tr.translate(0, -1.2, -15)])
    deco2.childs += [gpuLogo2]

    ########################################## Arbol
    
    cuerpo = sg.SceneGraphNode("cuerpo")
    cuerpo.childs += [Cube1]
    cuerpo.childs += [Cube2]
    cuerpo.childs += [Cube3]
    cuerpo.childs += [Cube4]
    cuerpo.childs += [Cube5]
    cuerpo.childs += [Cube6]
    cuerpo.childs += [Cube7]
    cuerpo.childs += [Cube8]
    cuerpo.childs += [Cube9]
    cuerpo.childs += [Cube10]
    cuerpo.childs += [Cube11]
    cuerpo.childs += [Cube12]
    cuerpo.childs += [Cube13]
    cuerpo.childs += [Cube14]
    cuerpo.childs += [Cube15]
    cuerpo.childs += [Cube16]
    cuerpo.childs += [Cube17]
    cuerpo.childs += [Cube18]
    cuerpo.childs += [Cube19]
    

    ruedas = sg.SceneGraphNode("ruedas")
    ruedas.childs += [frontRightWheel]
    ruedas.childs += [frontLeftWheel]
    ruedas.childs += [backRightWheel]
    ruedas.childs += [backLeftWheel]
    
    
    deco = sg.SceneGraphNode("deco")
    deco.childs += [deco1]
    deco.childs += [deco2]
     

    car = sg.SceneGraphNode("car")
    car.childs+=[ruedas]
    car.childs+=[cuerpo]
    car.childs+=[deco]

    return car




################################### Mallas de la Pista

def generatePistaColor(P1,P2,P3,P4,T1,T2,N,indice):
    GMh = hermiteMatrix(P1,P2, T1, T2)
    curve1 = evalCurve(GMh, N)
    GMh2 = hermiteMatrix(P3, P4, T1, T2)
    curve2 = evalCurve(GMh2, N)
    vertices = []
    indices = []
    start_index = indice
    vertices += [curve2[0][0],curve2[0][1],curve2[0][2],0,0,0,
    curve1[0][0],curve1[0][1],curve1[0][2],0,0,0]
 
    
    for i in range(curve1.shape[0]-1):
        # d === c
        # |     |
        # |     |
        # a === b
        cord=curve1[i]
        cord2=curve1[i+1]
        cord3=curve2[i]
        cord4=curve2[i+1]
        a=np.array([ cord[0],   cord[1],           cord[2]])
        b=np.array([ cord3[0],   cord3[1],         cord3[2]])
        d=np.array([ cord2[0],  cord2[1],          cord2[2]])
        c=np.array([ cord4[0],  cord4[1]         ,  cord4[2]])
        _vertex, _indices = ls.createColorQuadIndexation(start_index, a, b, c, d,[0,0,0])
        indices  += _indices
        start_index += 2
        vertices += _vertex[12:len(_vertex) ]
    #print("color")
    #print(vertices)
    #print(indices)

    return (bs.Shape(vertices,indices),start_index)

def createTextureNormalsQuadIndexationFirst(start_index, a, b, c, d):

    # Computing normal from a b c
    v1 = np.array(a-b)
    v2 = np.array(b-c)
    v1xv2 = np.cross(v1, v2)

    # Defining locations and colors for each vertex of the shape    
    vertices = [
    #        positions               colors                 normals
        a[0], a[1], a[2],    1,1,     v1xv2[0], v1xv2[1], v1xv2[2],
        b[0], b[1], b[2],    0,1,     v1xv2[0], v1xv2[1], v1xv2[2],
        c[0], c[1], c[2],    1,1,     v1xv2[0], v1xv2[1], v1xv2[2],
        d[0], d[1], d[2],    0,1,     v1xv2[0], v1xv2[1], v1xv2[2]
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         start_index+1, start_index, start_index+2,
         start_index+2, start_index+3, start_index+1
        ]
    
    return (vertices, indices)

def createTextureNormalsQuadIndexationSecond(start_index, a, b, c, d):

    # Computing normal from a b c
    v1 = np.array(a-b)
    v2 = np.array(b-c)
    v1xv2 = np.cross(v1, v2)

    # Defining locations and colors for each vertex of the shape    
    vertices = [
    #        positions               colors                 normals
        a[0], a[1], a[2],    1,0,     v1xv2[0], v1xv2[1], v1xv2[2],
        b[0], b[1], b[2],    0,0,     v1xv2[0], v1xv2[1], v1xv2[2],
        c[0], c[1], c[2],    1,0,     v1xv2[0], v1xv2[1], v1xv2[2],
        d[0], d[1], d[2],    0,0,     v1xv2[0], v1xv2[1], v1xv2[2]
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         start_index+1, start_index, start_index+2,
         start_index+2, start_index+3, start_index+1
        ]
    
    return (vertices, indices)


####################### Pista Con Textura

def generatePistaTexture(P1,P2,P3,P4,T1,T2,T3,T4,N,indice):
    
    GMh = hermiteMatrix(P1,P2, T1, T2)
    curve1 = evalCurve(GMh, N)
    GMh2 = hermiteMatrix(P3, P4, T3, T4)
    curve2 = evalCurve(GMh2, N)
    vertices = []
    indices = []
    start_index = indice

    a=curve1[0]
    b=curve2[0]
    c=curve2[1]
    v1 = np.array(a-b)
    v2 = np.array(b-c)
    v1xv2 = np.cross(v1, v2)
    vertices += [curve2[0][0],curve2[0][1],curve2[0][2],1,1,v1xv2[0],v1xv2[1],v1xv2[2],
    curve1[0][0],curve1[0][1],curve1[0][2],0,1,v1xv2[0],v1xv2[1],v1xv2[2] ]


    first=False
    # We generate a rectangle for every latitude, 
    for i in range(curve1.shape[0]-1):
        # d === c
        # |     |
        # |     |
        # a === b
        cord=curve1[i]
        cord2=curve1[i+1]
        cord3=curve2[i]
        cord4=curve2[i+1]
        a=np.array([ cord[0],   cord[1],           cord[2]])
        b=np.array([ cord3[0],   cord3[1],         cord3[2]])
        d=np.array([ cord2[0],  cord2[1],          cord2[2]])
        c=np.array([ cord4[0],  cord4[1]         ,  cord4[2]])
        if first==True:
            _vertex, _indices = createTextureNormalsQuadIndexationFirst(start_index, a, b, c, d)
            first=False
        else:
            _vertex, _indices = createTextureNormalsQuadIndexationSecond(start_index, a, b, c, d)
            first=True
        indices  += _indices
        start_index += 2
        vertices += _vertex[16:len(_vertex) ]
    #print("texture")
    #print(vertices)
    #print(indices)

    return (bs.Shape(vertices,indices),start_index)


def create_pista_mesh_Color(pista):

    ## Creamos los vertices
    mesh_vertices = []

    for i in range(0, len(pista.vertices) - 1, 6):
        mesh_vertices.append((pista.vertices[i], pista.vertices[i + 1], pista.vertices[i + 2]))

    ## Creamos los triangulos
    mesh_triangles = []

    for i in range(0, len(pista.indices) - 1, 3):
        mesh_triangles.append(
            tm.Triangle(pista.indices[i], pista.indices[i + 1], pista.indices[i + 2]))

    ## Creamos la malla con un meshBuilder
    mesh_builder = tm.TriangleFaceMeshBuilder()

    for triangle in mesh_triangles:
        mesh_builder.addTriangle(triangle)

    return mesh_builder, mesh_triangles, mesh_vertices

def create_pista_mesh_Texture(pista):

    ## Creamos los vertices
    mesh_vertices = []

    for i in range(0, len(pista.vertices) - 1, 8):
        mesh_vertices.append((pista.vertices[i], pista.vertices[i + 1], pista.vertices[i + 2]))

    ## Creamos los triangulos
    mesh_triangles = []

    for i in range(0, len(pista.indices) - 1, 3):
        mesh_triangles.append(
            tm.Triangle(pista.indices[i], pista.indices[i + 1], pista.indices[i + 2]))

    ## Creamos la malla con un meshBuilder
    mesh_builder = tm.TriangleFaceMeshBuilder()

    for triangle in mesh_triangles:
        mesh_builder.addTriangle(triangle)

    return mesh_builder, mesh_triangles, mesh_vertices


def draw_mesh_Color(mesh, vertices):
    shape_indices = []
    shape_vertices = []
    # Creamos la lista con indices
    for triangle_mesh in mesh.getTriangleFaceMeshes():
        triangle = triangle_mesh.data
        shape_indices += [triangle.a, triangle.b, triangle.c]
    # Creamos la lista de vertices
    for vertice in vertices:
        shape_vertices += [vertice[0], vertice[1], vertice[2], 0,0,0]

    return bs.Shape(shape_vertices, shape_indices)

def draw_mesh_Texture(mesh, vertices,imagen):
    shape_indices = []
    # Creamos la lista con indices
    for triangle_mesh in mesh.getTriangleFaceMeshes():
        triangle = triangle_mesh.data
        shape_indices += [triangle.a, triangle.b, triangle.c]
    # Creamos la lista de vertices

    return bs.Shape(vertices, shape_indices,imagen)


#################################### Puntos para la Pista

###### Primeros puntos
P1 = np.array([-1,0, 0])
P2 = np.array([-2,10, 0])

T1 = np.array([[0,10, 0]]).T
T2 = np.array([[-4,10, 0]]).T 

P3 = np.array([1,0, 0])
P4 = np.array([0,10, 0])

###### Resto de puntos
P5 = np.array([-4, 11, 0])
P6 = np.array([-4, 13, 0])
T5 = np.array([[-10,0,0]]).T 

P7 = np.array([-7, 11, 0])
P8 = np.array([-7, 13, 0])
T7 = np.array([[0,10,0]]).T

P9 = np.array([-10, 14, 0])
P10 = np.array([-8, 14, 0])
T9 = np.array([[10,0,0]]).T 

P11 = np.array([-7, 17, 1])
P12 = np.array([-7, 15, 1])
T11 = np.array([[0,10,0]]).T 

P13 = np.array([-4, 17, 1])
P14 = np.array([-4, 15, 1])
T13 = np.array([[10,0,0]]).T 

P15 = np.array([-3, 18, 2])
P16 = np.array([-1, 18, 2])
T15 = np.array([[0,-10,0]]).T 

P17 = np.array([0, 23, 3])
P18 = np.array([0, 21, 3])
T17 = np.array([[10,0,0]]).T 

P19 = np.array([3, 19, 4])
P20 = np.array([1, 19, 4])
T19 = np.array([[0,-10,0]]).T 

### Puntos rectificadores de la pista

P39 = np.array([4, 18, 4])
P40 = np.array([4, 16, 4])

P41 = np.array([7, 18, 4])
P42 = np.array([7, 16, 4])

P43 = np.array([10, 15, 4])
P44 = np.array([8, 15, 4])

P45 = np.array([10, 12, 4])
P46 = np.array([8, 12, 4])

####### Elicoide

P21 = np.array([10, 10, 5])
P22 = np.array([8, 10, 5])
T21 = np.array([[0,-10,0]]).T 

P23 = np.array([6, 6, 4])
P24 = np.array([6, 8, 4])
T23 = np.array([[-10,0,0]]).T 

P25 = np.array([2, 10, 3])
P26 = np.array([4, 10, 3])
T25 = np.array([[0,10,0]]).T 

P27 = np.array([6, 14, 2])
P28 = np.array([6, 12, 2])
T27 = np.array([[10,0,0]]).T 

P29 = np.array([10, 10, 1])
P30 = np.array([8, 10, 1])
T29 = np.array([[0,-10,0]]).T 

#### Termina la elicoide

P31 = np.array([10, 8, 0])
P32 = np.array([8, 8, 0])
T31 = np.array([[0,-10,0]]).T 

P33 = np.array([6, 4, 1])
P34 = np.array([6, 6, 1])
T33 = np.array([[-10,0,0]]).T 

P35 = np.array([5, 3, 2])
P36 = np.array([3, 3, 2])
T35 = np.array([[10,0,0]]).T 

P47 = np.array([5, 0, 2])
P48 = np.array([3, 0, 2])

P49 = np.array([2, -4, 2])
P50 = np.array([2, -2, 2])

### Unión final con los primeros puntos

P37 = np.array([-1, -0.1, 0])
P38 = np.array([1, -0.1, 0])
T37 = np.array([[10,0,0]]).T 

########################## Ordenamos los puntos por izquierda y derecha

PI=np.array([P1, P2, P5, P7, P9, P11, P13, P15, P17, P19, P39, P41, P43, P45, P21, P23, P25, P27,
            P29, P31, P33, P35, P47, P49, P37])

PD=np.array([P3, P4, P6, P8, P10, P12, P14, P16, P18, P20, P40, P42, P44, P46, P22, P24, P26, P28,
            P30, P32, P34, P36, P48, P50 ,P38])

def Normalizar(v):
    modulo = np.sqrt((v[0]**2)+(v[1]**2)+(v[2]**2))
    VectorNormalizado = [v[0]/modulo,v[1]/modulo,v[2]/modulo]
    return np.array(VectorNormalizado)
        
def Escalar(Puntos,velocidad):
    Scale = []
    for i in range(len(Puntos)):
        j = (i+1)%len(Puntos)
        V = Puntos[j]-Puntos[i]
        d = np.sqrt((V[0]**2)+(V[1]**2)+(V[2]**2))
        a=d*velocidad
        Scale.append(a)
    return Scale

def Tangentes(Puntos):
    Tangentes = []
    for i in range(len(Puntos)):
        j = (i+1)%len(Puntos)
        V1 = Normalizar(Puntos[i-1]-Puntos[i])
        V2 = Normalizar(Puntos[j]-Puntos[i])
        Tangente = Normalizar(V2-V1)
        Tangentes.append(Tangente)
    return Tangentes
    

#################### Pista General    
    
def PistaDeCarreras(PI,PD,N,velocidad):
    Pista = []

    TI = Tangentes(PI)
    TD = Tangentes(PD)
    SI = Escalar(PI,velocidad)
    SD = Escalar(PD,velocidad)

    indice=0

    for i in range(len(PI)):
        j = (i+1)%len(PI)
        si = SI[i]
        sd = SD[i]
        PI1 = np.array([[PI[i][0], PI[i][1], PI[i][2]]]).T
        PI2 = np.array([[PI[j][0], PI[j][1], PI[j][2]]]).T
        TI1 = np.array([[TI[i][0], TI[i][1], TI[i][2]]]).T*si
        TI2 = np.array([[TI[j][0], TI[j][1], TI[j][2]]]).T*si

        PD1 = np.array([[PD[i][0], PD[i][1], PD[i][2]]]).T
        PD2 = np.array([[PD[j][0], PD[j][1], PD[j][2]]]).T
        TD1 = np.array([[TD[i][0], TD[i][1], TD[i][2]]]).T*sd
        TD2 = np.array([[TD[j][0], TD[j][1], TD[j][2]]]).T*sd

        pista,last = generatePistaTexture(PI1,PI2,PD1,PD2,TI1,TI2,TD1,TD2,N,indice)

        Pista.append(pista)
        indice=last
    
    vertices=[]
    indices =[]
    for i in range(len(Pista)):
        a=Pista[i]
        vertex=a.vertices
        vertices+=vertex
        index=a.indices
        indices+=index
    largo=len(indices)
    a=indices[largo-2]
    b=indices[largo-3]
    c=0
    d=1
    indices+=[a,b,0,0,1,a]
        
    return bs.Shape(vertices,indices)


def distance(P1,P2):
    return np.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2)

def heron(P1,P2,P3):
    a=distance(P1,P2)
    b=distance(P2,P3)
    c=distance(P3,P1)
    s=(a+b+c)/2
    area=np.sqrt( s * (s-a) * (s-b) * (s-c) )
    return area
    
def dentro(P1,P2,P3,P):             # Función que determina si estoy dentro de un triángulo
    t0=heron(P1,P2,P3)
    t1=heron(P1,P2,P)
    t2=heron(P2,P3,P)
    t3=heron(P3,P1,P)
    if t0-0.1<=t1+t2+t3<=t0+0.1:
        return True
    return False

def createTextureNormalQuad(image_filename, nx=1, ny=1):

    # Defining locations and texture coordinates for each vertex of the shape    
    vertices = [
    #   positions        texture
        -0.5, -0.5, 0.0,  0, ny,0,0,1,
         0.5, -0.5, 0.0, nx, ny,0,0,1,
         0.5,  0.5, 0.0, nx, 0,0,0,1,
        -0.5,  0.5, 0.0,  0, 0,0,0,1]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         0, 1, 2,
         2, 3, 0]

    textureFileName = image_filename

    return Shape(vertices, indices, textureFileName)


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Crazy Racer", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program
    pipeline = es.SimpleModelViewProjectionShaderProgram()
    pipeline2 = l.SimpleTexturePhongShaderProgram()
    textureShaderProgram = es.SimpleTextureModelViewProjectionShaderProgram()

    # Telling OpenGL to use our shader program

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    gpuAxis = es.toGPUShape(bs.createAxis(400))
    
    pistaFinal = PistaDeCarreras(PI,PD,20,1)

    pista_mesh, mesh_triangles, mesh_vertices = create_pista_mesh_Texture(pistaFinal)
    cpuSurface = draw_mesh_Texture(pista_mesh, pistaFinal.vertices,"suelo.jpg")
    
    gpupistaTexture2 = es.toGPUShape(cpuSurface,GL_REPEAT, GL_NEAREST)
    
    gpuTrack = es.toGPUShape(bs.createTextureQuad("mariokart-circuit3.png"),GL_REPEAT,GL_NEAREST)
    
    gpuFondo = es.toGPUShape(bs.createTextureQuad("fondo.jpg"),GL_REPEAT,GL_NEAREST)
    gpuFondo2 = es.toGPUShape(bs.createTextureQuad("fondo2.jpg"),GL_REPEAT,GL_NEAREST)

    
    redCarNode = createCar([1,0,0], [1,1,1])

    t0 = glfw.get_time()
    camera_theta = 0
    carX = 0
    carY = 0
    carZ = 0

    z_previuos=0

    posX = 0
    posY = 0
    posZ = 0

    Dentro = True
    lista = pista_mesh.getTriangleFaceMeshes()
    Triangulo = lista[0]    # Informa el triángulo actual
    last = "ab"             # Informa el último lado pasado
    n=0    

    while not glfw.window_should_close(window):
        
        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Using GLFW to check for input events
        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            camera_theta -= 2 * dt

        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            camera_theta += 2* dt
            
        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            carX += (30*dt)*np.sin(camera_theta)
            carY += (30*dt)*np.cos(camera_theta)

        if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            carX -= (30*dt)*np.sin(camera_theta)
            carY -= (30*dt)*np.cos(camera_theta)

        # Setting up the view transform

        viewPos = np.array([carX - (25+carZ)*np.sin(camera_theta), carY - (25+carZ)*np.cos(camera_theta),   20+carZ])

        view = tr.lookAt(
            viewPos,
            np.array([carX, carY,carZ]),
            np.array([0,0,1])
        )

        projection = tr.frustum(-10, 10, -10, 10, 10, 400)

        model=tr.scale(30,20,8)
    
        glUseProgram(pipeline.shaderProgram)

        # Setting up the projection transform

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
       
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.scale(10,10,10))
        pipeline.drawShape(gpuAxis, GL_LINES)


        ##################################################################################
        lightingPipeline=pipeline2

        glUseProgram(lightingPipeline.shaderProgram)


        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ka"), 0.3, 0.3, 0.3)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Kd"), 0.9, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "lightPosition"), 5+carX, 5+carY, carZ+10)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(lightingPipeline.shaderProgram, "shininess"), 10)
        
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture2)

    

        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            carRotation = tr.rotationZ((np.pi/2)-camera_theta+0.2)
        
        elif (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            carRotation = tr.rotationZ((np.pi/2)-camera_theta-0.2)
                
        else:
            carRotation = tr.rotationZ((np.pi/2)-camera_theta)
            
        
        lightingPipeline=pipeline2

        glUseProgram(lightingPipeline.shaderProgram)


        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Kd"), 0.9, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "lightPosition"), 5+carX, 5+carY, 15)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(lightingPipeline.shaderProgram, "shininess"), 10)
        
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    
        pos1 = mesh_vertices[Triangulo.data.a]
        pos2 = mesh_vertices[Triangulo.data.b]
        pos3 = mesh_vertices[Triangulo.data.c]
        A = [pos1[0],pos1[1],pos1[2],0]
        B = [pos2[0],pos2[1],pos2[2],0]
        C = [pos3[0],pos3[1],pos3[2],0]
        posX = tr.matmul([A,model])
        posY = tr.matmul([B,model])
        posZ = tr.matmul([C,model])
        carZ = max(posX[2],posY[2],posZ[2])   # Asigno su altura nueva como el máximo de la posicion
        

        if not dentro(posX,posY,posZ,[carX,carY,carZ]):
            Dentro= False
            if Triangulo.ab!=None and last!="ab" and Dentro ==False:
                Triangulo=Triangulo.ab           # Si el último es ab, lo guardo para bloquearlo
                last="ab"
                Dentro=True
            if Triangulo.ca!=None and last!="ca" and Dentro== False:
                Triangulo=Triangulo.ca           # Si el último es ca, lo guardo para bloquearlo
                last="ca"
                Dentro=True
            

        model= tr.matmul([
            tr.translate(carX,carY,carZ+5),
            carRotation,
            #tr.rotationY(-angulo),
            tr.scale(10,10,10)
            ])


        ##############################################
        trackTransform = tr.matmul([
            tr.translate(250,250,-6),
            tr.scale(500,500,1)
        ])
        trackTransform4 = tr.matmul([
            tr.translate(-250,250,-6),
            tr.scale(-500,500,1)
        ])
        trackTransform5 = tr.matmul([
            tr.translate(250,-250,-6),
            tr.scale(500,500,1)
        ])
        trackTransform6 = tr.matmul([
            tr.translate(-250,-250,-6),
            tr.scale(500,500,1)
        ])
        ########
        
        fondoTransform= tr.matmul([
            tr.translate(350,175,150),
            tr.scale(700,700,400),
            tr.rotationY(-math.pi/2),
            tr.rotationZ(-math.pi/2)
        ])
        
        fondoTransform2 = tr.matmul([
            tr.translate(-320,175,150),
            tr.scale(700,700,400),
            tr.rotationY(-math.pi/2), 
            tr.rotationZ(-math.pi/2)
        ])
        
        fondoTransform3 = tr.matmul([
            tr.translate(0,500,120),
            tr.scale(700,700,400),
            tr.rotationX(-math.pi/2),
            tr.rotationZ(-math.pi)
        ])
        
        fondoTransform4 = tr.matmul([
            tr.translate(0,-150,120),
            tr.scale(700,700,400),
            tr.rotationX(-math.pi/2),
            tr.rotationZ(-math.pi)
        ])

        glUseProgram(textureShaderProgram.shaderProgram)

        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, trackTransform)
        textureShaderProgram.drawShape(gpuTrack)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, trackTransform4)
        textureShaderProgram.drawShape(gpuTrack)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, trackTransform5)
        textureShaderProgram.drawShape(gpuTrack)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, trackTransform6)
        textureShaderProgram.drawShape(gpuTrack)
        
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, fondoTransform)
        textureShaderProgram.drawShape(gpuFondo)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, fondoTransform2)
        textureShaderProgram.drawShape(gpuFondo)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, fondoTransform3)
        textureShaderProgram.drawShape(gpuFondo2)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, fondoTransform4)
        textureShaderProgram.drawShape(gpuFondo2)

        
        ###############################################################################
        pipelineNormal = l.SimplePhongShaderProgram()
        textureShaderProgram = es.SimpleTextureModelViewProjectionShaderProgram()
        pipelinePhong = pipelineNormal
        pipelinePhongTexture = l.SimpleTexturePhongShaderProgram()
        

        cuerpo = sg.findNode(redCarNode, "cuerpo")
        cuerpo.transform = model
        
        ruedas = sg.findNode(redCarNode, "ruedas")
        ruedas.transform = model
        
        deco = sg.findNode(redCarNode, "deco")
        deco.transform = model
      
        glUseProgram(textureShaderProgram.shaderProgram)

        
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, trackTransform)
        textureShaderProgram.drawShape(gpuTrack)
        
        # Drawing shapes with different model transformations

        glUseProgram(pipelinePhongTexture.shaderProgram)

        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ka"), 0.8, 0.6, 0.6)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Kd"), 0.9, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ks"), 1.0, 1.0, 1.0)


        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "lightPosition"), -5, -5, 5)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "shininess"), 100)
        
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "view"), 1, GL_TRUE, view)


        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(ruedas, pipelinePhongTexture, "model")
        sg.drawSceneGraphNode(deco, pipelinePhongTexture, "model")


        redWheelRotationNode = sg.findNode(redCarNode, "wheelRotation")
        redWheelRotationNode.transform = tr.rotationZ(5 * glfw.get_time())
        
        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            redWheelRotationNode.transform = tr.rotationZ(5 * glfw.get_time())
        
        elif (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            redWheelRotationNode.transform = tr.rotationZ(-5 * glfw.get_time())
                
        else:
            redWheelRotationNode.transform = tr.identity()


        ########################################

        glUseProgram(pipelineNormal.shaderProgram)

        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ka"), 0.4, 0.4, 0.4)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Kd"), 0.9, 0.6, 0.6)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "lightPosition"), 0+carX, 5+carY, 5)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(pipelinePhong.shaderProgram, "shininess"), 80)
        
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "linearAttenuation"), 0.02)
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "quadraticAttenuation"), 0.004)

        glUniformMatrix4fv(glGetUniformLocation(pipelineNormal.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipelineNormal.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipelineNormal.shaderProgram, "model"), 1, GL_TRUE, trackTransform)
        
        # Drawing the Car

        car = sg.findNode(redCarNode, "car")
        
        sg.drawSceneGraphNode(cuerpo, pipelineNormal, "model")
        
        
        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()


# In[ ]:





# In[ ]:




