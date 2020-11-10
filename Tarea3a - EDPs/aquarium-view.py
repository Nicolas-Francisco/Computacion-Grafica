#!/usr/bin/env python
# coding: utf-8

# In[59]:


import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import random
import json

import transformations as tr
import easy_shaders as es
import scene_graph as sg
import basic_shapes as bs
import acuario as acuario
import pez as pez1
import pez2 as pez2
import pez3 as pez3


PROJECTION_ORTHOGRAPHIC = 0
PROJECTION_FRUSTUM = 1
PROJECTION_PERSPECTIVE = 2

archivo = sys.argv[1]
Data = None
with open(archivo) as file:
    data = json.load(file)
    Data = data

def createColorCube(i, j, k, X, Y, Z, color):
    l_x = X[i, j, k]
    r_x = X[i+1, j, k]
    b_y = Y[i, j, k]
    f_y = Y[i, j+1, k]
    b_z = Z[i, j, k]
    t_z = Z[i, j, k+1]
    c = color
    #   positions    colors
    vertices = [
    # Z+: number 1
        l_x, b_y,  t_z, c[0],c[1],c[2],
         r_x, b_y,  t_z, c[0],c[1],c[2],
         r_x,  f_y,  t_z, c[0],c[1],c[2],
        l_x,  f_y,  t_z, c[0],c[1],c[2],
    # Z-: number 6
        l_x, b_y, b_z, c[0],c[1],c[2],
         r_x, b_y, b_z, c[0],c[1],c[2],
         r_x,  f_y, b_z, c[0],c[1],c[2],
        l_x,  f_y, b_z, c[0],c[1],c[2],
    # X+: number 5
         r_x, b_y, b_z, c[0],c[1],c[2],
         r_x,  f_y, b_z, c[0],c[1],c[2],
         r_x,  f_y,  t_z, c[0],c[1],c[2],
         r_x, b_y,  t_z, c[0],c[1],c[2],
    # X-: number 2
        l_x, b_y, b_z, c[0],c[1],c[2],
        l_x,  f_y, b_z, c[0],c[1],c[2],
        l_x,  f_y,  t_z, c[0],c[1],c[2],
        l_x, b_y,  t_z, c[0],c[1],c[2],
    # Y+: number 4
        l_x,  f_y, b_z, c[0],c[1],c[2],
        r_x,  f_y, b_z, c[0],c[1],c[2],
        r_x,  f_y, t_z, c[0],c[1],c[2],
        l_x,  f_y, t_z, c[0],c[1],c[2],
    # Y-: number 3
        l_x, b_y, b_z, c[0],c[1],c[2],
        r_x, b_y, b_z, c[0],c[1],c[2],
        r_x, b_y, t_z, c[0],c[1],c[2],
        l_x, b_y, t_z, c[0],c[1],c[2],
        ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        4, 5, 1, 1, 0, 4,
        6, 7, 3, 3, 2, 6,
        5, 6, 2, 2, 1, 5,
        7, 4, 0, 0, 3, 7]

    return bs.Shape(vertices, indices)

def merge(destinationShape, strideSize, sourceShape):

    # current vertices are an offset for indices refering to vertices of the new shape
    offset = len(destinationShape.vertices)
    destinationShape.vertices += sourceShape.vertices
    destinationShape.indices += [(offset/strideSize) + index for index in sourceShape.indices]
    
    
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.projection = PROJECTION_PERSPECTIVE
        self.TipoA = False
        self.TipoB = False
        self.TipoC = False
        self.ZOOM = 0

# We will use the global controller as communication with the callback function
controller = Controller()

def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_1:
        print('Orthographic projection')
        controller.projection = PROJECTION_ORTHOGRAPHIC

    elif key == glfw.KEY_2:
        print('Frustum projection')
        controller.projection = PROJECTION_FRUSTUM

    elif key == glfw.KEY_3:
        print('Perspective projection')
        controller.projection = PROJECTION_PERSPECTIVE

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    
    elif key == glfw.KEY_A:
        print('Tipo A')
        controller.TipoA = not controller.TipoA
    
    elif key == glfw.KEY_B:
        print('Tipo B')
        controller.TipoB = not controller.TipoB
        
    elif key == glfw.KEY_C:
        print('Tipo C')
        controller.TipoC = not controller.TipoC
        

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Aquarium view", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program
    pipeline = es.SimpleModelViewProjectionShaderProgram()
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()

    # Telling OpenGL to use our shader program
    glUseProgram(pipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.08, 0.4, 1, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    gpuAxis = es.toGPUShape(bs.createAxis(7))
    
    # Load potentials and grid
    
    file = Data["filename"]
    ta = int(Data["t_a"])
    tb = int(Data["t_b"])
    tc = int(Data["t_c"])
    
    load_voxels = np.load(file)
    
    W = load_voxels.shape[0]
    L = load_voxels.shape[1]
    H = load_voxels.shape[2]
    
    Y, X, Z = np.meshgrid(np.linspace(0, L ,L), np.linspace(0, W, W), np.linspace(0, H, H))

    isosurfaceA = bs.Shape([], [])
    isosurfaceB = bs.Shape([], [])
    isosurfaceC = bs.Shape([], [])
    
    # Now let's draw voxels!
    for i in range(X.shape[0]-1):
        for j in range(X.shape[1]-1):
            for k in range(X.shape[2]-1):
                Temp = load_voxels[i,j,k]
                if Temp <= (ta+2) and Temp >= (ta-1):
                    temp_shape = createColorCube(i,j,k, X,Y, Z, [0.9, 0.9, 0])
                    merge(destinationShape=isosurfaceA, strideSize=6, sourceShape=temp_shape)
                if Temp <= (tb+2) and Temp >= (tb-1):
                    temp_shape = createColorCube(i,j,k, X,Y, Z, [0, 0.6, 1])
                    merge(destinationShape=isosurfaceB, strideSize=6, sourceShape=temp_shape)
                if Temp <= (tc+2) and Temp >= (tc-1):
                    temp_shape = createColorCube(i,j,k, X,Y, Z, [1, 0.2, 0])
                    merge(destinationShape=isosurfaceC, strideSize=6, sourceShape=temp_shape)

    gpu_surfaceA = es.toGPUShape(isosurfaceA)
    gpu_surfaceB = es.toGPUShape(isosurfaceB)
    gpu_surfaceC = es.toGPUShape(isosurfaceC)
    
    
    # Acuario y peces
    AquariumNode = acuario.createAquarium()

    na = int(Data["n_a"])
    nb = int(Data["n_b"])
    nc = int(Data["n_c"])
    
    # Tipo A : Agua tibia (peces globo)
    PosicionesA = []
    for vertice in range(0,len(isosurfaceA.vertices), 6):
        # Guardo los vértices de cada cubo en el que sé se cumple la temperatura
        Pos = (isosurfaceA.vertices[vertice]*3/W, isosurfaceA.vertices[vertice+1]*6/L, isosurfaceA.vertices[vertice+2]*4/H)
        PosicionesA.append(Pos)
    
    GpuPecesA = []
    GpuPosicionesA = []
    rotA = []
    for k in range(na):
        # Creo los nodos de peces y los guardo para dibujarlos
        pez_tipoA = pez3.createFish([0.52,0.25,0], [0.7,0.5,0.3])   # pez-globo para agua tibia
        pez_tipoA_pos = random.choice(PosicionesA)
        rotationA = random.randint(0,6)
        GpuPecesA.append(pez_tipoA)
        GpuPosicionesA.append(pez_tipoA_pos)
        rotA.append(rotationA)
        
        
    
    # Tipo B : Agua fria (peces cril)
    PosicionesB = []
    for vertice in range(0,len(isosurfaceB.vertices), 6):
        # Guardo los vértices de cada cubo en el que sé se cumple la temperatura
        Pos = (isosurfaceB.vertices[vertice]*3/W, isosurfaceB.vertices[vertice+1]*6/L, isosurfaceB.vertices[vertice+2]*4/H)
        PosicionesB.append(Pos)
    
    GpuPecesB = []
    GpuPosicionesB = []
    rotB = []
    for k in range(nb):
        # Creo los nodos de peces y los guardo para dibujarlos
        pez_tipoB = pez2.createFish([0,1,0], [0.5,1,0.5])   # Cril para agua fria
        pez_tipoB_pos = random.choice(PosicionesB)
        rotationB = random.randint(0,6)
        GpuPecesB.append(pez_tipoB)
        GpuPosicionesB.append(pez_tipoB_pos)
        rotB.append(rotationB)
        
        
    
    # Tipo C : Agua caliente (peces payaso)
    PosicionesC = []
    for vertice in range(0,len(isosurfaceC.vertices), 6):
        # Guardo los vértices de cada cubo en el que sé se cumple la temperatura
        Pos = (isosurfaceC.vertices[vertice]*3/W, isosurfaceC.vertices[vertice+1]*6/L, isosurfaceC.vertices[vertice+2]*4/H)
        PosicionesC.append(Pos)
    
    GpuPecesC = []
    GpuPosicionesC = []
    rotC = []
    for k in range(nc):
        # Creo los nodos de peces y los guardo para dibujarlos
        pez_tipoC = pez1.createFish([1,0,0], [1,0.5,0.5])   # Semi-pez-payaso para agua caliente
        pez_tipoC_pos = random.choice(PosicionesC)
        rotationC = random.randint(0,6)
        GpuPecesC.append(pez_tipoC)
        GpuPosicionesC.append(pez_tipoC_pos)
        rotC.append(rotationC)
    
    
    t0 = glfw.get_time()
    camera_theta = np.pi/4

    while not glfw.window_should_close(window):
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
            controller.ZOOM += 2 * dt
            
        if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            controller.ZOOM -= 2 * dt

        # Setting up the view transform

        camX = 1.1*1.5 + (9 - controller.ZOOM) * np.sin(camera_theta)
        camY = 1.1*3 + (9 - controller.ZOOM) * np.cos(camera_theta)

        viewPos = np.array([camX, camY, (10 - controller.ZOOM)])

        view = tr.lookAt(
            viewPos,
            np.array([1.5,3,0]),
            np.array([0,0,1])
        )

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)

        # Setting up the projection transform

        if controller.projection == PROJECTION_ORTHOGRAPHIC:
            projection = tr.ortho(-8, 8, -8, 8, 0.1, 100)

        elif controller.projection == PROJECTION_FRUSTUM:
            projection = tr.frustum(-5, 5, -5, 5, 9, 100)

        elif controller.projection == PROJECTION_PERSPECTIVE:
            projection = tr.perspective(60, float(width)/float(height), 0.1, 100)
        
        else:
            raise Exception()

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)


        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Drawing shapes with different model transformations
        
        #glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, 
        #                   tr.matmul([tr.translate(1.5,3,0), tr.uniformScale(1)]))
        #pipeline.drawShape(gpuAxis, GL_LINES)
        
        if controller.TipoA:
            glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, 
                           tr.matmul([tr.scale(1.1*3/W, 1.1*6/L, 1.1*4/H)]))
            pipeline.drawShape(gpu_surfaceA, GL_LINES)
        if controller.TipoB:
            glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, 
                           tr.matmul([tr.scale(1.1*3/W, 1.1*6/L, 1.1*4/H)]))
            pipeline.drawShape(gpu_surfaceB, GL_LINES)
        if controller.TipoC:
            glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, 
                           tr.matmul([tr.scale(1.1*3/W, 1.1*6/L, 1.1*4/H)]))
            pipeline.drawShape(gpu_surfaceC, GL_LINES)
            
        
        AquariumNode.transform = tr.matmul([tr.translate(1.1*1.5,1.1*3,0), tr.rotationZ(np.pi/2),
                                            tr.scale(1.1,1.1,1.1)])
        sg.drawSceneGraphNode(AquariumNode, mvpPipeline, "model")
        
        for k in range(0, len(GpuPecesA)):
            GpuPecesA[k].transform = tr.matmul([tr.translate(GpuPosicionesA[k][0], GpuPosicionesA[k][1], 
                                                            GpuPosicionesA[k][2]), 
                                                tr.uniformScale(0.3), tr.rotationZ(rotA[k])])
            sg.drawSceneGraphNode(GpuPecesA[k], mvpPipeline, "model")
        
        for k in range(0, len(GpuPecesB)):
            GpuPecesB[k].transform = tr.matmul([tr.translate(GpuPosicionesB[k][0], GpuPosicionesB[k][1], 
                                                            GpuPosicionesB[k][2]), 
                                                tr.uniformScale(0.2), tr.rotationZ(rotB[k])])
            sg.drawSceneGraphNode(GpuPecesB[k], mvpPipeline, "model")
        
        for k in range(0, len(GpuPecesC)):
            GpuPecesC[k].transform = tr.matmul([tr.translate(GpuPosicionesC[k][0], GpuPosicionesC[k][1], 
                                                            GpuPosicionesC[k][2]), 
                                                tr.uniformScale(0.5), tr.rotationZ(rotC[k])])
            sg.drawSceneGraphNode(GpuPecesC[k], mvpPipeline, "model")

        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()


# In[ ]:




