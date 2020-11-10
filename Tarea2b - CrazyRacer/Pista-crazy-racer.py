#!/usr/bin/env python
# coding: utf-8

# In[15]:


import math
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import lighting_shaders as l
import local_shapes2 as ls 

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
    
    #  Generate a matrix concatenating the columns
    G = np.concatenate((P1, P2, T1, T2), axis=1)
    
    # Hermite base matrix is a constant
    Mh = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])    
    
    return np.matmul(G, Mh) 

    

# M is the cubic curve matrix, N is the number of samples between 0 and 1
def evalCurve(M, N):
    ts = np.linspace(0.0, 1.0, N) 
    
    curve = np.ndarray(shape=(N, 3), dtype=float)
    
    for i in range(len(ts)):
        T = generateT(ts[i])  
        curve[i, 0:3] = np.matmul(M, T).T 
        
    return curve 


def generateMesh(curve, color):

    vertices = []   
    indices = []   

    # We generate a vertex for each sample x,y,z
    for i in range(len(curve)): 
        xyz=curve[i] 
        vertices.extend(xyz) 
        vertices.extend([color[0],color[1],color[2]]) 

    for i in range(len(curve)-1): 
        for j in range(0,2):
            a=i+j
            indices += [a]

    return  bs.Shape(vertices, indices) 


def generatePistaColor(P1,P2,P3,P4,T1,T2,N,color):

    GMh = hermiteMatrix(P1,P2, T1, T2)
    curve1 = evalCurve(GMh, N)
    GMh2 = hermiteMatrix(P3, P4, T1, T2)
    curve2 = evalCurve(GMh2, N1)


    vertices = []
    indices = []

    # Angle step
    start_index = 0

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

        a=np.array([cord[0], cord[1], cord[2]])
        b=np.array([cord3[0], cord3[1], cord3[2]])
        d=np.array([cord2[0], cord2[1], cord2[2]])
        c=np.array([cord4[0], cord4[1], cord4[2]])

        _vertex, _indices = ls.createNormalsQuadIndexation(start_index, a, b, c, d)

        vertices += _vertex
        indices  += _indices
        start_index += 4
    return bs.Shape(vertices,indices)


 
def generatePistaTexture(P1,P2,P3,P4,T1,T2,N,imagen):

    GMh = hermiteMatrix(P1,P2, T1, T2)
    curve1 = evalCurve(GMh, N)
       
    GMh2 = hermiteMatrix(P3, P4, T1, T2)
    curve2 = evalCurve(GMh2, N)


    vertices = []
    indices = []

    # Angle step
    start_index = 0

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

        a=np.array([cord[0], cord[1], cord[2]])
        b=np.array([cord3[0], cord3[1], cord3[2]])
        d=np.array([cord2[0], cord2[1], cord2[2]])
        c=np.array([cord4[0], cord4[1], cord4[2]])

        _vertex, _indices = ls.createTextureNormalsQuadIndexation(start_index, a, b, c, d)

        vertices += _vertex
        indices  += _indices
        start_index += 4
    return bs.Shape(vertices,indices,imagen)


#################################### Puntos y Tangentes para la Pista
###### Primeros puntos
P1 = np.array([[2,0, 0]]).T
P2 = np.array([[2,10, 0]]).T

T1 = np.array([[0,10, 0]]).T
T2 = np.array([[-4,10, 0]]).T 

P3 = np.array([[4,0, 0]]).T
P4 = np.array([[4,10, 0]]).T

###### Resto de puntos
P5 = np.array([[-5, 12, 0]]).T
P6 = np.array([[-5, 14, 0]]).T
T5 = np.array([[-10,0,0]]).T 

P7 = np.array([[-12, 18, 0]]).T
P8 = np.array([[-10, 18, 0]]).T
T7 = np.array([[0,10,0]]).T

P9 = np.array([[-5, 24, 0]]).T
P10 = np.array([[-5,22, 0]]).T
T9 = np.array([[10,0,0]]).T 

P11 = np.array([[0, 26, 1]]).T
P12 = np.array([[2, 26, 1]]).T
T11 = np.array([[0,10,0]]).T 

P13 = np.array([[8, 32, 1]]).T
P14 = np.array([[8, 30, 1]]).T
T13 = np.array([[10,0,0]]).T 

P15 = np.array([[12, 24, 2]]).T
P16 = np.array([[10, 24, 2]]).T
T15 = np.array([[0,-10,0]]).T 

P17 = np.array([[14, 18, 3]]).T
P18 = np.array([[14, 16, 3]]).T
T17 = np.array([[10,0,0]]).T 

P19 = np.array([[18, 12, 4]]).T
P20 = np.array([[16, 12, 4]]).T
T19 = np.array([[0,-10,0]]).T 

#### Elicoide

P21 = np.array([[18, 4, 5]]).T
P22 = np.array([[16, 4, 5]]).T
T21 = np.array([[0,-10,0]]).T 

P23 = np.array([[12, -2, 4]]).T
P24 = np.array([[12, 0, 4]]).T
T23 = np.array([[-10,0,0]]).T 

P25 = np.array([[6, 4, 3]]).T
P26 = np.array([[8, 4, 3]]).T
T25 = np.array([[0,10,0]]).T 

P27 = np.array([[12, 10, 2]]).T
P28 = np.array([[12, 8, 2]]).T
T27 = np.array([[10,0,0]]).T 

P29 = np.array([[18, 4, 1]]).T
P30 = np.array([[16, 4, 1]]).T
T29 = np.array([[0,-10,0]]).T 

#### Termina la elicoide

P31 = np.array([[18, -4, 0]]).T
P32 = np.array([[16, -4, 0]]).T
T31 = np.array([[0,-10,0]]).T 

P33 = np.array([[12, -8, 1]]).T
P34 = np.array([[12, -6, 1]]).T
T33 = np.array([[-10,0,0]]).T 

P35 = np.array([[6, -4, 2]]).T
P36 = np.array([[8, -4, 2]]).T
T35 = np.array([[10,0,0]]).T 

### Unión final con los primeros puntos

P37 = np.array([[2, 0, 0]]).T
P38 = np.array([[4, 0, 0]]).T
T37 = np.array([[10,0,0]]).T 

GMh = hermiteMatrix(P1, P2, T1, T2)
N = 50 
hermiteCurve = evalCurve(GMh, N)
        

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "2D Plotter", None, None)

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
    gpupistaTexture2 = es.toGPUShape(generatePistaTexture(P1,P2,P3,P4,T1,T2,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture3 = es.toGPUShape(generatePistaTexture(P2,P5,P4,P6,T2,T5,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture4 = es.toGPUShape(generatePistaTexture(P5,P7,P6,P8,T5,T7,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture5 = es.toGPUShape(generatePistaTexture(P7,P9,P8,P10,T7,T9,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture6 = es.toGPUShape(generatePistaTexture(P9,P11,P10,P12,T9,T11,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture7 = es.toGPUShape(generatePistaTexture(P11,P13,P12,P14,T11,T13,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture8 = es.toGPUShape(generatePistaTexture(P13,P15,P14,P16,T13,T15,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture9 = es.toGPUShape(generatePistaTexture(P15,P17,P16,P18,T15,T17,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture10 = es.toGPUShape(generatePistaTexture(P17,P19,P18,P20,T17,T19,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture11 = es.toGPUShape(generatePistaTexture(P19,P21,P20,P22,T19,T21,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture12 = es.toGPUShape(generatePistaTexture(P21,P23,P22,P24,T21,T23,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture13 = es.toGPUShape(generatePistaTexture(P23,P25,P24,P26,T23,T25,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture14 = es.toGPUShape(generatePistaTexture(P25,P27,P26,P28,T25,T27,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture15 = es.toGPUShape(generatePistaTexture(P27,P29,P28,P30,T27,T29,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture16 = es.toGPUShape(generatePistaTexture(P29,P31,P30,P32,T29,T31,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpupistaTexture17 = es.toGPUShape(generatePistaTexture(P31,P33,P32,P34,T31,T33,30,"suelo.jpg"),GL_REPEAT,GL_NEAREST)
    gpucubo = es.toGPUShape(bs.createTextureNormalsCube("logo1.jpg"),GL_REPEAT,GL_NEAREST)
    #gpupista= es.toGPUShape(generatePista(hermiteCurve,2,[0,0,0]))

    gpuTrack = es.toGPUShape(bs.createTextureQuad("mariokart-circuit3.png"),GL_REPEAT,GL_NEAREST)


    #gpuSurface = es.toGPUShape(cpuSurface)

    t0 = glfw.get_time()
    camera_theta = 0
    carX = 0
    carY = 0

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
            carX += (70*dt)*np.sin(camera_theta)
            carY += (70*dt)*np.cos(camera_theta)

        if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            carX -= (70*dt)*np.sin(camera_theta)
            carY -= (70*dt)*np.cos(camera_theta)

        # Setting up the view transform

        viewPos = np.array([carX - 20*np.sin(camera_theta), carY - 20*np.cos(camera_theta), 15])

        view = tr.lookAt(
            viewPos,
            np.array([carX+3, carY+3,0]),
            np.array([0,0,1])
        )

        #projection = tr.perspective(60, float(width)/float(height), 0.1, 100)
        projection = tr.frustum(-10, 10, -10, 10, 10, 400)

        model=tr.scale(20,20,8)
    

        glUseProgram(pipeline.shaderProgram)

        # Setting up the projection transform

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
       
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.scale(10,10,10))
        pipeline.drawShape(gpuAxis, GL_LINES)


        #glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.translate(-4,-2,0))
        #pipeline.drawShape(gpupista)


        lightingPipeline=pipeline2

        glUseProgram(lightingPipeline.shaderProgram)


        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Kd"), 0.9, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "lightPosition"), 0+carX, 5+carY, 7)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(lightingPipeline.shaderProgram, "shininess"), 10)
        
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

        ############################## Dinujo la pista entera ######################################

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture2)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture3)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture4)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture5)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture6)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture7)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture8)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture9)
        
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture10)
        
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture11)
        
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture12)
        
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture13)
        
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture14)
        
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture15)
        
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture16)
        
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture17)

        #######################################################################################################  
        
        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            carRotation = tr.rotationZ((np.pi/2)-camera_theta+0.2)
        
        elif (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            carRotation = tr.rotationZ((np.pi/2)-camera_theta-0.2)
                
        else:
            carRotation = tr.rotationZ((np.pi/2)-camera_theta)



        model= tr.matmul([
            tr.translate(carX,carY,0),
            carRotation,
            tr.scale(8,12,8)
            ])

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpucubo)

   
      ##########################################################################################################################

        trackTransform = tr.matmul([
            tr.translate(250,250,-3),
            tr.uniformScale(500)
        ])
        trackTransform2 = tr.matmul([
            tr.translate(250,750,-3),
            tr.uniformScale(500)
        ])
        trackTransform3 = tr.matmul([
            tr.translate(250,1250,-3),
            tr.uniformScale(500)
        ])
        trackTransform4 = tr.matmul([
            tr.translate(-250,250,-3),
            tr.scale(-500,500,500)
        ])
        trackTransform5 = tr.matmul([
            tr.translate(-250,750,-3),
            tr.uniformScale(500)
        ])
        trackTransform6 = tr.matmul([
            tr.translate(-250,1250,-3),
            tr.uniformScale(500)
        ])

        glUseProgram(textureShaderProgram.shaderProgram)

        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, trackTransform)
        textureShaderProgram.drawShape(gpuTrack)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, trackTransform2)
        textureShaderProgram.drawShape(gpuTrack)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, trackTransform3)
        textureShaderProgram.drawShape(gpuTrack)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, trackTransform4)
        textureShaderProgram.drawShape(gpuTrack)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, trackTransform5)
        textureShaderProgram.drawShape(gpuTrack)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, trackTransform6)
        textureShaderProgram.drawShape(gpuTrack)


        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)


        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()
    


# In[ ]:




