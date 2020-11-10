#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glfw
import math
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import transformations as tr
import basic_shapes as bs
import scene_graph as sg 
import easy_shaders as es
import lighting_shaders as ls
import local_shapes2 as lshapes


PROJECTION_ORTHOGRAPHIC = 0
PROJECTION_FRUSTUM = 1
PROJECTION_PERSPECTIVE = 2


# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.projection = PROJECTION_ORTHOGRAPHIC


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
        
        
# def paraboloid(x, y, a, b):
#    return (x*x) / (a*a) + (y*y) / (b*b)
        
def Paraboloide(x,y,r):
    z=-(x**2 +y**2) +3
    if z>-1:
        return z
        
def Esfera(x,y,r):
    z=np.sqrt(r**2-(x**2 +y**2) )
    if z>-1:
        return z

def createCar(color1, color2):
    
    gpuBlackCube = es.toGPUShape(bs.createColorNormalsCube(0,0,0))
    gpuCube1 = es.toGPUShape(bs.createColorNormalsCube(color1[0], color1[1], color1[2]))
    gpuCube2 = es.toGPUShape(bs.createColorNormalsCube(color2[0], color2[1], color2[2]))
    gpuCylinder = es.toGPUShape(lshapes.generateTextureNormalsCylinder(15, "neumatico.jpeg" , 3, 2, 1),GL_REPEAT,GL_NEAREST)
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

    

def generateMesh(xs, ys, function, color):

    vertices = []
    indices = []

    # We generate a vertex for each sample x,y,z
    for i in range(len(xs)):
        for j in range(len(ys)):
            x = xs[i]
            y = ys[j]
            z = function(x, y)
            
            vertices += [x, y, z] + color

    # The previous loops generates full columns j-y and then move to
    # the next i-x. Hence, the index for each vertex i,j can be computed as
    index = lambda i, j: i*len(ys) + j 
    
    # We generate quads for each cell connecting 4 neighbor vertices
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):

            # Getting indices for all vertices in this quad
            isw = index(i,j)
            ise = index(i+1,j)
            ine = index(i+1,j+1)
            inw = index(i,j+1)

            # adding this cell's quad as 2 triangles
            indices += [
                isw, ise, ine,
                ine, inw, isw
            ]

    return bs.Shape(vertices, indices)

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
    pipelineNormal = ls.SimplePhongShaderProgram()
    textureShaderProgram = es.SimpleTextureModelViewProjectionShaderProgram()
    pipelinePhong = pipelineNormal
    pipelinePhongTexture = ls.SimpleTexturePhongShaderProgram()

    # Setting up the clear screen color
    glClearColor(0.0, 0.6, 0.9, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)


    # generate a numpy array with 40 samples between -10 and 10
    xs = np.ogrid[-10:10:40j]
    ys = np.ogrid[-10:10:40j]
  
    
    redCarNode = createCar([1,0,0], [1,1,1])

    gpuTrack = es.toGPUShape(bs.createTextureQuad('mariokart-circuit3.png'),GL_REPEAT,GL_NEAREST)

    t0 = glfw.get_time()
    camera_theta = 0
    carX = 200
    carY = -100

    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()
        
        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        
        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        projection = tr.frustum(-5, 5, -5, 5, 9, 400)

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            camera_theta -= 1.5 * dt

        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            camera_theta += 1.5 * dt
            
        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            carX += (60*dt)*np.sin(camera_theta)
            carY += (60*dt)*np.cos(camera_theta)

        if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            carX -= (60*dt)*np.sin(camera_theta)
            carY -= (60*dt)*np.cos(camera_theta)

            
        ###############################
        
        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            carRotation = tr.rotationZ((np.pi/2)-camera_theta+0.2)
        
        elif (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            carRotation = tr.rotationZ((np.pi/2)-camera_theta-0.2)
                
        else:
            carRotation = tr.rotationZ((np.pi/2)-camera_theta)

        # Setting up the view transform

        viewPos = np.array([carX - 14*np.sin(camera_theta), carY - 14*np.cos(camera_theta), 0.2])

        view = tr.lookAt(
            viewPos,
            np.array([carX, carY, 0]),
            np.array([0,0,1])
        )
        
        
        # Drawing shapes with different model transformations
            
        trackTransform = tr.matmul([
            tr.translate(0,0,-10),
            tr.uniformScale(500)
        ])
        
        # Setting up the projection transform
        
        transf = tr.matmul([tr.translate(carX,carY,-4), tr.uniformScale(3), carRotation])

        cuerpo = sg.findNode(redCarNode, "cuerpo")
        cuerpo.transform = transf
        
        ruedas = sg.findNode(redCarNode, "ruedas")
        ruedas.transform = transf
        
        deco = sg.findNode(redCarNode, "deco")
        deco.transform = transf
      
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
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(pipelineNormal.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipelineNormal.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipelineNormal.shaderProgram, "model"), 1, GL_TRUE, trackTransform)
        
        # Drawing the Car

        car = sg.findNode(redCarNode, "car")
        
        sg.drawSceneGraphNode(cuerpo, pipelineNormal, "model")
    
        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)
        

    glfw.terminate()


# In[ ]:




