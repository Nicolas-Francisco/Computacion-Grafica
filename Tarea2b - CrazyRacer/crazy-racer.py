#!/usr/bin/env python
# coding: utf-8

# In[12]:


import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import transformations as tr
import basic_shapes as bs
import scene_graph as sg
import easy_shaders as es
import lighting_shaders as ls


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
    
    
    # Cheating a single wheel
    wheel = sg.SceneGraphNode("wheel")
    wheel.transform = tr.scale(0.2, 0.1, 0.2)
    wheel.childs += [gpuBlackCube]

    wheelRotation = sg.SceneGraphNode("wheelRotation")
    wheelRotation.childs += [wheel]

    # Instanciating 2 wheels, for the front and back parts
    frontRightWheel = sg.SceneGraphNode("frontRightWheel")
    frontRightWheel.transform = tr.translate(0.35, 0.3, -0.3)
    frontRightWheel.childs += [wheelRotation]
    
    frontLeftWheel = sg.SceneGraphNode("frontLeftWheel")
    frontLeftWheel.transform = tr.translate(0.35, -0.3, -0.3)
    frontLeftWheel.childs += [wheelRotation]

    backRightWheel = sg.SceneGraphNode("backRightWheel")
    backRightWheel.transform = tr.translate(-0.35, 0.3, -0.3)
    backRightWheel.childs += [wheelRotation]
    
    backLeftWheel = sg.SceneGraphNode("backLeftWheel")
    backLeftWheel.transform = tr.translate(-0.35, -0.3, -0.3)
    backLeftWheel.childs += [wheelRotation]
    
    # Creating the chasis of the car
    Cube1 = sg.SceneGraphNode("chasis1")
    Cube1.transform = tr.matmul([tr.translate(0, 0, -0.25), tr.scale(1.2, 0.5, 0.2)])
    Cube1.childs += [gpuCube1]
    
    Cube2 = sg.SceneGraphNode("chasis2")
    Cube2.transform = tr.matmul([tr.translate(0.57, -0.2, -0.15), tr.scale(0.1, 0.1, 0.1)])
    Cube2.childs += [gpuCube2]
    
    Cube3 = sg.SceneGraphNode("chasis3")
    Cube3.transform = tr.matmul([tr.translate(0.57, 0.2, -0.15), tr.scale(0.1, 0.1, 0.1)])
    Cube3.childs += [gpuCube2]
    
    Cube4 = sg.SceneGraphNode("chasis1")
    Cube4.transform = tr.matmul([tr.translate(-0.05, 0, -0.15), tr.scale(1.1, 0.5, 0.1)])
    Cube4.childs += [gpuCube1]
    
    Cube5 = sg.SceneGraphNode("chasis1")
    Cube5.transform = tr.matmul([tr.translate(0.56, 0, -0.15), tr.scale(0.1, 0.3, 0.1)])
    Cube5.childs += [gpuBlackCube]
    
    Cube6 = sg.SceneGraphNode("chasis1")
    Cube6.transform = tr.matmul([tr.translate(0, 0, -0.05), tr.scale(1.2, 0.5, 0.1)])
    Cube6.childs += [gpuCube1]
    
    Cube7 = sg.SceneGraphNode("chasis1")
    Cube7.transform = tr.matmul([tr.translate(-0.1, 0, 0.05), tr.scale(0.6, 0.5, 0.3)])
    Cube7.childs += [gpuCube1]
    
    Cube8 = sg.SceneGraphNode("chasis3")
    Cube8.transform = tr.matmul([tr.translate(0.2, 0.01, 0.05), tr.scale(0.02, 0.45, 0.2)])
    Cube8.childs += [gpuCube2]

    # All pieces together
    car = sg.SceneGraphNode("chasis")
    car.childs += [Cube1]
    car.childs += [Cube2]
    car.childs += [Cube3]
    car.childs += [Cube4]
    car.childs += [Cube5]
    car.childs += [Cube6]
    car.childs += [Cube7]
    car.childs += [Cube8]
    car.childs += [frontRightWheel]
    car.childs += [frontLeftWheel]
    car.childs += [backRightWheel]
    car.childs += [backLeftWheel]

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

    window = glfw.create_window(width, height, "Projections Demo", None, None)

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

    # Setting up the clear screen color
    glClearColor(0.0, 0.6, 0.9, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    
    simpleEsfera = lambda x, y: Esfera(x, y, 8)
    simpleParaboloide = lambda x, y: Paraboloide(x, y, 8)

    # generate a numpy array with 40 samples between -10 and 10
    xs = np.ogrid[-10:10:40j]
    ys = np.ogrid[-10:10:40j]
  
    
    redCarNode = createCar([1,0,0], [1,1,1])

    gpuTrack = es.toGPUShape(bs.createTextureQuad('mariokart-circuit3.png'),GL_REPEAT,GL_NEAREST)

    t0 = glfw.get_time()
    camera_theta = 0
    carX = 30
    carY = -20

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

        viewPos = np.array([carX - 10*np.sin(camera_theta), carY - 10*np.cos(camera_theta), 0.2])

        view = tr.lookAt(
            viewPos,
            np.array([carX, carY,0]),
            np.array([0,0,1])
        )
        
        
        # Drawing shapes with different model transformations
            
        trackTransform = tr.matmul([
            tr.translate(0,0,-10),
            tr.uniformScale(900)
        ])
        
        # Setting up the projection transform
        
        glUseProgram(textureShaderProgram.shaderProgram)

        
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "model"), 1, GL_TRUE, trackTransform)
        textureShaderProgram.drawShape(gpuTrack)
        
        # Drawing shapes with different model transformations
        glUseProgram(pipelineNormal.shaderProgram)

        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Kd"), 0.9, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "lightPosition"), 0+carX, 5+carY, 5)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(pipelinePhong.shaderProgram, "shininess"), 100)
        
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(pipelineNormal.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipelineNormal.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipelineNormal.shaderProgram, "model"), 1, GL_TRUE, trackTransform)
        
    
        # Moving the red car and rotating its wheels
        redWheelRotationNode = sg.findNode(redCarNode, "wheelRotation")
        redWheelRotationNode.transform = tr.rotationY(5 * glfw.get_time())
        
        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            redWheelRotationNode.transform = tr.rotationY(5 * glfw.get_time())
        
        elif (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            redWheelRotationNode.transform = tr.rotationY(-5 * glfw.get_time())
                
        else:
            redWheelRotationNode.transform = tr.identity()



        # Uncomment to print the red car position on every iteration
        #print(sg.findPosition(redCarNode, "car"))

        # Drawing the Car

        car = sg.findNode(redCarNode, "chasis")
        
        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            carRotation = tr.rotationZ((np.pi/2)-camera_theta+0.2)
        
        elif (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            carRotation = tr.rotationZ((np.pi/2)-camera_theta-0.2)
                
        else:
            carRotation = tr.rotationZ((np.pi/2)-camera_theta)
            
        car.transform= tr.matmul([
            tr.translate(carX,carY,-4),
            tr.uniformScale(2),
            carRotation
            ])
        #redCarNode = p-c

        sg.drawSceneGraphNode(redCarNode, pipelineNormal, "model")
    
        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()


# In[ ]:





# In[ ]:




