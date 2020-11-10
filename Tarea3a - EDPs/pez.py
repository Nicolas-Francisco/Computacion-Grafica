#!/usr/bin/env python
# coding: utf-8

# In[14]:


import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys

import math
import transformations as tr
import basic_shapes as bs
import scene_graph as sg
import easy_shaders as es


# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True


# we will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_LEFT_CONTROL:
        controller.showAxis = not controller.showAxis

    elif key == glfw.KEY_ESCAPE:
        sys.exit()

    else:
        print('Unknown key')


def createFish(color1, color2):

    gpuBlackCube = es.toGPUShape(bs.createColorCube(0,0,0))
    gpuCube1 = es.toGPUShape(bs.createColorCube(color1[0], color1[1], color1[2]))
    gpuCube2 = es.toGPUShape(bs.createColorCube(color2[0], color2[1], color2[2]))
    
    
    # Cheating a single wheel
    aleta1 = sg.SceneGraphNode("wheel")
    aleta1.transform = tr.matmul([tr.translate(0.2, 0.4, 0), tr.scale(0.4, 0.15, 0.05), tr.rotationZ(-np.pi/6)])
    aleta1.childs += [gpuCube2]
    
    aleta2 = sg.SceneGraphNode("wheel")
    aleta2.transform = tr.matmul([tr.translate(0.1, 0.4, 0), tr.scale(0.4, 0.15, 0.05), tr.rotationZ(np.pi/6)])
    aleta2.childs += [gpuCube2]
    
    aleta3 = sg.SceneGraphNode("wheel")
    aleta3.transform = tr.matmul([tr.translate(0.2, -0.4, 0), tr.scale(0.4, 0.15, 0.05), tr.rotationZ(np.pi/6)])
    aleta3.childs += [gpuCube2]
    
    aleta4 = sg.SceneGraphNode("wheel")
    aleta4.transform = tr.matmul([tr.translate(0.1, -0.4, 0), tr.scale(0.4, 0.15, 0.05), tr.rotationZ(-np.pi/6)])
    aleta4.childs += [gpuCube2]
    
    aleta5 = sg.SceneGraphNode("wheel")
    aleta5.transform = tr.matmul([tr.translate(-1.15, 0, 0.1), tr.scale(0.15, 0.05, 0.4), tr.rotationY(np.pi/6)])
    aleta5.childs += [gpuCube2]
    
    aleta6 = sg.SceneGraphNode("wheel")
    aleta6.transform = tr.matmul([tr.translate(-1.15, 0, -0.1), tr.scale(0.15, 0.05, 0.4), tr.rotationY(-np.pi/6)])
    aleta6.childs += [gpuCube2]
    
    
    
    # Creating the chasis of the car
    Cube1 = sg.SceneGraphNode("chasis1")
    Cube1.transform = tr.matmul([tr.translate(0, 0, 0), tr.scale(0.4, 0.4, 1)])
    Cube1.childs += [gpuCube1]
    
    Cube2 = sg.SceneGraphNode("chasis2")
    Cube2.transform = tr.matmul([tr.translate(0.1, 0, 0), tr.scale(0.4, 0.35, 0.9)])
    Cube2.childs += [gpuCube2]
    
    Cube3 = sg.SceneGraphNode("chasis2")
    Cube3.transform = tr.matmul([tr.translate(0.2, 0, 0), tr.scale(0.4, 0.3, 0.8)])
    Cube3.childs += [gpuCube1]
    
    Cube4 = sg.SceneGraphNode("chasis2")
    Cube4.transform = tr.matmul([tr.translate(-0.2, 0, 0), tr.scale(0.4, 0.35, 0.8)])
    Cube4.childs += [gpuCube2]
    
    Cube5 = sg.SceneGraphNode("chasis2")
    Cube5.transform = tr.matmul([tr.translate(-0.4, 0, 0), tr.scale(0.4, 0.30, 0.6)])
    Cube5.childs += [gpuCube1]
    
    Cube6 = sg.SceneGraphNode("chasis2")
    Cube6.transform = tr.matmul([tr.translate(-0.6, 0, 0), tr.scale(0.4, 0.25, 0.4)])
    Cube6.childs += [gpuCube2]
    
    Cube7 = sg.SceneGraphNode("chasis2")
    Cube7.transform = tr.matmul([tr.translate(0.3, 0, 0), tr.scale(0.4, 0.3, 0.7)])
    Cube7.childs += [gpuCube2]
    
    Cube8 = sg.SceneGraphNode("chasis2")
    Cube8.transform = tr.matmul([tr.translate(-0.8, 0, 0), tr.scale(0.4, 0.2, 0.2)])
    Cube8.childs += [gpuCube1]
    
    Cube9 = sg.SceneGraphNode("chasis2")
    Cube9.transform = tr.matmul([tr.translate(0.4, 0, 0), tr.scale(0.4, 0.25, 0.6)])
    Cube9.childs += [gpuCube1]
    
    Cube10 = sg.SceneGraphNode("chasis2")
    Cube10.transform = tr.matmul([tr.translate(0.35, 0, 0.29), tr.scale(0.09, 0.41, 0.2)])
    Cube10.childs += [gpuBlackCube]

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
    car.childs += [Cube9]
    car.childs += [Cube10]
    car.childs += [aleta1]
    car.childs += [aleta2]
    car.childs += [aleta3]
    car.childs += [aleta4]
    car.childs += [aleta5]
    car.childs += [aleta6]

    return car


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "3D fish", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    
    # Telling OpenGL to use our shader program
    glUseProgram(mvpPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    gpuAxis = es.toGPUShape(bs.createAxis(7))
    redFishNode = createFish([1,0,0], [1,0.5,0.5])


    # Using the same view and projection matrices in the whole application
    projection = tr.perspective(45, float(width)/float(height), 0.1, 100)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    
    view = tr.lookAt(
            np.array([5,5,7]),
            np.array([0,0,0]),
            np.array([0,0,1])
        )
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if controller.showAxis:
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            mvpPipeline.drawShape(gpuAxis, GL_LINES)

        # Moving the red car and rotating its wheels

        # Uncomment to print the red car position on every iteration
        #print(sg.findPosition(redCarNode, "car"))
        
        redFishNode.transform = np.matmul(tr.rotationZ(1 * glfw.get_time()), tr.translate(0,0,0))

        # Drawing the Car
        sg.drawSceneGraphNode(redFishNode, mvpPipeline, "model")

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    
    glfw.terminate()


# In[ ]:





# In[ ]:




