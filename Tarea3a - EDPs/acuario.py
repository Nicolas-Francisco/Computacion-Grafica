#!/usr/bin/env python
# coding: utf-8

# In[37]:


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


def createAquarium():

    gpuBlackCube = es.toGPUShape(bs.createColorCube(0,0,0))
    
    gpuCube = es.toGPUShape(bs.createColorCube(0,1,1))
    
    
    # Creating the chasis of the car
    Cube1 = sg.SceneGraphNode("chasis1")
    Cube1.transform = tr.matmul([tr.translate(0, 1.5, 4), tr.scale(6, 0.05, 0.05)])
    Cube1.childs += [gpuBlackCube]
    
    Cube2 = sg.SceneGraphNode("chasis1")
    Cube2.transform = tr.matmul([tr.translate(0, -1.5, 4), tr.scale(6, 0.05, 0.05)])
    Cube2.childs += [gpuBlackCube]
    
    Cube3 = sg.SceneGraphNode("chasis1")
    Cube3.transform = tr.matmul([tr.translate(0, 1.5, 0), tr.scale(6, 0.05, 0.05)])
    Cube3.childs += [gpuBlackCube]
    
    Cube4 = sg.SceneGraphNode("chasis1")
    Cube4.transform = tr.matmul([tr.translate(0, -1.5, 0), tr.scale(6, 0.05, 0.05)])
    Cube4.childs += [gpuBlackCube]
    
    
    
    Cube5 = sg.SceneGraphNode("chasis1")
    Cube5.transform = tr.matmul([tr.translate(3, 1.5, 2), tr.scale(0.05, 0.05, 4)])
    Cube5.childs += [gpuBlackCube]
    
    Cube6 = sg.SceneGraphNode("chasis1")
    Cube6.transform = tr.matmul([tr.translate(-3, 1.5, 2), tr.scale(0.05, 0.05, 4)])
    Cube6.childs += [gpuBlackCube]
    
    Cube7 = sg.SceneGraphNode("chasis1")
    Cube7.transform = tr.matmul([tr.translate(3, -1.5, 2), tr.scale(0.05, 0.05, 4)])
    Cube7.childs += [gpuBlackCube]
    
    Cube8 = sg.SceneGraphNode("chasis1")
    Cube8.transform = tr.matmul([tr.translate(-3, -1.5, 2), tr.scale(0.05, 0.05, 4)])
    Cube8.childs += [gpuBlackCube]
    
    
    
    Cube9 = sg.SceneGraphNode("chasis1")
    Cube9.transform = tr.matmul([tr.translate(3, 0, 4), tr.scale(0.05, 3, 0.05)])
    Cube9.childs += [gpuBlackCube]
    
    Cube10 = sg.SceneGraphNode("chasis1")
    Cube10.transform = tr.matmul([tr.translate(-3, 0, 4), tr.scale(0.05, 3, 0.05)])
    Cube10.childs += [gpuBlackCube]
    
    Cube11 = sg.SceneGraphNode("chasis1")
    Cube11.transform = tr.matmul([tr.translate(3, 0, 0), tr.scale(0.05, 3, 0.05)])
    Cube11.childs += [gpuBlackCube]
    
    Cube12 = sg.SceneGraphNode("chasis1")
    Cube12.transform = tr.matmul([tr.translate(-3, 0, 0), tr.scale(0.05, 3, 0.05)])
    Cube12.childs += [gpuBlackCube]
    
    
    Cube13 = sg.SceneGraphNode("chasis1")
    Cube13.transform = tr.matmul([tr.translate(0, 0, 0), tr.scale(6, 3, 0.05)])
    Cube13.childs += [gpuCube]
    
    

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
    car.childs += [Cube11]
    car.childs += [Cube12]
    
    car.childs += [Cube13]

    return car


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "3D Aquarium", None, None)

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
    glClearColor(0.08, 0.4, 1, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    gpuAxis = es.toGPUShape(bs.createAxis(7))
    AquariumNode = createAquarium()


    # Using the same view and projection matrices in the whole application
    projection = tr.perspective(45, float(width)/float(height), 0.1, 100)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    
    view = tr.lookAt(
            np.array([10,10,14]),
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

        # Drawing the Car
        sg.drawSceneGraphNode(AquariumNode, mvpPipeline, "model")

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)
    
    glfw.terminate()


# In[ ]:





# In[ ]:




