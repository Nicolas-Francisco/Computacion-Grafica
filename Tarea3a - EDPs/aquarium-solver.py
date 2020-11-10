#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import sys
import json
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt

archivo = sys.argv[1]
Data = None
with open(archivo) as file:
    data = json.load(file)
    Data = data

H = Data["height"]   # Height
W = Data["width"]    # Width
L = Data["lenght"]   # Length

F = Data["window_loss"]  # Condicion de Newman
h = 0.2 # Steps

heater_a = Data["heater_a"] # Calentador a
heater_b = Data["heater_a"] # Calentador b
T = Data["ambient_temperature"]  # Temperatura

# Number of unknowns
# only the top side and the heaters at the bottom are known (Dirichlet condition)
# right, left, front, and backside are unknown (Neumann condition)
nx = int(W / h) + 1
ny = int(L / h) + 1
nk = int(H / h)

# In this case, the domain is an aquarium with parallelepiped form
N = nx * ny * nk

# We define a function to convert the indices from i,j,k to K and viceversa
# i,j,k indexes the discrete domain in 3D.
# K parametrize those i,j,k this way we can tidy the unknowns
# in a column vector and use the standard algebra

def getG(i,j,k):
    return  i+j * nx +k*nx*ny

def getIJK(g):
    i = (g %( nx*ny))%nx
    j = (g %( nx*ny))//nx
    k = g // (nx*ny)
    return (i, j, k)

# In this matrix we will write all the coefficients of the unknowns
#A = np.zeros((N,N))
A = sparse.lil_matrix((N,N)) # We use a sparse matrix in order to spare memory, since it has many 0's

# In this vector we will write all the right side of the equations
b = np.zeros((N,))

# Note: To write an equation is equivalent to write a row in the matrix system

# We iterate over each point inside the domain
# Each point has an equation associated
# The equation is different depending on the point location inside the domain
for k in range(0, nk):
    for j in range(0, ny):
        for i in range(0, nx):
            # We will write the equation associated with row K
            g = getG(i, j, k)
            # We obtain indices of the other coefficients
            g_right = getG(i+1, j, k)
            g_left = getG(i-1, j, k)
            g_front = getG(i, j+1, k)
            g_back = getG(i, j-1, k)
            g_up = getG(i, j, k+1)
            g_down = getG(i, j, k-1)
            
            # Depending on the location of the point, the equation is different
            # Interior
            if (1 <= i) and (i <= nx - 2) and (1 <= j) and (j <= ny - 2) and (1 <= k) and (k <= nk-2):
                #print("(",str(i)," ",str(j), " ",str(k),")"," ",str(1))
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = 0

            #################### Faces
     
            # right face
            elif i == nx-1 and (1 <= j) and (j <= ny - 2) and (1 <= k) and (k <= nk-2):
                A[g, g_left] = 2
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F

            # left face
            elif i == 0 and (1 <= j) and (j <= ny - 2) and (1 <= k) and (k <= nk-2):
                A[g, g_right] = 2
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F
            
            # front face
            elif (1 <= i) and (i <= nx-2) and j == ny-1 and (1 <= k) and (k <= nk-2):
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_back] = 2
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F

            # back face
            elif (1 <= i) and (i <= nx-2) and j == 0 and (1 <= k) and (k <= nk-2):
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 2
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F
    
            # upper face
            elif (1 <= i) and (i <= nx - 2) and (1 <= j) and (j <= ny-2) and k == nk-1:
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -T

            # heater A
            elif (nx//3 <= i) and (i <= 2*nx//3) and (ny-(2*ny//5) <= j) and (j <= ny-(ny//5)) and k == 0:
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 1
                A[g, g] = -6
                b[g] =  -heater_a

            # heater B
            elif (nx//3 <= i) and (i <= 2*nx//3) and (ny//5 <= j) and (j <= 2*ny//5) and k == 0:
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 1
                A[g, g] = -6
                b[g] =  -heater_b

            # bottom face
            elif (1 <= i) and (i <= nx - 2) and (1 <= j) and (j <= ny-2) and k == 0:
                #print("(",str(i)," ",str(j), " ",str(k),")"," ",str(7))
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = 0


            ################### Corners

            # right front up
            
            elif i == nx-1 and j == ny-1 and k == nk-1:
                A[g, g_left] = 2
                A[g, g_back] = 2
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F -T
            
            # left front up
            elif i == 0 and j == ny-1 and k == nk-1:
                A[g, g_right] = 2
                A[g, g_back] = 2
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F -T
            
            # right back up
            elif i == nx-1 and j == 0 and k == nk-1:
                A[g, g_left] = 2
                A[g, g_front] = 2
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F -T
            
            # left back up
            elif i == 0 and j == 0 and k == nk-1:
                A[g, g_right] = 2
                A[g, g_front] = 2
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F -T

            # front right down
            elif (i, j, k) == (nx-1,ny-1,0):
                A[g, g_left] = 2
                A[g, g_back] = 2
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -4 * h * F

            #  front left down
            elif (i, j, k) == (0,ny-1,0):
                A[g, g_right] = 2
                A[g, g_back] = 2
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -4 * h * F

            # back right down
            elif (i, j, k) == (nx-1,0,0):
                A[g, g_left] = 2
                A[g, g_front] = 2
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -4 * h * F

            # back left down
            elif (i, j, k) == (0,0,0):
                A[g, g_right] = 2
                A[g, g_front] = 2
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -4 * h * F



            ############3 Aristas

            # Cara derecha, abajo
            elif i == nx-1 and (1 <= j) and (j <= ny - 2) and k == 0:
                A[g, g_left] = 2
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -2 * h * F

            # Cara izquierda, abajo
            elif i == 0 and (1 <= j) and (j <= ny - 2) and k == 0:
                A[g, g_right] = 2
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -2 * h * F

            # Cara frontal, abajo
            elif (1 <= i) and (i <= nx-2) and j == ny-1 and k == 0:
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_back] = 2
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -2 * h * F

            # Cara trasera, abajo
            elif (1 <= i) and (i <= nx-2) and j == 0 and k == 0:
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 2
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -2 * h * F

            # Cara frontal, derecha
            elif i == nx-1 and j == ny-1 and (1 <= k) and (k <= nk-2):
                A[g, g_left] = 2
                A[g, g_back] = 2
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F
            
            # Cara frontal, izquierda
            elif i == 0 and j == ny-1 and (1 <= k) and (k <= nk-2):
                A[g, g_right] = 2
                A[g, g_back] = 2
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F
            
            # Cara trasera, derecha
            elif i == nx-1 and j == 0 and (1 <= k) and (k <= nk-2):
                A[g, g_left] = 2
                A[g, g_front] = 2
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F
            
            # Cara trasera, izquierda
            elif i == 0 and j == 0 and (1 <= k) and (k <= nk-2):
                A[g, g_right] = 2
                A[g, g_front] = 2
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F
            
            # Cara superior, derecha
            elif i == nx-1 and (1 <= j) and (j <= ny - 2) and k == nk-1:
                A[g, g_left] = 2
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F -T

            # Cara superior, izquierda
            elif i == 0 and (1 <= j) and (j <= ny - 2) and k == nk-1:
                A[g, g_right] = 2
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F -T
            
            # Cara superior, abajo
            elif (1 <= i) and (i <= nx-2) and j == ny-1 and k == nk-1:
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_back] = 2
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F -T

            # Cara superior, arriba
            elif (1 <= i) and (i <= nx-2) and j == 0 and k == nk-1:
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 2
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F -T
                
            else:
                print("Point (" + str(i) + ", " + str(j) + ", " + str(k) + ") missed!")
                print("Associated point index is " + str(K))
                raise Exception()


# A quick view of a sparse matrix
#plt.spy(A)

# Solving our system
#x = np.linalg.solve(A, b)
x = linalg.spsolve(A, b)

# Now we return our solution to the 3d discrete domain
# In this matrix we will store the solution in the 3d domain
u = np.zeros((nx, ny, nk))

for g in range(0, N):
    i, j, k = getIJK(g)
    u[i, j, k] = x[g]

# Adding the borders, as they have known values
ub = np.zeros((nx,ny,nk+1))
ub[0:nx, 0:ny, 0:nk] = u[:,:,:]

# Dirichlet boundary condition on the top side
ub[0:nx, 0:ny, nk] = T

print(ub)

# Saving results for temperatures
file = Data["filename"]
np.save(file, ub)

#h = 0.5 -> 7,13,9
#h = 0.2 -> 16,31,21
#h = 0.1 -> 41,71,51
Y, X, Z = np.meshgrid(np.linspace(0, L, ny), np.linspace(0, W, nx), np.linspace(0, H, nk + 1))

fig = plt.figure()
ax = fig.gca(projection='3d')

scat = ax.scatter(X,Y,Z, c=ub, alpha=0.5, s=100, marker='s')

fig.colorbar(scat, shrink=0.5, aspect=5) # This is the colorbar at the side

# Showing the result
ax.set_title('Laplace equation solution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Note:
# imshow is also valid but it uses another coordinate system,
# a data transformation is required
#ax.imshow(ub.T)


plt.show()
