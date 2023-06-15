"""
Finite element code to model a composite laminate plate. 
Based on First Order Shear Deformation Theory (FSDT)
"""


import enum
import imp
from re import L
from tkinter import W
from xml.etree import ElementInclude
import numpy as np
import matplotlib.pyplot as plt
from ufl import det 
from scipy.sparse.linalg.dsolve import linsolve
import scipy.sparse.linalg
import scipy.linalg 
import plotly.graph_objects as go


class Mesh: 
    def __init__(self, lx, ly, nx, ny):
        self.lx = lx                # Length of the computational domain in x direction 
        self.ly = ly                # Length of the computational domain in y direction 
        self.nx = nx                # Number of divisions in x direction 
        self.ny = ny                # Number of divisions in y direction 
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.nel = self.nx * self.ny          # Total number of elements in the computational domain 
        self.nnodes = (self.nx + 1) * (self.ny + 1)       # Total number of nodes in the computational domain

        self.nodes = [Node(int(i)) for i in range(self.nnodes)]

        # Assigning properties to the nodes
        for node in self.nodes:
            node.setCoords( self.dx*(node.n % (nx+1)), self.dy*(int(node.n / (nx+1))))
            
            # Assigning neighbors
            if ( (node.n % (nx+1) == 0) and (node.n == 0) ):
                node.setType("SWCorner")
                node.appendNeighbor( node.n + 1 )
                node.appendNeighbor( node.n + self.nx + 1 )
            elif ( (node.n % (nx+1) == 0) and (node.n == (self.nnodes - 1 - self.nx)) ):
                node.setType("NWCorner")
                node.appendNeighbor( node.n + 1 )
                node.appendNeighbor( node.n - self.nx - 1 )
            elif ( (node.n % (nx+1) == nx) and (node.n == nx) ):
                node.setType("SECorner")
                node.appendNeighbor( node.n - 1)
                node.appendNeighbor( node.n + self.nx + 1 )
            elif ( (node.n % (nx+1) == nx) and (node.n == self.nnodes - 1) ):
                node.setType("NECorner")
                node.appendNeighbor( node.n - 1)
                node.appendNeighbor( node.n - self.nx - 1 ) 
            elif ( node.n % (nx+1) == 0 ):
                node.setType("WWall")
                node.appendNeighbor( node.n + 1)
                node.appendNeighbor( node.n - self.nx - 1)
                node.appendNeighbor( node.n + self.nx + 1)
            elif ( node.n % (nx+1) == nx ):
                node.setType("EWall")
                node.appendNeighbor( node.n - 1 )
                node.appendNeighbor( node.n - self.nx - 1 )
                node.appendNeighbor( node.n + self.nx + 1 )
            elif ( node.n > (self.nnodes - self.nx - 1)):
                node.setType("NWall")
                node.appendNeighbor( node.n - 1 )
                node.appendNeighbor( node.n + 1 )
                node.appendNeighbor( node.n - self.nx - 1 )
            elif ( node.n < ( self.nx + 1 )):
                node.setType("SWall")
                node.appendNeighbor( node.n - 1 )
                node.appendNeighbor( node.n + 1 )
                node.appendNeighbor( node.n + self.nx + 1 )
            else: 
                node.setType("Internal")
                node.appendNeighbor( node.n + 1 )
                node.appendNeighbor( node.n - 1 )
                node.appendNeighbor( node.n + self.nx + 1 )
                node.appendNeighbor( node.n - self.nx - 1 )
            
        self.elements = [Element(int(i)) for i in range(self.nel)]

        # Assigning properties to the elements 
        for element in self.elements:
            a = int((element.n) % self.nx)       # Element count along the x direction in the domain
            b = int((element.n) / self.nx)       # Element count along the y direction in the domain
            
            # Appending the nodes to the element
            # The nodes are appended in a clockwise manner starting from the NE 
            # corner so that it is easier to index this list when required to 
            # construct the global stiffness matrix...or any global matrix for 
            # that matter
            element.appendNode( int(a + (b+1) * (nx + 1) + 1 ))
            element.appendNode( int(a + (b+1) * (nx + 1)))
            element.appendNode( int(a + b * (nx + 1)))
            element.appendNode( int(a + b * (nx + 1) + 1 ))

    def getElementNodes(self, elnum):
        """ Function returns the nodes that belong to a given element given the element number """
        return self.elements[elnum].nodes
    
    def getNodeCoords(self, elnum, nnum):
        """ Function that returns the coordinates of a specified nodes as a list. nnum is numbered starting from 0 """
        # Getting the actual node number from the mesh 
        NNUM = int(self.elements[elnum].nodes[nnum])
        return [self.nodes[NNUM].x, self.nodes[NNUM].y]

    def getBoundaries(self):
        """ Function that returns a list of all the bondary nodes """

        boundNodes = []

        for node in self.nodes:
            if ( node.type == "SWCorner" ):
                boundNodes.append( node.n )
            elif ( node.type == "SECorner" ):
                boundNodes.append( node. n)
            elif ( node.type == "NECorner" ):
                boundNodes.append( node. n)
            elif ( node.type == "NWCorner" ):
                boundNodes.append( node. n)
            elif ( node.type == "NWall" ):
                boundNodes.append( node. n)
            elif ( node.type == "SWall" ):
                boundNodes.append( node. n)
            elif ( node.type == "WWall" ):
                boundNodes.append( node. n)
            elif ( node.type == "EWall" ):
                boundNodes.append( node. n)
            
        return boundNodes
    

            
                


class Node: 
    def __init__(self,n):
        self.x = 0             # Location in the x direction 
        self.y = 0             # Location in the y direction 
        self.n = n             # Node number 
        self.neighbors = []         # Empty list for the node numbers 
        self.type = ""          # Whether the node is a corner node, a boundary node or an internal node
    
    def setCoords(self, x, y):
        """ Function sets the coordinates for a given node """
        self.x = x
        self.y = y 

    def appendNeighbor(self, nei):
        """ Neighboring nodes are appended to the neighbors list """
        self.neighbors.append(nei)
    
    def setType(self, type):
        """ Function that sets the type of the node """
        self.type = type



class Element: 
    def __init__(self,n):
        self.n = n              # Number of the element
        self.nodes = []         # List of node numbers that belong to the element

    def appendNode(self, nnum):
        """ Function that appends the nodes that belong to this element """
        self.nodes.append(nnum)
    


def matB(s, t, c1, c2, c3, c4): 
    """
        Function that takes the value of s and t and returns the value of [B]
    
        This function is to be called multiple times during the calculation and 
        construction of the stiffness matrix. 
        This B Matrix corresponds to that for a Q4 element specifically implemented 
        to handle FSDT theory. 
    """

    #   Assembling [B]
    a = 0.25 * (c1[1]*(s-1) + c2[1]*(-1-s) + c3[1]*(1+s) + c4[1]*(1-s))
    b = 0.25 * (c1[1]*(t-1) + c2[1]*(1-t) + c3[1]*(1+t) + c4[1]*(-1-t))
    c = 0.25 * (c1[0]*(t-1) + c2[0]*(1-t) + c3[0]*(1+t) + c4[0]*(-1-t))
    d = 0.25 * (c1[0]*(s-1) + c2[0]*(-1-s) + c3[0]*(1+s) + c4[0]*(1-s))

    N1s = 0.25 * (t - 1)
    N1t = 0.25 * (s - 1)
    N2s = 0.25 * (1 - t)
    N2t = -0.25 * (1 + s)
    N3s = 0.25 * (1 + t)
    N3t = 0.25 * (1 + s)
    N4s = -0.25 * (1 + t)
    N4t = 0.25 * (1 - s)

    Ns = np.array([N1s, N2s, N3s, N4s])
    Nt = np.array([N1t, N2t, N3t, N4t])
    
    B = np.zeros((8,20))

    for i in range(4):
        ddx = a*Ns[i] - b*Nt[i]
        ddy = c*Nt[i] - d*Ns[i]

        B[0, i*5 + 0] = ddx
        B[1, i*5 + 1] = ddy
        B[2, i*5 + 0] = ddy
        B[2, i*5 + 1] = ddx
        B[3, i*5 + 3] = ddx
        B[4, i*5 + 4] = ddy
        B[5, i*5 + 3] = ddy
        B[5, i*5 + 4] = ddx
        B[6, i*5 + 2] = ddx
        B[6, i*5 + 3] = 1
        B[7, i*5 + 2] = ddy
        B[7, i*5 + 4] = 1

    J = detJ(s, t, c1, c2, c3, c4)
    B = B/J
        
    return B



def detJ(s, t, c1, c2, c3, c4):
    """
        Function that returns the determinant 
        of the Jacobian 
        
        Args: 
            s: 
            t: natural coordinates 
            cn: Coordinates of the nodes of the element
        
        Returns: 
            detJ: Determinant of the jacobian 
    """

    xc = np.array([c1[0], c2[0], c3[0], c4[0]])
    yc = np.array([c1[1], c2[1], c3[1], c4[1]])

    matst = np.zeros((4,4))
    matst[0,1] = 1 - t
    matst[0,2] = t - s
    matst[0,3] = s - 1
    matst[1,2] = s + 1
    matst[1,3] = - (t + s)
    matst[2,3] = t + 1

    matst[1,0] = -matst[0,1]
    matst[2,0] = -matst[0,2]
    matst[3,0] = -matst[0,3]
    matst[2,1] = -matst[1,2]
    matst[3,1] = -matst[1,3]
    matst[3,2] = -matst[2,3]

    detJ = 0.125 * np.matmul(np.matmul(xc.T, matst), yc)

    if ( detJ == 0 ):
        detJ = 1

    return detJ



def matKe(elemNum, H, c1, c2, c3, c4, h): 
    """
        Function that calculates the element stiffness matrix 
        
        Args: 
            elemNum:  The element number 
            H:  The matrix of material properties 
            c1: List of coordinates x and y of node 0 of the element 'elnum'
            c2: List of coordinates x and y of node 1 of the element 'elnum'
            c3: List of coordinates x and y of node 2 of the element 'elnum'
            c4: List of coordinates x and y of node 3 of the element 'elnum'
            h: plate thickness, rather, this is the sum of the thickness of the plates
            
        Returns: 
            Ke: The element stiffness
    """

    # Weights and Locations of Gaussian Points 
    #       Locations in the natural coordinate system
    sGP = np.array([0.7745966, 0.7745966, 0.0, -0.7745966, -0.7745966, -0.7745966, 0.0, 0.7745966])
    tGP = np.array([0.0, 0.7745966, 0.7745966, 0.7745966, 0.0, -0.7745966, -0.7745966, -0.7745966])
    #       Weights of the Gauss Points
    Ws = np.array([5/9, 5/9, 8/9, 5/9, 5/9, 5/9, 8/9, 5/9])
    Wt = np.array([8/9, 5/9, 5/9, 5/9, 8/9, 5/9, 5/9, 5/9])
    #       Number of Gauss Points
    ngp = 8                                        

    #  Element Stiffness Matrix
    Ke = np.zeros((20,20))

    for i in range(ngp):
        B = matB(sGP[i], tGP[i], c1, c2, c3, c4) 
        J = detJ(sGP[i], tGP[i], c1, c2, c3, c4)

        Ke += h * J * np.matmul(B.T, np.matmul(H, B)) * Ws[i] * Wt[i]
    
    return Ke



def shapeFunction(s,t):
    """ Function that returns the shape function for a single element that is meant for the 
        the calculations when applying boundary conditions. 
        
        Args: 
            s: element / natural coordinate in x 
            t: element / natural coordinate in y 
        Returns: 
            N: Shape function for a single Q4 element
    """

    N = np.zeros((5,20))

    N1 = 0.25 * (1 - s) * (1 - t)
    N2 = 0.25 * (1 + s) * (1 - t)
    N3 = 0.25 * (1 + s) * (1 + t)
    N4 = 0.25 * (1 - s) * (1 + t)

    for i in range(5):
        N[i,i] = N1
        N[i,5+i] = N2
        N[i,10+i] = N3
        N[i,15+i] = N4

    return N



def ElementBodyForce(bf, c1, c2, c3, c4, h):
    """ Function that returns the body force vector for all the nodes of a single element given 
        the body force applied on the body. 
        
        Args: 
            bf:     Body force values (user defined values). This needs to be spread over entire element.
            c1:
            c2:
            c3:
            c4:
            h:      Thickness of the laminate
        Returns: 
            BF:     Body force to be applied at each node
    """
    # Weights and Locations of Gaussian Points 
    #       Locations in the natural coordinate system
    sGP = np.array([0.7745966, 0.7745966, 0.0, -0.7745966, -0.7745966, -0.7745966, 0.0, 0.7745966])
    tGP = np.array([0.0, 0.7745966, 0.7745966, 0.7745966, 0.0, -0.7745966, -0.7745966, -0.7745966])
    #       Weights of the Gauss Points
    Ws = np.array([5/9, 5/9, 8/9, 5/9, 5/9, 5/9, 8/9, 5/9])
    Wt = np.array([8/9, 5/9, 5/9, 5/9, 8/9, 5/9, 5/9, 5/9])
    #       Number of Gauss Points
    ngp = 8                                        

    # Creating body force vector
    BF = np.zeros(20)

    for i in range(ngp):
        N = shapeFunction(sGP[i], tGP[i])
        J = detJ(sGP[i], tGP[i], c1, c2, c3, c4)

        BF += np.matmul(N.T, bf) * J * h * Ws[i] * Wt[i]
    
    # print("[DEBUG]\tBody force vector for this element: ", BF)

    return BF



def Q4_composite_laminate_plate():
    """

    """

    # Material Properties of Boron / Epoxy
    import numpy as np 

    E1 = 210E09
    E2 = 19E09
    E3 = 19E09
    G12 = 4.8E09
    G23 = 4.8E09
    G31 = 4.8E09
    nu12 = 0.25
    nu23 = 0.25
    nu31 = 0.25

    # Laminate Properties 
    # nPlys = 5                                   # Number of Plys 
    # plyAngles = np.array([0, 30, 60, 90, 120])            # Angular Orientation of the plys 
    # nPlys = 2
    # plyAngles = np.array([0, 90])
    # plyAngles = np.pi / 180 * plyAngles
    # # plyHeights = np.array([0, 0.001, 0.002, 0.003, 0.004, 0.005])
    # plyHeights = np.array([0, 0.001, 0.02])
    nPlys = 16
    plyAngles = np.array([45,-45,45,-45,-45,90,45,0,0,45,90,-45,-45,45,-45,45])
    plyAngles = np.pi / 180 * plyAngles
    plyHeights = np.linspace(0, 0.025, num=17)

    # Calculation of the (8x8) Material Property Matrix 
    C = np.zeros((6,6))                             # Creating the constitutive matrix
    Q = np.zeros((nPlys,6,6))                       # Creating the constitutive matrix based on ply orientation
    A = np.zeros((3,3))                             # Extensional Matrix
    B = np.zeros((3,3))                             # Extensional Bending Matrix 
    D = np.zeros((3,3))                             # Bending Matrix 
    As = np.zeros((2,2))                            # Some shearing matrix. It is calculated exactly like A 
    H = np.zeros((8,8))
    
    #   Filling the constitutive matrix
    C[0,0] = E1 / (1 - nu12 * nu12)
    C[0,1] = nu12 * E2 / (1 - nu12 * nu12)
    C[1,1] = E2 / (1 - nu12 * nu12)
    C[5,5] = G12

    #   Filling the constitutive matrix based on ply orientation 
    U1 = 1/8 * (3*C[0,0] + 3*C[1,1] + 2*C[0,1] + 4*C[5,5])
    U2 = 1/2 * (C[0,0] - C[1,1])
    U3 = 1/8 * (C[0,0] + C[1,1] - 2*C[0,1] - 4*C[5,5])
    U4 = 1/8 * (C[0,0] + C[1,1] + 6*C[0,1] - 4*C[5,5])
    U5 = 1/8 * (C[0,0] + C[1,1] + 4*C[5,5] - 2*C[0,1])

    Q[:,0,0] = U1 + U2*np.cos(2*plyAngles[:]) + U3*np.cos(4*plyAngles[:])
    Q[:,0,1] = U4 - U3*np.cos(4*plyAngles[:])
    Q[:,1,1] = U1 - U2*np.cos(2*plyAngles[:]) + U3*np.cos(4*plyAngles[:])
    Q[:,0,5] = 0.5*U2*np.sin(2*plyAngles[:]) + U3*np.sin(4*plyAngles[:])
    Q[:,1,5] = 0.5*U2*np.sin(2*plyAngles[:]) - U3*np.sin(4*plyAngles[:])
    Q[:,3,3] = G23*np.cos(plyAngles[:])*np.cos(plyAngles[:]) + G31*np.sin(plyAngles[:])*np.sin(plyAngles[:])
    Q[:,4,4] = G23*np.sin(plyAngles[:])*np.sin(plyAngles[:]) + G31*np.cos(plyAngles[:])*np.cos(plyAngles[:]) 
    Q[:,3,4] = (G31 - G23) * np.cos(plyAngles[:]) * np.sin(plyAngles[:])
    
    #   Filling the extensional matrix 
    A[0,0] = np.sum(Q[:,0,0] * (plyHeights[1:] - plyHeights[:-1]))
    A[0,1] = np.sum(Q[:,0,1] * (plyHeights[1:] - plyHeights[:-1]))
    A[1,1] = np.sum(Q[:,1,1] * (plyHeights[1:] - plyHeights[:-1]))
    A[0,2] = np.sum(Q[:,0,5] * (plyHeights[1:] - plyHeights[:-1]))
    A[1,2] = np.sum(Q[:,2,5] * (plyHeights[1:] - plyHeights[:-1]))
    A[2,2] = np.sum(Q[:,5,5] * (plyHeights[1:] - plyHeights[:-1]))
    A[1,0] = A[0,1]
    A[2,0] = A[0,2]
    A[2,1] = A[1,2]

    #   Filling the extensional-bending matrix 
    B[0,0] = np.sum(Q[:,0,0] * (np.power(plyHeights[1:],2) - np.power(plyHeights[:-1],2))) * 1/2
    B[0,1] = np.sum(Q[:,0,1] * (np.power(plyHeights[1:],2) - np.power(plyHeights[:-1],2))) * 1/2
    B[1,1] = np.sum(Q[:,1,1] * (np.power(plyHeights[1:],2) - np.power(plyHeights[:-1],2))) * 1/2
    B[0,2] = np.sum(Q[:,0,5] * (np.power(plyHeights[1:],2) - np.power(plyHeights[:-1],2))) * 1/2
    B[1,2] = np.sum(Q[:,2,5] * (np.power(plyHeights[1:],2) - np.power(plyHeights[:-1],2))) * 1/2
    B[2,2] = np.sum(Q[:,5,5] * (np.power(plyHeights[1:],2) - np.power(plyHeights[:-1],2))) * 1/2
    B[1,0] = B[0,1]
    B[2,0] = B[0,2]
    B[2,1] = B[1,2]

    #   Filling the bending matrix 
    D[0,0] = np.sum(Q[:,0,0] * (np.power(plyHeights[1:],3) - np.power(plyHeights[:-1],3))) * 1/3
    D[0,1] = np.sum(Q[:,0,1] * (np.power(plyHeights[1:],3) - np.power(plyHeights[:-1],3))) * 1/3
    D[1,1] = np.sum(Q[:,1,1] * (np.power(plyHeights[1:],3) - np.power(plyHeights[:-1],3))) * 1/3
    D[0,2] = np.sum(Q[:,0,5] * (np.power(plyHeights[1:],3) - np.power(plyHeights[:-1],3))) * 1/3
    D[1,2] = np.sum(Q[:,2,5] * (np.power(plyHeights[1:],3) - np.power(plyHeights[:-1],3))) * 1/3
    D[2,2] = np.sum(Q[:,5,5] * (np.power(plyHeights[1:],3) - np.power(plyHeights[:-1],3))) * 1/3
    D[1,0] = D[0,1]
    D[2,0] = D[0,2]
    D[2,1] = D[1,2]
    
    #   Filling the extensional matrix 
    As[0,0] = np.sum(Q[:,3,3] * (plyHeights[1:] - plyHeights[:-1]))
    As[0,1] = np.sum(Q[:,3,4] * (plyHeights[1:] - plyHeights[:-1]))
    As[1,1] = np.sum(Q[:,4,4] * (plyHeights[1:] - plyHeights[:-1]))
    As[1,0] = As[0,1]

    #   Filling the constitutive matrix. Calling it that because I do not know what else to call it 
    H[0,0] = A[0,0]
    H[0,1] = A[0,1]
    H[1,0] = A[1,0]
    H[1,1] = A[1,1]
    H[1,2] = A[1,2]
    H[0,2] = A[0,2]
    H[2,2] = A[2,2]
    H[2,1] = A[2,1]
    H[2,0] = A[2,0]

    H[0,3] = B[0,0]
    H[0,4] = B[0,1]
    H[0,5] = B[0,2]
    H[1,4] = B[1,1]
    H[1,5] = B[1,2]
    H[2,5] = B[2,2]
    H[1,3] = H[0,4]
    H[2,3] = H[0,5]
    H[2,4] = H[1,5]

    H[3,0] = B[0,0] 
    H[3,1] = B[0,1] 
    H[3,2] = B[0,2]
    H[4,1] = B[1,1]
    H[4,2] = B[1,2]
    H[5,2] = B[2,2]
    H[4,0] = H[3,1]
    H[5,0] = H[3,2]
    H[5,1] = H[4,2]
    
    H[3,3] = D[0,0]  
    H[3,4] = D[0,1]
    H[4,3] = D[1,0]
    H[4,4] = D[1,1]
    H[4,5] = D[1,2]
    H[3,5] = D[0,2]
    H[5,5] = D[2,2]
    H[5,4] = D[2,1]
    H[5,3] = D[2,0]

    H[6,6] = As[0,0]
    H[7,7] = As[1,1]
    H[6,7] = As[0,1]
    H[7,6] = H[6,7]

    np.savetxt("constitutiveRelation.csv", H, delimiter=',')

    print("\n[LOG]    Material properties have been used to construct the constitutive relation.")

    ############################################ MESH CREATION
    # Creation of the mesh 
    #       Mesh Properties 
    lx = 0.25                          # Length of computational domain in x direction
    ly = 0.25                           # Length of computational domain in y direction 
    nx = 9                        # Number of element divisions in x direction
    ny = 9                         # Number of element divisions in y direction
    nel = nx*ny                     # Total number of elements in the computational domain 
    
    # Creating a Mesh Object 
    mesh = Mesh(lx, ly, nx, ny)

    print("\n[LOG]    Mesh object has been created.")

    ############################################ STIFFNESS MATRIX
    # Creating the stiffness matrix 
    K = np.zeros((mesh.nnodes*5, mesh.nnodes*5))

    # Calculation of the element stiffness matrix
    for i in range(nel):
        Ke = matKe(i, H, mesh.getNodeCoords(i,0), mesh.getNodeCoords(i,1), mesh.getNodeCoords(i,2), mesh.getNodeCoords(i,3), plyHeights[-1])

        # #DEBUG
        # stiffnessName = "stiffnessMatrixElement" + str(i) + ".csv"
        # np.savetxt(stiffnessName, Ke, delimiter=',')

        # Getting list of nodes that belong to the element 
        eloc = mesh.getElementNodes(i)

        # Adding the element stiffnesses to the global stiffness matrix 
        for ii in range(4):
            for jj in range(4):
                # Here we are adding the element stiffness as a 5 x 5 sub matrix corresponding to the stiffness relation between 2 nodes 
                # By looping over each of the 16 (4 x 4) divisions of the 20 x 20 element stiffness matrix
                # print("\[DEBUG]\tElement ", i, " Stiffness Matrix [", ii*5,":",ii*5+5,", ",jj*5,":",jj*5+5, "] is placed at K[",eloc[ii]*5,":",eloc[ii]*5 + 5,", ", eloc[jj]*5,":",eloc[jj]*5 + 5, "]")
                K[eloc[ii]*5:eloc[ii]*5 + 5, eloc[jj]*5:eloc[jj]*5 + 5] += Ke[ii*5:ii*5+5, jj*5:jj*5+5]

    np.savetxt('K.csv', K, delimiter=',')
    print("\n[LOG]    Stiffness matrix has been created.")


    ############################################ FORCING
    # Creating the Force Vector 
    F = np.zeros(mesh.nnodes*5)
    #       Adding forces to the force vector 
    BodyForce = np.array([0, 0, 1e6, 0, 0])

    print("\n[LOG]    Using body force vector as: ")
    print(BodyForce)

    # Applying the body force to all the elements 
    for i in range(nel): 
        Fe = ElementBodyForce(BodyForce, mesh.getNodeCoords(i,0), mesh.getNodeCoords(i,1), mesh.getNodeCoords(i,2), mesh.getNodeCoords(i,3), plyHeights[-1])

        # Fe is a 20 x 1 vector. The first 5 elements is the force on the NE node,  5 is the force on the NW node, usw. 

        # We add the nodal forces for an element to the force vector. More than 1 element can share the same node, so we must add the forces 
        # at least that is the way that I understand it. 
        for ii in range(4):
            F[mesh.elements[i].nodes[ii]*5:mesh.elements[i].nodes[ii]*5+5] += Fe[ii*5:ii*5+5]
        
    print("\n[LOG]    Force vector has been constructed.")

    ############################################ BOUNDARY CONDITIONS
    #       Applying boundary conditions 
    #               Fully clamped edges  
    bnodes = np.array(mesh.getBoundaries())

    bdofsList = []                  # Boundary DOF list
    for bnode in bnodes:
        for bdof in range(5):
            bdofsList.append(bnode + bdof)
    
    # Array of boundary degree of freedom
    bdofs = np.array(bdofsList)
    
    # #DEBUG
    # print("\n[DEBUG]\tThe boundary nodes are: ", bnodes)
    # print("\n[DEBUG]\tThe shape of F: ", np.shape(F))
    # print("\n[DEBUG]\tThe shape of K: ", np.shape(K))
    # print(F)
    # for i, f in enumerate(F):
    #     if ( i%5 == 2 ):
    #         print(f)

    #               Deleting the rows and columns from K and rows from F that correspond to the clamped nodes 
    # K = np.delete(K, bnodes*5 + 2, 0)
    # K = np.delete(K, bnodes*5 + 2, 1)
    # F = np.delete(F, bnodes*5 + 2, 0)
    K = np.delete(K, bdofs, 0)
    K = np.delete(K, bdofs, 1)
    F = np.delete(F, bdofs, 0)

    np.savetxt('K_constraints.csv', K, delimiter=',')

    print("\n[LOG]    Completed applying boundary conditions.")


    ############################################ SOLVING MATRIX EQUATIONS

    print("\n[LOG]    Solving matrix equations...")

    # Solving the equation 
    #X = linsolve.spsolve(K, F)
    X = scipy.linalg.solve(K, F)
    # X, exitcode= scipy.sparse.linalg.gmres(K,F)
    print(X)
    # print(exitcode)

    # # Reconstructing the nodal DOFs
    # print("\n[DEBUG]\tThe shape of X: ", np.shape(X))
    # print("\n[DEBUG]\tThe shape of F: ", np.shape(F))
    # print("\n[DEBUG]\tThe shape of K: ", np.shape(K))

    boundedDOF = np.zeros(1)
    boundarynodes = bnodes

    for i in range(mesh.nnodes):
        if ( i == boundarynodes[0] ):
            # Inserting the previously removed node
            for ii in range(5):
                X = np.insert(X, 5*i, boundedDOF)

            # Popping off the boundary node
            boundarynodes = boundarynodes[1:]

            # Exiting the loop if all is done 
            if ( boundarynodes.size == 0 ): break        

    # print("\n[DEBUG]\tThe shape of X: ", np.shape(X))
    # print("\n[DEBUG]\tThe shape of F: ", np.shape(F))
    # print("\n[DEBUG]\tThe shape of K: ", np.shape(K))
    # for i, x in enumerate(X):
    #     if ( i%5 == 2 ):
    #         print(x)

    print("\n[LOG]    Solved the matrix equation.")

    ############################################ POST PROCESSING SOLUTION
    # Visualizing the displacement of the plate subjected to the loading condition
    xCoordinates = np.zeros(mesh.nnodes)
    yCoordinates = np.zeros(mesh.nnodes)
    zCoordinates = np.zeros(mesh.nnodes)

    #   Looping over all the nodes and getting the coordinates (this is prior to the displacement and displacement need to be added to it)
    for i, node in enumerate(mesh.nodes):
        xCoordinates[i] = node.x
        yCoordinates[i] = node.y
        zCoordinates[i] = 0.0               # This analysis is based on the assumption that the plate is laid out flat 

    fig = go.Figure(data=[go.Mesh3d(x=xCoordinates, y=yCoordinates, z=zCoordinates, color='red', opacity=0.2)]) 
    fig.show()
    
    # for i, node in enumerate(mesh.nodes):
    #     # Adding the displacements from the solution to the nodal DOFs
    #     # u = u_o + w*psi_x
    #     xCoordinates[i] += X[5*i + 0] + X[5*i + 2]*X[5*i + 3]
    #     # v = v_o + w*psi_y
    #     yCoordinates[i] += X[5*i + 1] + X[5*i + 2]*X[5*i + 4]
    #     # w = w_o
    #     zCoordinates[i] += X[5*i + 2]
    
    xElem = np.zeros(mesh.nel)
    yElem = np.zeros(mesh.nel)
    zElem = np.zeros(mesh.nel)
    
    for i, element in enumerate(mesh.elements):
        xe = np.zeros(5)                                    # Elemental degree of freedomg 
        elnodes = element.nodes
        for j, elnode in enumerate(elnodes): 
            xElem[i] += 0.25 * mesh.getNodeCoords(i,j)[0]
            yElem[i] += 0.25 * mesh.getNodeCoords(i,j)[1]

            # Normally we would need to use the shape functions, but since we are dealing with rectangular Q4 elements, this is sufficient
            xe += 0.25 * X[elnode:elnode+5]
        
        xElem[i] += xe[0] + xe[2]*xe[3]
        yElem[i] += xe[1] + xe[2]*xe[4]
        zElem[i] += xe[2]
    
    # fig = go.Figure(data=[go.Mesh3d(x=xCoordinates, y=yCoordinates, z=zCoordinates, color='red', opacity=0.2)]) 
    # fig.show()
    fig = go.Figure(data=[go.Mesh3d(x=xElem, y=yElem, z=zElem, color='green', opacity=0.2)]) 
    fig.show()


    return 


if __name__ == "__main__":

    Q4_composite_laminate_plate()