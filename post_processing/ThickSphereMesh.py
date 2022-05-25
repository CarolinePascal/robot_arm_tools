import os
import numpy as np
from numpy import outer
from numpy import delete
import trimesh as tr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Mesh:
    def __init__(self):

        ###MESH DATA
        self.vertices = []
        self.triangles = []
        self.tetrahedra = []
        self.trianglesBoundaries = {}

        ###MESH SPECIFICATIONS
        self.path = "meshes/mesh.mesh"

    def plot(self):

        #Create figure and axis
        fig = plt.figure() 
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.cm.get_cmap('hsv', len(self.tetrahedra))

        #Plot vertices
        for i,vertex in enumerate(self.vertices):
            ax.scatter(vertex[0],vertex[1],vertex[2],color='b')
        
        #Plot triangles
        for triangle in self.triangles:
            triangleVertices = self.vertices[triangle-1]
            triangleVertices = np.vstack((triangleVertices,self.vertices[triangle[0]-1]))
            ax.plot(triangleVertices[:,0],triangleVertices[:,1],triangleVertices[:,2],'k')

        #Plot tetrahedra
        for k,tetrahedron in enumerate(self.tetrahedra):
            tetrahedraVertices = self.vertices[tetrahedron-1]
            for i in range(4):
                apexes = np.array([tetrahedraVertices[i],tetrahedraVertices[(i+1)%4],tetrahedraVertices[(i+2)%4]])           
                ax.add_collection3d(Poly3DCollection(apexes,alpha=0.1,color=cmap(k)))

        #Set the same scaling to all axis
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        plt.show()

    def write(self, format = "gmsh"):

        #Create directories if needed
        directory = os.path.split(self.path)[0]
        if(not os.path.isdir(directory)):
            os.makedirs(directory,exist_ok=True)

        #Check if the mesh already exist
        else:
            if(os.path.isfile(self.path)):
                return()

        if(format == "gmsh"):

            with open(self.path,'w') as writer:
                writer.write("MeshVersionFormatted 2\n")
                writer.write("Dimension 3\n")

                writer.write("\nVertices\n")
                writer.write(str(len(self.vertices))+"\n")

                for vertex in self.vertices:
                    writer.write(str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + " 1\n")

                writer.write("\nTriangles\n")
                writer.write(str(len(self.triangles))+"\n")

                if(bool(self.trianglesBoundaries)):
                    for id,indexes in self.trianglesBoundaries.items():
                        for triangle in self.triangles[indexes]:
                            writer.write(str(triangle[0]) + " " + str(triangle[1]) + " " + str(triangle[2]) + " " + id + "\n")
                else:
                    for triangle in self.triangles:
                            writer.write(str(triangle[0]) + " " + str(triangle[1]) + " " + str(triangle[2]) + " 0\n")

                writer.write("\nTetrahedra\n")
                writer.write(str(len(self.tetrahedra))+"\n")

                for tetrahedron in self.tetrahedra:
                    writer.write(str(tetrahedron[0]) + " " + str(tetrahedron[1]) + " " + str(tetrahedron[2]) + " " + str(tetrahedron[3]) + " 1\n")

        elif(format == "xml"):

            with open(self.path,'w') as writer:
                writer.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')

                writer.write('<dolfin xmlns:dolfin="http://www.fenicsproject.org">\n')
                writer.write('\t<mesh celltype="tetrahedron" dim="3">\n')

                writer.write('\t\t<vertices size="' + str(len(self.vertices)) + '">\n')

                for i,vertex in enumerate(self.vertices):
                    writer.write('\t\t\t<vertex index="' + str(i) + '" x="' + str(vertex[0]) + '" y="' + str(vertex[1]) + '" z="' + str(vertex[2]) + '"/>\n')

                writer.write('\t\t</vertices>\n')

                writer.write('\t\t<cells size="' + str(len(self.tetrahedra)) + '">\n')

                for i,tetrahedron in enumerate(self.tetrahedra):
                    writer.write('\t\t\t<tetrahedron index="' + str(i) + '" v0="' + str(tetrahedron[0]-1) + '" v1="' + str(tetrahedron[1]-1) + '" v2="' + str(tetrahedron[2]-1) + '" v3="' + str(tetrahedron[3]-1) +'"/>\n')

                writer.write('\t\t</cells>\n')
                writer.write('\t</mesh>\n')
                writer.write('</dolfin>\n')

class ThickSphericMesh(Mesh) : 
    def __init__(self, Ntheta, Nphi, Rmin, Rmax, layers = 1):
        Mesh.__init__(self)

        ###MESH SPECIFICATIONS
        self.Ntheta = Ntheta
        self.Nphi = Nphi
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.layers = layers

        self.path = "meshes/thick_sphere_" + str(self.layers) + "/TS_" + str(self.Ntheta) + "_" + str(self.Rmin) + "_" + str(self.Rmax) + ".mesh"

        ###MESH DATA
        self.vertices, self.triangles, self.tetrahedra = ThickSphericMesh.createMeshData(self.Ntheta, self.Nphi, self.Rmin, self.Rmax, self.layers)
        self.trianglesBoundaries = {}
        self.trianglesBoundaries["1"] = np.arange(0,len(self.triangles)//2)
        self.trianglesBoundaries["2"] = np.arange(len(self.triangles)//2,len(self.triangles))

    @staticmethod
    def createMeshData(Ntheta, Nphi, Rmin, Rmax, layers = 1):
    
        sphericPose = lambda r,theta,phi : np.array([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])

        halfVerticesNumber = 2*(Nphi*(Ntheta-1) + 1)
        halfTrianglesNumber = 4*Nphi*(Ntheta - 1)
        tetrahedraNumber = 2*Nphi*(6 + 5*(Ntheta-2))

        if(layers == 1):

            ###VERTICES
            vertices = np.empty((2*halfVerticesNumber,3))

            #Inner vertex
            vertices[0] = np.array([0,0,Rmin])
            vertices[halfVerticesNumber - 1] = np.array([0,0,-Rmin])
            
            #Outer vertex
            vertices[halfVerticesNumber] = np.array([0,0,Rmax])
            vertices[2*halfVerticesNumber - 1] = np.array([0,0,-Rmax])

            for i in range(1,Ntheta):
                for j in range(2*Nphi):
                    #Inner vertex
                    vertices[1 + (i-1)*2*Nphi + j] = sphericPose(Rmin,i*np.pi/Ntheta,j*np.pi/Nphi)
                    #Outer vertex
                    vertices[halfVerticesNumber + 1 + (i-1)*2*Nphi + j] = sphericPose(Rmax,i*np.pi/Ntheta,j*np.pi/Nphi)


            #BOURNDARY FACES

            #Triangular faces (top and bottom)
            #    A
            #   / \
            #  /   \
            # B-----C  
            # j    j+1   

            #Square faces (middle)
            #  i  A------D
            #     |      |
            #     |      |
            # i+1 B------C
            #     j     j+1 

            triangles = np.empty((2*halfTrianglesNumber,3),dtype=int)

            for i in range(1,Ntheta-1):
                for j in range(2*Nphi):

                    #Upper triangular faces
                    if(i == 1):
                        A = 0
                        B = 1 + j
                        C = 2 + j if j < 2*Nphi - 1 else 1
                        triangles[j] = np.array([A,B,C])
                        triangles[halfTrianglesNumber + j] = np.array([A,B,C]) + halfVerticesNumber

                    #Lower triangular faces
                    if(i == Ntheta - 2):
                        A = halfVerticesNumber - 1
                        B = 1 + (Ntheta-2)*2*Nphi + j
                        C = 2 + (Ntheta-2)*2*Nphi + j if j < 2*Nphi - 1 else 1 + (Ntheta - 2)*2*Nphi
                        triangles[(2*(Ntheta-2) + 1)*2*Nphi + j] = np.array([A,C,B])
                        triangles[halfTrianglesNumber + (2*(Ntheta-2) + 1)*2*Nphi + j] = np.array([A,C,B]) + halfVerticesNumber

                    #Square faces
                    A = 1 + (i-1)*2*Nphi + j
                    B = 1 + i*2*Nphi + j
                    C = 2 + i*2*Nphi + j if j < 2*Nphi - 1 else 1 + i*2*Nphi
                    D = 2 + (i-1)*2*Nphi + j if j < 2*Nphi - 1 else 1 + (i-1)*2*Nphi
                    triangles[(2*i - 1)*2*Nphi + 2*j] = np.array([A,B,D]) 
                    triangles[(2*i - 1)*2*Nphi + 2*j + 1] = np.array([B,C,D])
                    triangles[halfTrianglesNumber + (2*i - 1)*2*Nphi + 2*j] = np.array([A,B,C]) + halfVerticesNumber
                    triangles[halfTrianglesNumber + (2*i - 1)*2*Nphi + 2*j + 1] = np.array([A,C,D]) + halfVerticesNumber

            triangles += 1

            #TETRAHEDRON
            tetrahedra = np.empty((tetrahedraNumber,4),dtype=int)

            for i in range(1,Ntheta-1):
                for j in range(2*Nphi):

                    #Upper triangular volumes
                    if(i == 1):
                        innerFace = triangles[j]
                        outerFace = triangles[halfTrianglesNumber + j]

                        tetrahedra[3*j] = np.array([outerFace[0],outerFace[1],innerFace[2],outerFace[2]])
                        tetrahedra[3*j+1] = np.array([innerFace[0],innerFace[1],innerFace[2],outerFace[1]])
                        tetrahedra[3*j+2] = np.array([innerFace[0],outerFace[1],innerFace[2],outerFace[0]])

                    #Lower triangular volumes
                    if(i == Ntheta - 2):
                        innerFace = triangles[(2*(Ntheta-2) + 1)*2*Nphi + j]
                        outerFace = triangles[halfTrianglesNumber + (2*(Ntheta-2) + 1)*2*Nphi + j]

                        tetrahedra[(5*(Ntheta-2) + 3)*2*Nphi + 3*j] = np.array([outerFace[0],outerFace[1],innerFace[2],outerFace[2]])
                        tetrahedra[(5*(Ntheta-2) + 3)*2*Nphi + 3*j+1] = np.array([innerFace[0],innerFace[1],innerFace[2],outerFace[1]])
                        tetrahedra[(5*(Ntheta-2) + 3)*2*Nphi + 3*j+2] = np.array([innerFace[0],outerFace[1],innerFace[2],outerFace[0]])

                    leftInnerFace = triangles[(2*i - 1)*2*Nphi + 2*j]
                    leftOuterFace = triangles[halfTrianglesNumber + (2*i - 1)*2*Nphi + 2*j]
                    rightInnerFace = triangles[(2*i - 1)*2*Nphi + 2*j + 1]
                    rightOuterFace = triangles[halfTrianglesNumber + (2*i - 1)*2*Nphi + 2*j + 1]

                    tetrahedra[(5*i - 2)*Nphi*2 + 5*j] = np.array([leftOuterFace[0],leftInnerFace[1],leftOuterFace[2],leftOuterFace[1]])
                    tetrahedra[(5*i - 2)*Nphi*2 + 5*j+1] = np.array([rightOuterFace[0],rightOuterFace[1],leftInnerFace[2],rightOuterFace[2]])
                    tetrahedra[(5*i - 2)*Nphi*2 + 5*j+2] = np.array([leftInnerFace[1],leftOuterFace[0],leftInnerFace[2],leftInnerFace[0]])
                    tetrahedra[(5*i - 2)*Nphi*2 + 5*j+3] = np.array([rightInnerFace[0],leftOuterFace[2],rightInnerFace[1],rightInnerFace[2]])
                    tetrahedra[(5*i - 2)*Nphi*2 + 5*j+4] = np.array([leftInnerFace[1],leftOuterFace[2],leftInnerFace[2],leftOuterFace[0]])

            return(vertices,triangles,tetrahedra)

        else:

            deltaR = (Rmax - Rmin)/layers

            ###INNER LAYER

            totalVertices = np.empty(((layers+1)*halfVerticesNumber,3))
            totalTetrahedra = np.empty((layers*tetrahedraNumber,4),dtype=int)
            totalTriangles = np.empty((2*halfTrianglesNumber,3),dtype=int)

            vertices, triangles, tetrahedra = ThickSphericMesh.createMeshData(Ntheta, Nphi, Rmin, Rmin + deltaR)

            totalVertices[:halfVerticesNumber] = vertices[:halfVerticesNumber]
            totalTetrahedra[:tetrahedraNumber] = tetrahedra
            totalTriangles[:halfTrianglesNumber] = triangles[:halfTrianglesNumber]

            ###INTERMEDIATE LAYERS
            for i in range(1,layers):
                vertices, _, _ = ThickSphericMesh.createMeshData(Ntheta, Nphi, Rmin + i*deltaR, Rmin + (i+1)*deltaR)
                totalVertices[halfVerticesNumber*i:halfVerticesNumber*(i+1)] = vertices[:halfVerticesNumber]
                totalTetrahedra[i*tetrahedraNumber:(i+1)*tetrahedraNumber] = tetrahedra + i*halfVerticesNumber
                
            ###OUTER LAYER
            totalVertices[halfVerticesNumber*layers:] = vertices[halfVerticesNumber:]
            totalTriangles[halfTrianglesNumber:] = triangles[halfTrianglesNumber:] + (layers-1)*halfVerticesNumber

            return(totalVertices,totalTriangles,totalTetrahedra)
    