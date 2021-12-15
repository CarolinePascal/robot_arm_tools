from os import write
import numpy as np
import trimesh as tr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
cmap = plt.cm.get_cmap('hsv', 100)

def plotMesh(vertices,faces,tetrahedra):
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    for i,vertex in enumerate(vertices):
        if(i%2==0):
            ax.scatter(vertex[0],vertex[1],vertex[2],color='b')
        else:
            ax.scatter(vertex[0],vertex[1],vertex[2],color='r')
    
    for face in faces:
        faceVertices = vertices[face]
        faceVertices = np.vstack((faceVertices,vertices[face[0]]))
        ax.plot(faceVertices[:,0],faceVertices[:,1],faceVertices[:,2],'k')

    for k,tetrahedron in enumerate(tetrahedra):
        tetrahedraVertices = vertices[tetrahedron]
        for i in range(4):
            apexes = np.array([tetrahedraVertices[i],tetrahedraVertices[(i+1)%4],tetrahedraVertices[(i+2)%4]])           
            ax.add_collection3d(Poly3DCollection(apexes,alpha=0.1,color=cmap(10*k%100)))

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
    plt.show()

def createMesh(Ntheta,Nphi,Rmin,Rmax):

    vertices = np.empty((2*(2*Nphi*(Ntheta-1)+2),3))

    sphericPose = lambda r,theta,phi : np.array([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])

    ###VERTICES

    vertices[0] = np.array([0,0,Rmin])
    vertices[1] = np.array([0,0,Rmax])

    index = lambda i,j : 2*Nphi*(i-1) + j + 1

    for i in range(1,Ntheta):
        for j in range(2*Nphi):
            #Inner vertex
            vertices[2*index(i,j)] = sphericPose(Rmin,i*np.pi/Ntheta,j*np.pi/Nphi)
            #Outer vertex
            vertices[2*index(i,j) + 1] = sphericPose(Rmax,i*np.pi/Ntheta,j*np.pi/Nphi)

    vertices[-2] = np.array([0,0,-Rmin])
    vertices[-1] = np.array([0,0,-Rmax])

    """ DEPRECATED 
    ###FILLING FACES

    #Triangular volumes (top and bottom)
    #    A
    #   /|\
    #  / | \
    # B-----C 
    # |  D  |
    # | / \ |
    # |/   \|
    # E-----F  
    # j    j+1   

    #Top triangular volumes
    i = 1
    for j in range(2*N2):
        A = 1
        D = 0
        B = 2 + 2*((N1-1)*j + i-1) + 1
        E = 2 + 2*((N1-1)*j + i-1)
        if(j != 2*N2 - 1):
            C = 2 + 2*((N1-1)*(j+1) + i-1) + 1
            F = 2 + 2*((N1-1)*(j+1) + i-1)
        else:
            C = 2 + 2*(i-1) + 1
            F = 2 + 2*(i-1)

        ##Top face
        faces[5*j] = np.array([A,B,C])
        ##Bottom face
        faces[5*j+1] = np.array([D,E,F])
        ##Traversing face
        faces[5*j+2] = np.array([A,E,F])
        ##Left faces
        faces[5*j+3] = np.array([A,E,B])
        faces[5*j+4] = np.array([A,E,D])

    #Bottom triangular volumes
    i = N1 - 1
    for j in range(2*N2):
        A = 2*(2*N2*(N1-1))+2
        D = 2*(2*N2*(N1-1))+3
        B = 2 + 2*((N1-1)*j + i-1)
        E = 2 + 2*((N1-1)*j + i-1) + 1
        if(j != 2*N2 - 1):
            C = 2 + 2*((N1-1)*(j+1) + i-1)
            F = 2 + 2*((N1-1)*(j+1) + i-1) + 1
        else:
            C = 2 + 2*(i-1)
            F = 2 + 2*(i-1) + 1

        #Top face
        faces[5*2*N2+5*j] = np.array([A,B,C])
        ##Bottom face
        faces[5*2*N2+5*j+1] = np.array([D,E,F])
        ##Traversing face
        faces[5*2*N2+5*j+2] = np.array([A,E,F])
        ##Left faces
        faces[5*2*N2+5*j+3] = np.array([A,E,B])
        faces[5*2*N2+5*j+4] = np.array([A,E,D])


    #Cubical volumes (middle)
    #Triangular volumes (top and bottom)
    #       E------F
    #      /|     /|
    #  i  A------B |
    #     | G----|-H    
    #     |/     |/
    # i+1 C------D
    #     j     j+1     

    for i in range(1,N1-1):
        for j in range(2*N2):
            A = 2 + 2*((N1-1)*j + i-1) + 1
            C = 2 + 2*((N1-1)*j + i) + 1
            E = 2 + 2*((N1-1)*j + i-1)
            G = 2 + 2*((N1-1)*j + i)
            if(j != 2*N2 - 1):
                B = 2 + 2*((N1-1)*(j+1) + i-1) + 1
                D = 2 + 2*((N1-1)*(j+1) + i) + 1
                F = 2 + 2*((N1-1)*(j+1) + i-1)
                H = 2 + 2*((N1-1)*(j+1) + i)
            else:
                B = 2 + 2*(i-1) + 1
                D = 2 + 2*i + 1
                F = 2 + 2*(i-1)
                H = 2 + 2*i

            ##Top faces
            faces[2*5*2*N2+10*((N1-2)*j + i-1)] = np.array([A,B,E])
            faces[2*5*2*N2+10*((N1-2)*j + i-1)+1] = np.array([E,B,F])
            ##Left faces
            faces[2*5*2*N2+10*((N1-2)*j + i-1)+2] = np.array([A,E,C])
            faces[2*5*2*N2+10*((N1-2)*j + i-1)+3] = np.array([C,E,G])
            ##Outer faces
            faces[2*5*2*N2+10*((N1-2)*j + i-1)+4] = np.array([A,D,C])
            faces[2*5*2*N2+10*((N1-2)*j + i-1)+5] = np.array([A,D,B])
            ##Inner faces
            faces[2*5*2*N2+10*((N1-2)*j + i-1)+6] = np.array([E,G,H])
            faces[2*5*2*N2+10*((N1-2)*j + i-1)+7] = np.array([E,H,F])
            ##Traversing faces
            faces[2*5*2*N2+10*((N1-2)*j + i-1)+8] = np.array([E,D,C])
            faces[2*5*2*N2+10*((N1-2)*j + i-1)+9] = np.array([E,D,F])

            if(i == N1-2):
                #Bottom faces
                faces[2*5*2*N2+2*N2*(N1-2)*10+2*j] = np.array([G,D,C])
                faces[2*5*2*N2+2*N2*(N1-2)*10+2*j+1] = np.array([G,D,H])
    """

    #BOURNDARY FACES

    #Triangular faces (top and bottom)
    #    A
    #   / \
    #  /   \
    # B-----C  
    # j    j+1   

    #Square faces (middle)
    #  i  B------A
    #     |      |
    #     |      |
    # i+1 C------D
    #     j     j+1 

    #Inner boundary
    innerBoundary = np.empty((2*Nphi*2+2*Nphi*(Ntheta-2)*2,3),dtype=int)
    outerBoundary = np.empty((2*Nphi*2+2*Nphi*(Ntheta-2)*2,3),dtype=int)

    #Triangle faces
    for k,i in enumerate([1,Ntheta-1]):
        for j in range(2*Nphi):
            A = 0 if i == 1 else 2*(2*Nphi*(Ntheta-1)+2)-2
            B = 2*index(i,j)
            if(j != 2*Nphi - 1):
                C = 2*index(i,j+1) 
            else:
                C = 2*index(i,0)

            if(i==1):
                innerBoundary[2*Nphi*k+j] = np.array([A,B,C])
            else:
                innerBoundary[2*Nphi*k+j] = np.array([A,C,B])

            A = 1 if i == 1 else 2*(2*Nphi*(Ntheta-1)+2)-1
            B = 2*index(i,j) + 1
            if(j != 2*Nphi - 1):
                C = 2*index(i,j+1) + 1
            else:
                C = 2*index(i,0) + 1

            if(i==1):
                outerBoundary[2*Nphi*k+j] = np.array([A,B,C])
            else:
                outerBoundary[2*Nphi*k+j] = np.array([A,C,B])

    #Square faces
    for i in range(1,Ntheta-1):
        for j in range(2*Nphi):
            B = 2*index(i,j)
            C = 2*index(i+1,j)

            if(j != 2*Nphi - 1):
                A = 2*index(i,j+1) 
                D = 2*index(i+1,j+1) 
            else:
                A = 2*index(i,0) 
                D = 2*index(i+1,0) 

            innerBoundary[2*Nphi*2 + 2*(2*Nphi*(i-1)+j)] = np.array([B,C,A])
            innerBoundary[2*Nphi*2 + 2*(2*Nphi*(i-1)+j)+1] = np.array([A,C,D])

            B = 2*index(i,j) + 1
            C = 2*index(i+1,j) + 1

            if(j != 2*Nphi - 1):
                A = 2*index(i,j+1) + 1
                D = 2*index(i+1,j+1) + 1
            else:
                A = 2*index(i,0) + 1
                D = 2*index(i+1,0) + 1

            outerBoundary[2*Nphi*2 + 2*(2*Nphi*(i-1)+j)] = np.array([B,C,D])
            outerBoundary[2*Nphi*2 + 2*(2*Nphi*(i-1)+j)+1] = np.array([B,D,A])

    innerBoundary+=1
    outerBoundary+=1
    triangles = np.vstack((innerBoundary,outerBoundary))

    #TETRAHEDRON
    tetrahedra = np.empty((2*Nphi*2*3 + 2*Nphi*(Ntheta-2)*5,4),dtype=int)

    #Triangular volumes
    for k in range(2*Nphi*2):
        innerFace = innerBoundary[k]
        outerFace = outerBoundary[k]

        tetrahedra[3*k] = np.array([outerFace[0],outerFace[1],innerFace[2],outerFace[2]])
        tetrahedra[3*k+1] = np.array([innerFace[0],innerFace[1],innerFace[2],outerFace[1]])
        tetrahedra[3*k+2] = np.array([innerFace[0],outerFace[1],innerFace[2],outerFace[0]])

    #Cubic volumes
    for k in range(2*Nphi*(Ntheta-2)):
        leftInnerFace = innerBoundary[2*Nphi*2 + 2*k]
        leftOuterFace = outerBoundary[2*Nphi*2 + 2*k]
        rightInnerFace = innerBoundary[2*Nphi*2 + 2*k+1]
        rightOuterFace = outerBoundary[2*Nphi*2 + 2*k+1]

        tetrahedra[2*Nphi*2*3 + 5*k] = np.array([leftOuterFace[0],leftInnerFace[1],leftOuterFace[2],leftOuterFace[1]])
        tetrahedra[2*Nphi*2*3 + 5*k+1] = np.array([rightOuterFace[0],rightOuterFace[1],leftInnerFace[2],rightOuterFace[2]])
        tetrahedra[2*Nphi*2*3 + 5*k+2] = np.array([leftInnerFace[1],leftOuterFace[0],leftInnerFace[2],leftInnerFace[0]])
        tetrahedra[2*Nphi*2*3 + 5*k+3] = np.array([rightInnerFace[0],leftOuterFace[2],rightInnerFace[1],rightInnerFace[2]])
        tetrahedra[2*Nphi*2*3 + 5*k+4] = np.array([leftInnerFace[1],leftOuterFace[2],leftInnerFace[2],leftOuterFace[0]])

    with open("Mesh_"+str(Ntheta)+"_"+str(Rmin)+"_"+str(Rmax)+".mesh",'w') as writer:
        writer.write("MeshVersionFormatted 2\n")
        writer.write("Dimension 3\n")

        writer.write("\nVertices\n")
        writer.write(str(len(vertices))+"\n")

        for vertex in vertices:
            writer.write(str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + " 1\n")

        writer.write("\nTriangles\n")
        writer.write(str(len(triangles))+"\n")

        for triangle in innerBoundary:
            writer.write(str(triangle[0]) + " " + str(triangle[1]) + " " + str(triangle[2]) + " 1\n")
        for triangle in outerBoundary:
            writer.write(str(triangle[0]) + " " + str(triangle[1]) + " " + str(triangle[2]) + " 2\n")

        writer.write("\nTetrahedra\n")
        writer.write(str(len(tetrahedra))+"\n")

        for tetrahedron in tetrahedra:
            writer.write(str(tetrahedron[0]) + " " + str(tetrahedron[1]) + " " + str(tetrahedron[2]) + " " + str(tetrahedron[3]) + " 1\n")
