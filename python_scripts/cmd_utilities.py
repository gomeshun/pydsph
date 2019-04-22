import numpy as np

def inpolygon(x, y, xs_vertex, ys_vertex):
    ''' 
    xs_vertex[:], ys_vertex[:]: polygon
    x, y: point
    '''     
    n_p = len(xs_vertex)
    inside = False
    for i1 in range(n_p): 
        i2 = (i1+1)%n_p
        if min(xs_vertex[i1], xs_vertex[i2]) < x < max(xs_vertex[i1], xs_vertex[i2]):
            #a = (y[i2]-y[i1])/(x[i2]-x[i1])
            #b = y[i1] - a*x[i1]
            #dy = a*x+b - y
            #if dy >= 0:
            if (ys_vertex[i1] + (ys_vertex[i2]-ys_vertex[i1])/(xs_vertex[i2]-xs_vertex[i1])*(x-xs_vertex[i1]) - y) > 0:
                inside = not inside

    return inside

def inpoly(x,y,xs_vertex,ys_vertex):
    ''' 
    xs_vertex[:], ys_vertex[:]: polygon
    x, y: point
    '''
    x1 = xs_vertex[:,np.newaxis]
    y1 = ys_vertex[:,np.newaxis]
    x2 = np.roll(x1,-1)
    y2 = np.roll(y1,-1)
    print(np.array([x1[:,0],y1[:,0]]))
    print(np.array([x2[:,0],y2[:,0]]))
    ispointbetweenithsep = (x1-x)*(x2-x)<0
    ispointaboveithsep = (x2-x1)*((x2-x1)*(y-y1)-(y2-y1)*(x-x1))>0
    numofsepbelowpoint = (ispointbetweenithsep*ispointaboveithsep).sum(axis=0)
    isoddnumbersepabovepoint = (numofsepbelowpoint%2 == 1)
    #print(ispointbetweenithsep)
    #print(ispointaboveithsep)
    #print(numofsepbelowpoint)
    #print(isoddnumbersepabovepoint)
    return isoddnumbersepabovepoint

