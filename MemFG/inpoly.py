import numpy as np

def inpolygon(sx, sy, x, y):
    ''' 
    x[:], y[:]: polygon
    sx, sy: point
    '''     
    np = len(x)
    inside = False
    for i1 in range(np): 
        i2 = (i1+1)%np
        if min(x[i1], x[i2]) < sx < max(x[i1], x[i2]):
            #a = (y[i2]-y[i1])/(x[i2]-x[i1])
            #b = y[i1] - a*x[i1]
            #dy = a*sx+b - sy
            #if dy >= 0:
            if (y[i1] + (y[i2]-y[i1])/(x[i2]-x[i1])*(sx-x[i1]) - sy) > 0:
                inside = not inside

    return inside

def inpoly(sx,sy,x,y,verbose=False):
    ''' 
    x[:], y[:]: polygon
    sx, sy: point
    '''    
    x1 = x[:,np.newaxis]
    y1 = y[:,np.newaxis]
    x2 = np.roll(x1,-1)
    y2 = np.roll(y1,-1)
    if verbose:
        print(np.array([x1[:,0],y1[:,0]]))
        print(np.array([x2[:,0],y2[:,0]]))
    ispointbetweenithsep = (x1-sx)*(x2-sx)<0
    ispointaboveithsep = (x2-x1)*((x2-x1)*(sy-y1)-(y2-y1)*(sx-x1))>0
    numofsepbelowpoint = (ispointbetweenithsep*ispointaboveithsep).sum(axis=0)
    isoddnumbersepabovepoint = (numofsepbelowpoint%2 == 1)
    #print(ispointbetweenithsep)
    #print(ispointaboveithsep)
    #print(numofsepbelowpoint)
    #print(isoddnumbersepabovepoint)
    return isoddnumbersepabovepoint