'''
Created on 17.01.2012
Calculate all extreme equilibria of a bimatrix game 
with vertex enumeration or one equilibrium with Lemke-
Howson. The solutions are returned as a list of tuples 
[(P*,H1*,Q*,H2*), ...]
@author: Mort Canty
'''
import subprocess
from fractions import Fraction 
from sympy import *
from cvxopt import matrix, solvers
import numpy as np 
import mpmath as mp  

solvers.options['show_progress'] = False

def fract(n,asfloat=True,den=30):
    if asfloat:
        return float(Fraction(str(n)).limit_denominator(den))
    else:
        return str(Fraction(str(n)).limit_denominator(den))

def minimax(A):
    '''Minimax solution of a matrix game. A is a numpy matrix.
       Returns minimax strategy of player1 and value of the game.'''    
    m,n = A.shape
    s = np.min(A)
    A = A-s+1
    G = np.bmat([[A.T],[np.eye(m)]])
    g = -matrix(G)
    c = matrix([1.0 for i in range(m)])  
    H = np.zeros(m+n)
    H[0:n] = np.ones(n)
    h = -matrix(H)   
    sol = solvers.lp(c,g,h)
    X = np.array(sol['x'].T)[0]
    v = 1/np.sum(X)
    P = map(fract,(v*X).tolist())
    V = fract(v+s-1)
    return (P,V)

def undominated(P,A):
    ''' Returns True if mixed strategy P is undominated for payoff matrix A
        else False. A and P are numpy matrices.'''
    m,n = A.shape
# construct associated matrix game Ap   
    PA = np.tile((P*A).T, (1,m))
    Ap = (A.T - PA).T
    xxx = minimax(Ap)[1]
    if xxx == 0:
#     if its value is zero, proceed to setup LP         
        s = np.min(Ap)
        Ap = Ap-s+1
        m1 = np.mat(np.ones((m,1))) 
        n1 = np.mat(np.ones((n,1)))
        n0 = np.mat(np.zeros((n,1)))
        m0 = np.mat(np.zeros((m,1)))
        t0 = np.mat(np.zeros((2,1)))
        mm0 = np.mat(np.zeros((m,m)))
        nn0 = np.mat(np.zeros((n,n)))  
        imn = np.mat(np.eye(m+n))     
        C = np.bmat( [[-(Ap*n1)], [n0]] )
        c = matrix(C)
        B = np.bmat( [[t0],[n1],[-m1], [n0], [m0]] )
        b = -matrix(B)
        M = np.bmat( [[m1.T, -n1.T], [-m1.T, n1.T], [Ap.T, nn0], [mm0, -Ap], [imn]]  )
        mm = -matrix(M)
        sol = solvers.lp(c,mm,b)
        X = np.array(sol['x'].T)[0]
        X = np.mat(X)
        v = fract((X*C)[0,0])
        if -v == n:
            return True
        else:
            return False
    else:
#      dominated strategy         
        return False
            

# routines for Lemke-Howson
def leaving(T,s):
    
    def lexLess(X,Y):
        i = np.where(X-Y)[0]
        i0 = int(i[0])
        return X[i0]<Y[i0]
#        return X[i[0]]<Y[i[0]]
    
    def lexMax(S):
        
        if len(S) == 1:    # singleton
            return S[0]
        else:              # degenerate game   
            Vs = [ T[S[i],:]/T[S[i],s+1] for i in range(len(S)) ]
            posmax = S[0]  
            Vmax = Vs[0]
            ell = 1
            while ell < len(S):
                if lexLess(Vmax,Vs[ell]):
                    posmax = S[ell]
                else:
                    Vmax = Vs[ell] 
                ell += 1
        return posmax
    
    E = -(T.T)[0,:]
    U =  (T.T)[s+1,:]
    t = [E[j]/U[j] if U[j] > 0 else oo for j in range(T.rows)] 
    tmin = min(t)
    So = np.where(np.array(t)==tmin)[0]
    return  lexMax(So)

def pivot(i,j,r,s,T):
    if i == r:
        return T[r,j]/T[r,s+1]
    else:
        return T[i,j] - T[r,j]*T[i,s+1]/T[r,s+1]

def nashEquilibria(A,B=None,select='all'): 
    ''' Calculate all extreme equilibria or one
        equilibrium of a bimatrix game.
        A and B are list of lists.'''
    if B is None:
#      zero-sum game        
        A = np.asmatrix(A)
        return (minimax(A)+minimax(-A.T))
    if select =='all':   
#      complete vertex enumeration with Avis-Fukuda (lrslib)       
        m = len(A) 
        n = len(A[0])   
#      generate rational fraction game string 
        g = str(m)+' '+str(n)+'\n\n'
        for i in range(m):
            for j in range (n):
                aij = fract(A[i][j],asfloat=False)
                g += str(aij)+' '
            g += '\n'
        g += '\n' 
        for i in range(m):
            for j in range (n):
                bij = fract(B[i][j],asfloat=False)
                g += str(bij)+' '      
            g += '\n'       
#      write game file to disk   
        f = open('game','w') 
        print >>f, g
        f.close()      
#      invoke nash from lrslib           
        subprocess.call(['setupnash','game','game1','game2'],
                        stdout=subprocess.PIPE) 
        p2 = subprocess.Popen(['nash','game1','game2'],
                              stdout=subprocess.PIPE)
#      collate the equilibria        
        result = []; qs = []; H1s = []   
        line = p2.stdout.readline()    
        while line:
            if line[0] == '2' or line[0] == '1':  
                line = line.replace('/','./').split() 
                if line[0] == '2':
                    qs.append(map(eval,line[1:-1]))
                    H1s.append(eval(line[-1]))
                else:
                    p = map(eval,line[1:-1])
                    H2 = eval(line[-1])
                    for i in range(len(H1s)):
                        result.append( (p,H1s[i],qs[i],H2) )
                    qs = []  
                    H1s = []  
            line = p2.stdout.readline()    
        return result
    elif select=='perfect':
#      select the normal form perfect equilibria        
        eqs = nashEquilibria(A,B,select='all')
        perfect = []
        A = np.asmatrix(A)
        Bt = np.asmatrix(B).T
        for eq in eqs:
            P = np.asmatrix(eq[0])       
            Q = np.asmatrix(eq[2])
            if undominated(P,A) and undominated(Q,Bt):
                perfect.append(eq)
        return perfect   
    elif select=='one':
#  Lemke Howson algorithm for one equilibrium.
#      setup tableau with np matrices
        A = [ map(fract,row) for row in A ]
        B = [ map(fract,row) for row in B ]
        a = np.matrix(A)
        b = np.matrix(B)
        m, n = a.shape 
        m = int(m)
        n = int(n)
        smallest = min(np.min(a),np.min(b))
        A = a - np.ones((m,n))*(smallest - 1)
        B = b - np.ones((m,n))*(smallest - 1)
        k = m + n
        M = np.bmat([ [np.zeros((m,m)), A],[B.T, np.zeros((n,n))] ])
        T = np.bmat( [-np.ones((k,1)), M, np.eye(k)] )
#     convert it to mpmath matrix
        T = mp.matrix(T)
#     initialize       
        beta = range(k,2*k)
        s = 0
        br = -1
    #  complementary pivoting
        while (br != 0) and (br != k):
            tmp = mp.zeros(k,2*k+1)
            r = leaving(T,s)
            for i in range(k):
                for j in range(2*k+1):
                    tmp[i,j] = pivot(i,j,r,s,T)
            T = tmp
            br = beta[r]
            beta[r] = s
            if br >= k:
                s = br-k
            else:
                s = br+k       
        Y = np.zeros(2*k)
        for i in range(k):
            Y[beta[i]] = -T[i,0]
        P = Y[0:m]
        P = Matrix(P/np.sum(P))
        Q = Y[m:k]
        Q = Matrix(Q/np.sum(Q))
        H1 = (P*a*Q.T)[0]
        H2 = (P*b*Q.T)[0]
        P = list(P)
        Q = list(Q)
        return [(P,H1,Q,H2)]

    
if __name__ == '__main__':
#  Tests    

#  Winkels' game
    A = [[1,3],[1,3],[3,1],[3,1],[2.5,2.5],[2.5,2.5]]  
    B = [[1,2],[0,-1],[-2,2],[4,-1],[-1,6],[6,-1]] 
    print 'Winkels game, select=one'
    print nashEquilibria(A,B,select='one')
      
#  von Stengel's game    
    A = [[9504,-660,19976,-20526,1776,-8976],[-111771,31680,-130944,168124,-8514,52764], \
     [397584,-113850,451176,-586476,29216,-178761],[171204,-45936,208626,-263076,14124,-84436], \
     [1303104,-453420,1227336,-1718376,72336,-461736],[737154,-227040,774576,-1039236,48081,-300036]]        
    B = [[72336,48081,29216,14124,1776,-8514],[-461736,-300036,-178761,-84436,-8976,52764], \
     [1227336,774576,451176,208626,19976,-130944],[-1718376,-1039236,-586476,-263076,-20526,168124], \
     [1303104,737154,397584,171204,9504,-111771],[-453420,-227040,-113850,-45936,-660,31680]]  
    print 'von Stengels game, number of equilibria'
    ne = nashEquilibria(A,B)
    print len(ne)
    print 'select=one'
    print nashEquilibria(A,B,select='one')  
    print 'number of perfect equilibria'
    ne = nashEquilibria(A,B,select='perfect')
    print len(ne) 
     
#  Todd's game
    A = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[2/7.,2/7.,2/7.,2/7.],[3/19.,6/19.,6/19.,6/19.],[7/38.,7/38.,7/19.,7/19.]] 
    B = [[1.,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]   
    print 'Todds game. select=all'
    eqs = nashEquilibria(A,B)
    for eq in eqs:
        print eq
    print 'Todds game. select=perfect'
    eqs = nashEquilibria(A,B,select='perfect')
    for eq in eqs:
        print eq 
    print 'select=one'
    print nashEquilibria(A,B,select='one')   
     
#  Spectrum auction
    A = [[2.5,0,0,0],[4,2,0,0],[3,3,1.5,0],[2,2,2,1],[1,1,1,1]]
    B = [[1,1,0,-1],[0,0.5,0,-1],[0,0,0,-1],[0,0,0,-0.5],[0,0,0,0]]  
    print 'spectrum auction, select=all'  
    eqs = nashEquilibria(A,B,select='all')
    for eq in eqs:
        print eq
    print 'select=perfect'    
    print nashEquilibria(A,B,select='perfect')
  
#  Random game
    print 'random game, select=one'
    A = np.random.rand(100,10).tolist() 
    B = np.random.rand(100,10).tolist()    
    eq = nashEquilibria(A,B,select='one')
    print 'H1 = %f'%eq[0][1]
    
# APA
    print 'APA small'
    A=[[-19.0, -19.0, -19.0, -19.0, -13.6, -10.9, -13.6, -10.9, -46.0, -10.9, -0.0], [-19.0, -19.0, -19.0, -19.0, -13.6, -10.9, -13.6, -19.0, -24.400000000000006, -10.9, -0.0], [-19.0, -19.0, -19.0, -19.0, -13.6, -10.9, -19.0, -10.9, -24.400000000000006, -10.9, -0.0], [-19.0, -19.0, -19.0, -19.0, -46.0, -19.0, -46.0, -10.9, -24.400000000000006, -19.0, -0.0], [-19.0, -19.0, -19.0, -19.0, -19.0, -10.9, -13.6, -10.9, -46.0, -10.9, -0.0], [-100.0, -100.0, -100.0, -100.0, -13.6, -19.0, -13.6, -10.9, -24.400000000000006, -19.0, -0.0]];
    B=[[-1.1111234569510797e-06, -1.1111234569510797e-06, -1.1111234569510797e-06, -0.012501111123457198, -0.07555674075390965, -0.10300122223580263, -0.08121330641047525, -0.10363758587216625, 0.16666592591769505, -0.10416788890246928, -0.0], [-1.1111234569510797e-06, -1.1111234569510797e-06, -1.1111234569510797e-06, -0.012501111123457198, -0.07555674075390965, -0.10300122223580263, -0.08121330641047525, -0.036364747487093285, -1.0370485599153767e-06, -0.10416788890246928, -0.0], [-1.1111234569510797e-06, -1.1111234569510797e-06, -1.1111234569510797e-06, -0.012501111123457198, -0.07555674075390965, -0.10300122223580263, -0.036364747487093285, -0.10363758587216625, -1.0370485599153767e-06, -0.10416788890246928, -0.0], [-1.1111234569510797e-06, -1.1111234569510797e-06, -1.1111234569510797e-06, -0.012501111123457198, 0.24444370369547272, -0.0300011111234572, 0.18787804712981668, -0.10363758587216625, -1.0370485599153767e-06, -0.04166777779012369, -0.0], [-1.1111234569510797e-06, -1.1111234569510797e-06, -1.1111234569510797e-06, -0.012501111123457198, -0.022223333345679275, -0.10300122223580263, -0.08121330641047525, -0.10363758587216625, 0.16666592591769505, -0.10416788890246928, -0.0], [1.0, 1.0, 1.0, 0.8749999999999976, -0.07555674075390965, -0.0300011111234572, -0.08121330641047525, -0.10363758587216625, -1.0370485599153767e-06, -0.04166777779012369, -0.0]]
    print nashEquilibria(A,B,select='one')   
