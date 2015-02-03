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
init_printing()

solvers.options['show_progress'] = False

def fract(n,asfloat=True,den=20):
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
            


def leaving(T,s):
    
    def lexLess(X,Y):
        i = np.where(X-Y)[0][0]
        return X[i]<Y[i]
    
    def lexMax(S):
        if len(S) ==1:    # singleton
            return S[0][0]
        else:             # degenerate game   
            Vs = [ T.row(S[i])/(T.row(S[i])[s+1]) for i in range(len(S)) ]
            posmax = S[0]  
            Vmax = Vs[0]
            ell = 1
            while ell <= len(S):
                if lexLess(Vmax,Vs[ell]):
                    posmax = S[ell]
                else:
                    Vmax = Vs[ell]
            ell += 1
        return posmax
    
    E = -T.T.row(0)
    U =  T.T.row(s+1)
    t = [E[j]/U[j] if U[j] > 0 else oo for j in range(T.rows)] 
    tmin = min(t)
    So = np.where(np.array(t)==tmin)
    return  lexMax(So)

def pivot(i,j,r,s,T):
    if i == r:
        return T[r,j]/T[r,s+1]
    else:
        return T[i,j] - T[r,j]*T[i,s+1]/T[r,s+1]

def nashEquilibria(A,B=None,select='all'): 
    ''' Calculate all extreme equilibria or one
        equilibrium of a bimatrix game.
        A and B are list of lists matrices.'''
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
                aij = fract(A[i][j],asfloat=False,den=100)
                g += str(aij)+' '
            g += '\n'
        g += '\n' 
        for i in range(m):
            for j in range (n):
                bij = fract(B[i][j],asfloat=False,den=100)
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
#  Lemke Howson algorithm for one equilibrium. We use sympy matrices here.
        a = Matrix(A)
        b = Matrix(B)
        smallest = min(min(a),min(b))
        A = a - ones(a.shape)*(smallest - 1)
        B = b - ones(a.shape)*(smallest - 1)
        m,n = A.shape
        k = m + n
        M = zeros(m,m).col_insert(m,A).row_insert(m,B.T.col_insert(m,zeros(n,n)))
        C = M.col_insert(k,ones(k))
        T = C.col_insert(0,-ones(k,1))
        beta = range(k,2*k)
        s = 0
        br = -1
    #  complementary pivoting
        while (br != 0) and (br != k):
            tmp = zeros(k,2*k+1)
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
        return (P,H1,Q,H2)

    
if __name__ == '__main__':
#  Tests    
#  Zero sum game    
    A = [[4.0,-4.0,1.0],[-4.0,4.0,-2.0]]
    print nashEquilibria(A)
#  von Stengel's game    
    A = [[9504,-660,19976,-20526,1776,-8976],[-111771,31680,-130944,168124,-8514,52764], \
         [397584,-113850,451176,-586476,29216,-178761],[171204,-45936,208626,-263076,14124,-84436], \
         [1303104,-453420,1227336,-1718376,72336,-461736],[737154,-227040,774576,-1039236,48081,-300036]]        
    B = [[72336,48081,29216,14124,1776,-8514],[-461736,-300036,-178761,-84436,-8976,52764], \
         [1227336,774576,451176,208626,19976,-130944],[-1718376,-1039236,-586476,-263076,-20526,168124], \
         [1303104,737154,397584,171204,9504,-111771],[-453420,-227040,-113850,-45936,-660,31680]]   
    print len(nashEquilibria(A,B,select='perfect'))
    print nashEquilibria(A,B,select='one')
    print '--------'        
         
#  Winkels' game
    A = [[1,3],[1,3],[3,1],[3,1],[2.5,2.5],[2.5,2.5]]  
    B = [[1,2],[0,-1],[-2,2],[4,-1],[-1,6],[6,-1]] 
    result =  nashEquilibria(A,B,select='all')
    for i in range(len(result)):
        print result[i]
    print '--------'
      
#  Spectrum auction
    A = [[2.5,0,0,0],[4,2,0,0],[3,3,1.5,0],[2,2,2,1],[1,1,1,1]]
    B = [[1,1,0,-1],[0,0.5,0,-1],[0,0,0,-1],[0,0,0,-0.5],[0,0,0,0]]
    print nashEquilibria(A,B,select='perfect')
    print '--------'
    
#  Dollar sharing
    A = [[3,3,3,3,0,3,3,3,0,0,0,3,0,0,0,0], \
         [2,2,2,0,2,2,0,0,2,2,0,0,2,0,0,0], \
         [1,1,0,1,1,0,1,0,1,0,1,0,0,1,0,0], \
         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    B = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], \
         [1,1,1,0,1,1,0,0,1,1,0,0,1,0,0,0], \
         [2,2,0,2,2,0,2,0,2,0,2,0,0,2,0,0], \
         [3,0,3,3,3,0,0,3,0,3,3,0,0,0,3,0]]  
    print len(nashEquilibria(A,B,select='all'))  
    eqs = nashEquilibria(A,B,select='perfect')
    for eq in eqs:
        print eq
    

   