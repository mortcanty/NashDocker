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

solvers.options['show_progress'] = False

def fract(n,asfloat=True):
    if asfloat:
        return float(Fraction(str(n)).limit_denominator(100))
    else:
        return str(Fraction(str(n)).limit_denominator(100))

def minimax(A):
    '''Minimax solution of a matrix game. A is a sympy Matrix object.
       Returns minimax strategy of player1 and value of the game.'''    
    m,n = A.shape
    s = min(A)
    A = A-ones(m,n)*(s-1)
    G = A.T.row_insert(n,eye(m)).tolist()
#  convert G to a cvxopt matrix object (column major!!)    
    G = [map(float,G[i]) for i in range(m+n)]
    g = -matrix(G).T
    c = matrix([ 1.0 for i in range(m) ])   
    H = ones(n,1).row_insert(n,zeros(m,1)).tolist()
#  convert to a cvxopt matrix object (column major!!)      
    H = [map(float,H[i]) for i in range(m+n)]
    h = - matrix(H).T
    sol = solvers.lp(c,g,h)
    X = np.array(sol['x'].T)[0]
    v = 1/np.sum(X)
    P = map(fract,(v*X).tolist())
    V = fract(v+s-1)
    return (P,V)


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
        A and B are matrices in list form.'''
    if B is None:
        A = Matrix(A)
        return (minimax(A)+minimax(-A.T))
    if select =='all':   
#  complete vertex enumeration with Avis-Fukuda (lrslib)       
        m = len(A) 
        n = len(A[0])
#      generate rational fraction game string 
        g = str(m)+' '+str(n)+'\n\n'
        for i in range(m):
            for j in range (n):
#                aij = Fraction(A[i][j]).limit_denominator(100)
                aij = fract(A[i][j],asfloat=False)
                g += str(aij)+' '
            g += '\n'
        g += '\n' 
        for i in range(m):
            for j in range (n):
#                bij = Fraction(B[i][j]).limit_denominator(100)
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
    elif select=='one':
#  Lemke Howson algorithm for one equilibrium 
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
#  zero sum game    
    A = [[4,-4,1],[-4,4,-2]]
    print nashEquilibria(A)
# #  von Stengel's game    
#     A = [[9504,-660,19976,-20526,1776,-8976],[-111771,31680,-130944,168124,-8514,52764], \
#          [397584,-113850,451176,-586476,29216,-178761],[171204,-45936,208626,-263076,14124,-84436], \
#          [1303104,-453420,1227336,-1718376,72336,-461736],[737154,-227040,774576,-1039236,48081,-300036]]        
#     B = [[72336,48081,29216,14124,1776,-8514],[-461736,-300036,-178761,-84436,-8976,52764], \
#          [1227336,774576,451176,208626,19976,-130944],[-1718376,-1039236,-586476,-263076,-20526,168124], \
#          [1303104,737154,397584,171204,9504,-111771],[-453420,-227040,-113850,-45936,-660,31680]]   
#     print len(nashEquilibria(A,B,select='all'))
#      
#      
#  Winkels' game
    A = [[1,3],[1,3],[3,1],[3,1],[2.5,2.5],[2.5,2.5]]  
    B = [[1,2],[0,-1],[-2,2],[4,-1],[-1,6],[6,-1]] 
    result =  nashEquilibria(A,B,select='all')
    for i in range(len(result)):
        print result[i]
    print '--------'
    print nashEquilibria(A,B,select='one')      
   