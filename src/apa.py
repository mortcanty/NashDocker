'''
Created on 18.08.2012
Acquisition path analysis game theoretical effectiveness evaluation
Usage:
 python APA(ps,ls,betas,alphas,ws,W,b=10,f=1,a=10,c=100,e=1) 
 ps =    list of acquisition paths. Each path consists 
         of a list of locations/activities
 ls =    list of corresponding path lengths
 betas = list of non-detection probabilities in event of  
        violation and inspection at a location/activity
 alphas = false alarm probability in event of inspection of 
          location/activity where State has behaved legally
 ws =    list of inspection efforts (person-days) for each 
         activity/location
 W  =    available inspection effort (person-days)
 b =     perceived sanction in event of detection relative 
         to the payoff 10 for undetected violation 
         along the shortest path.   
 f =     false alarm resolution cost for any location/activity 
         on same scale as above
 k =     number of locations/activities inspected
 a,c,e = IAEA utilities (-a = loss for detection, 
         -c = loss for undetected violation, 
         -e loss incurred by false alarm)
 All utilities are normalized to the utility 0 for 
 legal behavior (deterrence) for both players     
 Returns equilibria as a list of tuples [(P*,H1*,Q*,H2*), ...]
@author: Mort Canty
'''

from numpy import * 
from itertools import combinations
from nash.nasheq import nashEquilibria
import time
import operator

def undominated(K,Ks):
    for KK in Ks:
        if set(K) <= set(KK):
            return False 
    return True

def APA(ps,ls,betas,alphas,ws,W,b=10,f=1,a=10,c=100,e=1,select='one'):  
    ls = asarray(ls)
    ps = asarray(ps)
    ws = asarray(ws) 
    betas = asarray(betas)
    alphas = asarray(alphas)             
    n = len(ps)
    m = len(betas)
#  pure strategies of Inspectorate    
    Ks = []
    print 'Enumerating feasible undominated inspection strategies ...'
    idx = argsort(ws)
    cs = cumsum(ws[idx])
#  largest possible number of activities   
    kl = 0
    while kl < len(cs) and cs[kl]<W:
        kl += 1        
    print 'Largest inspection strategy has %i activities'%kl       
    for i in range(kl,0,-1):
        cm = combinations(range(m),i)
        while True:
            try:
                K = list(cm.next())
                if (sum(ws[K]) < W) and undominated(K,Ks):
                    Ks.append(K)
            except:
                break    
        print 'length: %i cumulative inspectorate strategies: %i' %(i,len(Ks))               
    mk = len(Ks)
#  path preferences inversely proportional to path lengths    
    ds = 1./ls
#  path preferences in decreasing order beginning with 10    
    ds = 10*ds/ds[0]
#  construct bimatrix
    print 'Building bimatrix...'
    A = zeros((mk,n+1))
    B = zeros((mk,n+1))
    i = 0
    for K in Ks:
        K = set(K)
        for j in range(n+1):
            if j<n: 
                P = set(ps[j])
            else:
                P = set([])    
            beta = 1
            for ell in K & P:
                beta *= betas[ell]
            tmp = 1
            for ell in K - (K & P):
                tmp *= 1 - alphas[ell]
            alpha = 1 - tmp 
            if j < n:         
                A[i,j] = -(c)*beta - (a)*(1-beta) - e*alpha
                B[i,j] = (ds[j])*beta - b*(1-beta) -f*alpha
            else:
                A[i,j] = -e*alpha
                B[i,j] = -f*alpha         
        i += 1
#  solve the game     
    print 'Calling nashEquilibria ...'      
    return (Ks,nashEquilibria(A.tolist(),B.tolist(),select=select))

if __name__ == '__main__':   
#    random.seed(1234)
    betas = random.rand(10)*0.5 
    alphas = zeros(10)+0.05
    ws = random.rand(10)
    W = 1.0
    ps = [[0,1,2],[1,3,6],[2,4,5,7],[0,9],[1,4,8]]
    ls = [5,4,3,2,1]
    Ks, eqs = APA(ps,ls,betas,alphas,ws,W,select='all')
    print 'Found %i equilibria'%len(eqs)
    k = 1
    for eq in eqs:
        print 'equlibrium %i --------------'%k
        P = eq[0]
        H1 = eq[1]
        Q = array(eq[2])*100
        H2 = eq[3]
        print 'P:'
        for i in range(len(Ks)):
            if P[i] != 0:
                print Ks[i], P[i]
        print 'Q:'
        print array(map(round,Q))/100.
        print 'H1 = %f, H2 = %f'%(H1,H2)
        k += 1
