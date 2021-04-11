import numpy as np
from scipy.stats import multivariate_normal as mvn
import pdb

np.random.seed(202)

COVARS_MIN = 1e-5
WEIGHT_MIN = 1e-10
TMAT_MIN   = 1e-10

def getEV(model, X):
    states = hmm.viterbi(X)
    next_state = (model.tmat[states[-1]]).argmax()
    print("x",X)
    print("Viterbi",states)
    print("Next",next_state)
    sm = 0
    #for m in range(model.tmat.shape[0]):
    #    sm += sum(hmm.means[m][:,0] * hmm.weights[m] * hmm.tmat[states[-1]][m])
    sm = sum(model.means[next_state][:,0] * model.weights[next_state])
    #assert(np.max(model.means[next_state][:,0]) > sm)

    # This is flawed, but may be the reference
    #sm = sum(model.means[states[-1]][:,0] * model.weights[states[-1]])
    print("Next EV", sm)
    return sm

class Theory():
  def __init__(self, M=5, K=4, D=1):
    self.M = M
    self.K = K
    self.D = D

    pi = np.random.random(M)
    self.pi = pi /sum(pi)

    self.weights = np.ones((M,K)) / K
    self.means = np.array((0.3 - 0.6 * np.random.random_sample((M,K,D)))) 
    self.covars = np.ones((M, K, D))

    tmat = np.random.rand(M,M)
    self.tmat = tmat / tmat.sum(axis=1)[:,None]

    # Vars for EMA
    self.k = 100.0
    self.rho = 2/(self.k+1)

    self._eta = self._etaema

  def genWeights(self, obs):
      T = len(obs)
      weights = np.zeros(T)
      for t in range(T):
          weights[t] = self._eta(t,T)

      return weights

  def calcDensities(self, obs):
    T = len(obs)
    self.bjot = np.zeros((self.M, T))
    self.bjkot = np.zeros((self.M, self.K, T))

    for t in range(T):
      for j in range(self.M):
        for k in range(self.K):
          self.bjkot[j][k][t] = self.weights[j][k] * self._pdf(obs[t], self.means[j][k], self.covars[j][k], 0)
          self.bjot[j][t] += self.bjkot[j][k][t]


  def calcAlpha(self, obs, obsweights):
    T = len(obs)
    self.alpha = np.zeros((T, self.M))
    self.scale = np.ones((T))

    # Initialization
    for x in range(self.M):
      self.alpha[0][x] = self.pi[x]*self.bjot[x][0]
    self.scale[0] = 1 / self.alpha[0].sum()
    self.alpha[0] *= self.scale[0]

    for t in range(1, T):
      for j in range(self.M):
        for i in range(self.M):
          self.alpha[t][j] += self.alpha[t-1][i] * self.tmat[i][j]
        self.alpha[t][j] *= self.bjot[j][t]
      self.scale[t] = 1 / self.alpha[t].sum()
      self.alpha[t] *= self.scale[t] 

    assert(np.all(self.alpha.sum(axis=1)) == 1)

    sm = 0
    for t in range(T):
      sm += obsweights[t] * np.log(self.scale[t])
    return -sm
#    return - np.sum(np.log(self.scale))

  # Probability of being in state i and observing symbols from t+1 to the end
  def calcBeta(self, obs):
    T = len(obs)
    self.beta = np.zeros((T, self.M))

    # Initialization
    self.beta[-1] = self.scale[-1]

    # Induction
    for t in range(T-2, -1, -1):
      for i in range(self.M):
        for j in range(self.M):
          try:
            self.beta[t][i] += self.beta[t+1,j] * self.tmat[i][j] * self.bjot[j,t+1]
          except:
            pass
        self.beta[t][i] *= self.scale[t]

  def calcGamma(self, obs):
    T = len(obs)
    self.gamma = np.zeros((T,self.M))

    for t in range(T):
      alphabeta = 0
      for i in range(self.M):
        alphabeta += self.alpha[t][i] * self.beta[t][i]

      for i in range(self.M):
        self.gamma[t][i] = self.alpha[t][i] * self.beta[t][i] / alphabeta

  def calcMixDensities(self, obs):
    T = len(obs)
    mix = np.zeros((self.M, self.K, T))

    for t in range(T):
      for j in range(self.M):
        v = self.getV(obs[t], self.means[j], self.covars[j])
        for k in range(self.K):
          mix[j][k][t] = self.weights[j,k] * self._pdf(obs[t], self.means[j][k], self.covars[j][k], v)

    return mix

  def calcGammaMix(self, obs):
    T = len(obs)
    self.gamma_mix = np.zeros((T, self.M, self.K))

    for t in range(T):
      for j in range(self.M):
        for k in range(self.K):
          try:
            self.gamma_mix[t,j,k] = self.gamma[t][j] * (self.mix[j,k,t] / self.mix[j,:,t].sum())
          except:
            pass

  def reestimateMeans(self, obs, multiweights):
    T = len(obs)
    means_new = np.zeros((self.M, self.K, self.D))
  
    for m in range(self.M):
      for k in range(self.K):
        numer = 0
        denom = 0
        for t in range(T):
          try:
            numer += self.gamma_mix[t][m][k] * multiweights[t] * obs[t]
            denom += self.gamma_mix[t][m][k] * multiweights[t]
          except:
            pass
        means_new[m][k] = numer/denom
    return means_new

  def reestimateCovars(self, obs, multiweights, means, min_cov=COVARS_MIN, prior_cov=1e-5):
    T = len(obs)
    covars_new = np.zeros((self.M, self.K, self.D))
  
    for m in range(self.M):
      for k in range(self.K):
        numer = 0
        denom = 0
        for t in range(T):
          try:
            numer += self.gamma_mix[t][m][k] * multiweights[t] * (obs[t] - means[m][k])** 2 + min_cov * prior_cov
            denom += self.gamma_mix[t][m][k] * multiweights[t] + prior_cov
          except:
            pass
        covars_new[m][k] = numer/denom
    return covars_new

  def reestimateTmat(self, obs, multiweights, tmat_min=TMAT_MIN):
    T = len(obs)
    tmat_new = np.zeros((self.M, self.M))
  
    for i in range(self.M):
      for j in range(self.M):
        numer = 0
        denom = 0
        for t in range(T-1):
          try:
            numer += self.alpha[t][i] * self.tmat[i][j] * self.bjot[j][t+1] * self.beta[t+1][j] * multiweights[t]
            denom += self.gamma[t][i] * multiweights[t]
          except:
            pass
        tmat_new[i][j] = max(numer/denom, TMAT_MIN)
    assert(np.all(tmat_new.sum(axis=1)) == 1)
    return tmat_new

  def reestimateWeights(self, obs, multiweights):
    T = len(obs)
    w_new = np.zeros((self.M, self.K))
  
    for m in range(self.M):
      for k in range(self.K):
        numer = 0
        denom = 0
        for t in range(T):
          try:
            numer += self.gamma_mix[t][m][k] * multiweights[t]
            denom += self.gamma_mix[t][m][:].sum() * multiweights[t]
          except:
            pass
        w_new[m][k] = max(numer/denom, WEIGHT_MIN)
    return w_new

  def viterbi(self, x):
    # returns the most likely state sequence given observed sequence x
    # using the Viterbi algorithm
    T = len(x)

    # make the emission matrix B
    logB = np.zeros((self.M, T))
    for j in range(self.M):
        for t in range(T):
            for k in range(self.K):
                p = np.log(self.weights[j][k]) + mvn.logpdf(x[t], self.means[j][k], self.covars[j][k])
                logB[j,t] += p

    # perform Viterbi as usual
    delta = np.zeros((T, self.M))
    psi = np.zeros((T, self.M))

    # smooth pi in case it is 0
    pi = self.pi + 1e-10
    pi /= pi.sum()

    delta[0] = np.log(pi) + logB[:,0]
    for t in range(1, T):
        for j in range(self.M):
            next_delta = delta[t-1] + np.log(self.tmat[:,j])
            delta[t,j] = np.max(next_delta) + logB[j,t]
            psi[t,j] = np.argmax(next_delta)

    # backtrack
    states = np.zeros(T, dtype=np.int32)
    states[T-1] = np.argmax(delta[T-1])
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]
    return states

  def _etaema(self, t, T):
    if t < self.k:
      return (1/self.k)*((1-self.rho)**(T-self.k))

    return self.rho*((1-self.rho)**(T-1-t))

  def getV(self,x,means,covars):
    v = -np.finfo(np.double).max
    for k in range(self.K):
      #covar = np.matrix(np.diag(covars[k]))
      covar = np.matrix(covars[k])
      mean = means[k]
      v1 = -0.5 * np.dot( np.dot((x-mean),covar.I), (x-mean))
      if v1 > v:
        v=v1
    return -v

  def _pdf(self,x,mean,covar,v=0):
    '''
    Gaussian PDF function
    '''
    covar = np.matrix(covar)
    covar_det = np.linalg.det(covar);

    c = (1 / ( (2.0*np.pi)**(float(self.D/2.0)) * (covar_det)**(0.5)))
    pdfval = 0
    try:
      pdfval = c * np.exp(-0.5 * np.dot( np.dot((x-mean),covar.I), (x-mean)) +v)
    except FloatingPointError:
      pass

    return pdfval

  def train(self, obs, iterations=100, tol=1e-5):
    obsweights = self.genWeights(obs)
    prob_old = -np.inf
    for i in range(iterations):
      prob_new = self.trainiter(obs, obsweights)

      print("Iter %s Log prob = old(%f) new(%f)" % ( i, prob_old, prob_new))

      if abs(prob_new - prob_old) < tol:
        break
      prob_old = prob_new

  def trainiter(self, multiobs, obsweights):
      self.calcDensities(multiobs)
      prob = self.calcAlpha(multiobs, obsweights)
      self.calcBeta(multiobs)
      self.calcGamma(multiobs)
      self.mix = self.calcMixDensities(multiobs)
      self.calcGammaMix(multiobs)

      tmat = self.reestimateTmat(multiobs, obsweights)
      weights = self.reestimateWeights(multiobs, obsweights)
      means = self.reestimateMeans(multiobs, obsweights)
      covars = self.reestimateCovars(multiobs, obsweights, means)

      self.weights = weights
      self.tmat = tmat
      self.means = means
      self.covars = covars
      self.pi = self.gamma[0]
      return prob

#import testdata
#fullobs = np.array(testdata.testfunc(0,400))
import pandas as pd
spx = pd.read_csv('gspc.csv').sort_values(by='Date')
sdiff = spx['Adj Close'].pct_change()[1:]
fullobs = sdiff.to_numpy()

hmm = Theory()

prediction = -np.inf
window = 100
pos = 0
neg = 0
postot = 0
negtot = 0
myret = 1
indexret = 1

for x in range(400-window):
  seq = fullobs[0:x+window]
  #seq = fullobs[x:x+window]
  hmm.train(seq, 100)
  print(hmm.means)
  print(hmm.covars)
  if x > 0:
    indexret *= (1 + seq[-1])
    if seq[-1] < 0:
      negtot += 1
      # prediction and actual both negative
      if nextev < 0:
        neg += 1
        myret *= (1 - seq[-1])
      else:
        myret *= (1 + seq[-1])
    elif seq[-1] >= 0:
      postot += 1
      if nextev >= 0:
        pos += 1
        myret *= (1 + seq[-1])
      else:
        myret *= (1 - seq[-1])
    print("obs ", seq[-1])
    print("pos %d/%d neg %d/%d pred=%s actual=%s" % (pos, postot, neg, negtot, nextev, np.array(seq[-1])))
    print("ret index=%f mine=%f" % (indexret, myret))
  nextev = getEV(hmm, seq[-window:])
pdb.set_trace()
