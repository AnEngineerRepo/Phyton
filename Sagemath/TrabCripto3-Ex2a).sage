import os, sys, types, hashlib, binascii, re
import numpy as np

Nbits = 224
class SetUP(object):
    def __init__(self,nbits=Nbits):
        p = next_prime(2^nbits)
        while p %3 != 2:
            p = next_prime(p+2)
        b = next_prime(ZZ.random_element(3,p-3))

# Definir o corpo base e a curva
        K = GF(p)
        E = EllipticCurve(K,[0,b])

        def largest_prime_factor(q):
            if is_prime(q):
                return q
            else:
                (n,_) = list(factor(q))[-1]
            return n

        n = largest_prime_factor(p+1)    # o maior factor primo da cardinalidade da curva
        while True:
            P = E.random_element()
            N = P.order()
            if n == largest_prime_factor(N):
                break

# 2. Falta encontrar o gerador "G".

        if is_prime(N):
            G = P
        else:
            G = P*(N//n)

# Determinar o grau de embebimento para a ordem "n"
#

        if gcd(p-1,n) == 1:
            k = 2
        else:
            k = 1

        Kx = GF(p^k,name='z')
        Ex = EllipticCurve(Kx,[0,b])
        Gx = Ex(G.xy())

        Fx.<X> = Kx[]
        (alpha,_) = Fx(X^2+X+1).roots()[0]

        def phi(P):
            if P == Ex(0):
                return P
            (x,y) = P.xy()
            return Ex(alpha*x , y)

        self.Ex = Ex
        self.Fx = Fx
        self.G1 = Gx
        self.G2 = phi(Gx)
        self.O  = Ex(0)
        self.n  = (self.G1).order()
        self.pairing = lambda P,Q: P.weil_pairing(Q,self.n)
        self.g  = self.pairing(self.G1,self.G2)


class Init(SetUP):
    def __init__(self):
        super(Init,self).__init__()
        self.s  = ZZ.random_element(1,self.n-1)
        self.beta = (self.G1)*(self.s)


class Data(object):
    def __init__(self,arg,Params=None):
        if Params != None and isinstance(Params,SetUP):
            self.params = Params
        else:
            self.params = None
        if isinstance(arg,np.ndarray) and arg.dtype == np.dtype('uint8'):
            self.array = arg
            self.data  = self.array.tobytes()
        elif isinstance(arg, types.StringTypes):
            self.data  = arg
            self.array = np.array(self.data,"c").view(np.uint8)
        elif isinstance(arg, Integer):
            self.data  = os.urandom(arg)
            self.array = np.array(self.data,"c").view(np.uint8)
        elif self.params != None and arg in (self.params).Ex:
            F = ((self.params).Kx).vector_space()
            (x,y) = G.xy() ; L = F(x).list() + F(y).list()
            self.array = np.packbits(reduce(lambda u,v: u+v, [a.lift().bits() for a in L],[]))
            self.data = self.array.tobytes()
        else:
            self.data =  str(arg)
            self.array = np.array(self.data,"c").view(np.uint8)
        self.len = len(self.data)
        self._hash = hashlib.sha256(self.data)

    def __repr__(self):
        return binascii.hexlify(self.data)

    def xor(self,other):
        if not isinstance(other,Data):
            raise TypeError("argument of type %s is not Data" % type(other))
        if self.len < other.len:
            return Data(np.bitwise_xor(self.array,other.array[:self.len]))
        elif other.len < self.len:
            return Data(np.bitwise_xor(self.array[:other.len],other.array))
        else:
            return Data(np.bitwise_xor(self.array,other.array))

    def pair(self,other):
        if not isinstance(other,Data):
            raise TypeError("argument of type %s is not Data" % type(other))
        return Data(self.data + other.data,self)

    def prefix(self,i):
        return Data(self.data[:i])

    def digest(self):
        return self._hash.digest()

    def iH(self):
        return ZZ(self._hash.hexdigest(),16)

    def H(self):
            return Data(self.digest())

    def mac(self,key=None):
        if key == None:
            return self.H()
        else:
            K = Data(key) ; T = K.pair(self)
            return (K.pair(T.H())).H()

    def H2(self,r):           #hash H2
        rr = r.norm().lift()
        res = self
        for i in range(rr+1):
            res = self.iH() % self.params.n
        return res

    def H1(self,key=None):    #hash H1
        if key == None:
            return (self.iH()) % self.params.n # TODO what happens if self.params == None?
        else:
            return (self.mac(key)).H1()

class Extract(Init):

    def __init__(self,pubKey, tipo='BLMQ'):      #geraçao de paramentros para BLMQ (SID)
        super(Extract,self).__init__()
        self.pubkey = pubKey
        self.pub = Data(pubKey,self)
        p = (self.pub).H1()                      
        salt = None
        while True:
                z = Integers(self.n)(p + self.s)
                if z.is_unit():
                    self.priv = {'tipo': 'SK' , 'salt': salt, 'SID': (self.G2) * (lift(z^(-1)))} #formula para o "calculo" de SID
                    break
                else:
                    salt = os.urandom(2)
                    p = (self.pub).H1(salt)

    def sign(self,message):                   #funcao que gera a assinatura de uma mensagem
        x  = ZZ.random_element(1,self.n-1)    #geração do número random x
        r = self.g^x                          #computar r=g^x
        dataM = Data(message,self)
        h = dataM.H2(r)                       #h=H2(M,R) 
        S = (x + h)*self.priv['SID']          #S=(x+h)SID
        return (h,S)                          # a assinatura é um tuplo (h,S)

    def verify(self,message,signature):                 # funcao que verifica se a assinatura na mensagem é aceite
        dataM = Data(message,self)
        (h,S) = signature
        h2 = dataM.H2((self.pairing(S,(self.pub).H1()*self.G1+self.beta))*self.g^(-h))  #expressao do verify
        return h2 == h                        #condicao para a verificação ser correta, dar rue

#Teste
Ez = Extract('#id:zzzz@xxxx.net#from:20170101#to:20171231#control:N#read:Y#write:N#', tipo='BLMQ')
signature = Ez.sign("ficheiro")     # gera  a assinatura para o documento
print "Tuplo assinatura: ", signature
print "Teste 1: a assinatura é aceite? "+str(Ez.verify("ficheiro",signature))      #testa se a assinatura é aceite
print "Teste 2: a assinatura é aceite? "+str(Ez.verify("ficheiro diferente",signature))    #testa se a assinatura é aceite por outro ficheiro diferente
