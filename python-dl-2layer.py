import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

#output
y = np.array([[0,0,1,1]]).T

#choose weight randomly 랜덤으로 생성
np.random.seed(1)
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(800000):

    l0 = X #first layer
    l1 = nonlin(np.dot(l0,syn0)) #second layer

    l1_error = y - l1 #calculate error
    if (iter% 50000) == 0:
        print "Error:" + str(np.mean(np.abs(l1_error))) +"\n"
    l1_delta = l1_error * nonlin(l1,True) #sigmoid function의 미분값 * 에러 = 델타 값 에러 수정 반영률

    syn0 += np.dot(l0.T,l1_delta) #update weight

print "After Training:"
print l1

#activation function, backpropagation #activation function, backpropagation
