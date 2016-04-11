import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x) # deriv sigmoid

	return 1/(1+np.exp(-x)) # sigmoid function

#input
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

#output
y = np.array([[0],
			[1],
			[1],
			[0]])

#choose weight randomly
np.random.seed(1)
syn0 = 2*np.random.random((3,4)) - 1 #first layer weight value
syn1 = 2*np.random.random((4,1)) - 1 #second layer weight value

for j in xrange(30000):

    l0 = X #first layer
    l1 = nonlin(np.dot(l0,syn0)) #second layer
    l2 = nonlin(np.dot(l1,syn1)) #third layer

    l2_error = y - l2 #calculate error

    if (j% 50000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error))) +"\n"
        #print "syn0:" + str(syn0)+"\n"
        #print "syn1:" + str(syn1)+"\n"
        #print "l1:" + str(l1)+"\n"
        #print "l2:" + str(l2)+"\n"

    ##calculate error
    l2_delta = l2_error*nonlin(l2,deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,deriv=True)

    ##update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
