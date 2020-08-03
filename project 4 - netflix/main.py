import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

for i in range(4):
    for j in range(5):
        initial_mixture, post = common.init(X,i+1,j)
        #M, L, cost_final = kmeans.run(X, initial_mixture, post)
        #title = "K means for K "+str(i+1)+" seed " +str(j)
        #common.plot(X, M, L, title)
        #print("For K "+ str(i+1) + " seed " + str(j) +" cost is " + str(cost_final))
        
        M, L, likelihood = naive_em.run(X, initial_mixture, post)
        bic = common.bic(X, M, likelihood)
        
        title = "EM for K "+str(i+1)+" seed " +str(j)
        common.plot(X, M, L, title)
        print("For K "+ str(i+1) + " seed " + str(j) +" likelihood is " + str(likelihood) + " bic is " + str(bic))
        

        

