# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 21:52:06 2021

@author: Karthik
"""

import numpy as np


x=np.array([1,0,-1,0,0,1,0,-1])
c=np.array([1,1,-1,-1]).reshape(4,1)

x=x.reshape(4,2)


def class_error_calc(w):
#Part 1
    w=np.array(w)
    h=[]
    class_outputs=[]
    condlist1=[ (x[:,0]>-0.5),(x[:,0]<=-0.5)]
    choicelist1=[1,-1]
    h1=np.select(condlist1,choicelist1)
    print('Overall Error Rate= Total samples - succesrate')
    print((len(c)-np.sum(h1==c.T))/len(c))
    h.append(np.sum((h1!=c.T)*w))
    class_outputs.append(h1)

    condlist2=[ (x[:,0]>-0.5),(x[:,0]<=-0.5)]
    choicelist2=[-1,1]
    h2=np.select(condlist2,choicelist2)
    print((len(c)-np.sum(h2==c.T))/len(c))
    h.append(np.sum((h2!=c.T)*w))
    class_outputs.append(h2)

    condlist3=[ (x[:,0]>0.5),(x[:,0]<=0.5)]
    choicelist3=[1,-1]
    h3=np.select(condlist3,choicelist3)
    print((len(c)-np.sum(h3==c.T))/len(c))
    h.append(np.sum((h3!=c.T)*w))
    class_outputs.append(h3)
    
    return h,class_outputs

# condlist4=[ (x[:,0]>0.5),(x[:,0]<=0.5)]
# choicelist4=[-1,1]
# h4=np.select(condlist4,choicelist4)
# #print((4-np.sum(h4))/4*100)

# condlist5=[ (x[:,1]>-0.5),(x[:,1]<=-0.5)]
# choicelist5=[1,-1]
# h5=np.select(condlist5,choicelist5)
# #print((4-np.sum(h5))/4*100)

# condlist6=[ (x[:,1]>-0.5),(x[:,1]<=-0.5)]
# choicelist6=[-1,1]
# h6=np.select(condlist6,choicelist6)
# #print((4-np.sum(h6))/4*100)

# condlist7=[ (x[:,1]>0.5),(x[:,1]<=0.5)]
# choicelist7=[1,-1]
# h7=np.select(condlist7,choicelist7)
# #print((4-np.sum(h7))/4*100)

# condlist8=[ (x[:,1]>0.5),(x[:,1]<=0.5)]
# choicelist8=[-1,1]
# h8=np.select(condlist8,choicelist8)
# #print((4-np.sum(h8))/4*100)


# #Used for Q1.b Classification Error Q2.a Training Error as well 
# #print('Overall Error Rate= Total samples - succesrate')
# Avg_m=(h1+h2+h3+h4+h5+h6+h7+h8)/8

# condlist_avg=[Avg_m<0,Avg_m==0,Avg_m>0]
# choicelist_avg=[-1,1,1]
# Avg_boost_result=np.select(condlist_avg,choicelist_avg)
# print('THe training error')
# print(np.sum(np.equal(Avg_boost_result,c.reshape(1,4)[0]))/len(c)*100)
# # THe output is


# #print(h1)

# #Used for Q2.b from above we find classifer 2,3,5 and 8gives 
# #best classsfication so using that to classify the updated error rates for bagging algorightm
# Avg_m=(h2+h3+h5+h8)/4

# condlist_avg_1=[Avg_m<0,Avg_m==0,Avg_m>0]
# choicelist_avg_1=[-1,0,1]
# Avg_boost_result_1=np.select(condlist_avg_1,choicelist_avg_1)
# print(np.sum(np.equal(Avg_boost_result_1,c.reshape(1,4)[0]))/len(c)*100)

w=1/len(c)
h_hat_classifier=[]
alpha_list=[]
k_max=2

for i in range(k_max):
    h,class_outputs=class_error_calc(w)
    # Change this parameter if the professor says max value is needed
    h_hat_argmin_error=min(h)
    h_hat_classifier.append(h.index(h_hat_argmin_error))
    h_hat_index=h.index(h_hat_argmin_error)


    if h_hat_argmin_error>0.5:
        exit        
    else:
        alpha=1/2*np.log((1-h_hat_argmin_error)/h_hat_argmin_error)
        w_temp=w*np.exp(-alpha*c.T*class_outputs[h_hat_index])
        w_temp1=w_temp/np.sum(w_temp)
        alpha_list.append(alpha)
        w=w_temp1

print(h)



