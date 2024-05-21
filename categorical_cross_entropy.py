#we are using one-hot encoding for the prediction of categorical data .
# one hot encoding marks 1 for the each of categories and 0 for all other .
# for example lets mark , 
#     good - male
#     bad - female
#     better - female 
#     bad -male
#     good - female
# to encode this we use one hot encoding 
# good  bad better male female
# 1       0   0       1   0
# 0       1   0       0   1
# 0       0   1       0   1
# 0       1   0       1   0
# 1       0   0       0   1
import math             
# we use this technique to avoid any baisness due to conversion of categorical to numerical data but this increases the complexity and time for training
predicted=[0.7, 0.2 ,0.1]
target=[1,0,0]
# loss=-(sum([traget1*log(predicted1)]))
# this is categorical cross entropy
loss1=-(math.log(predicted[0])*target[0]+
       math.log(predicted[1])*target[1]+
       math.log(predicted[2])*target[2])
print(loss1)
loss2= -(math.log(predicted[0])*target[0])
print(loss2)
loss3=-math.log(predicted[0])
print(loss3)



#as the loss increases we are getting wrong answers and as loss decreases we get correct this is evident from the loss of correct and incorrect 
#predicted values


print(f"correct, this is 1st predicted {loss3}")
loss_incorrect_1= -math.log(predicted[1])
print(f"incorrect, this is 2nd predicted {loss_incorrect_1}")
loss_incorrect_2= -math.log(predicted[2])
print(f"incorrect, this is 3rd predicted {loss_incorrect_2}")