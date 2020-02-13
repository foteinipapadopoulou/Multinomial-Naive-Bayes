from train import calltrain
from testmain import call
from test import calltest

if __name__ == '__main__':
    number = 6
    path_train = "enron/enron"+str(number)+"/all"
    path_test_from_test = "testing_data/from_test/enron"+str(number)
    path_test_from_train = "testing_data/from_train/enron"+str(number)
    print("###Train our algorithm###")
    calltrain(path_train)
    print("-----------Testing Data--------------")
    flag = False #means we have test data
    print("-----------Our code------------------")
    calltest(path_test_from_test, number, flag)
    print("-----------Compared code------------------")
    call(path_train, path_test_from_test, number, flag)
    print("-----------Training Data--------------")
    flag = True
    print("-----------Our code------------------")
    calltest(path_test_from_train, number, flag)
    print("-----------Compared code------------------")
    call(path_train, path_test_from_train, number, flag)
