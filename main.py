import NaiveBayes_Image
import LogicRegression_Image
import datetime
# Yuxue Piao
# Student ID : 1214904015
# Project 1
#
print("Start do NaiveBayes analysisng.")
NaiveBayes_Image.doAnalysis()

print("Start do LogicRegressioin analysisng.")
begin = datetime.datetime.now()
LogicRegression_Image.doAnalysis()

print(datetime.datetime.now() - begin)