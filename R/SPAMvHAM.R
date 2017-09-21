install.packages("kernlab")
library(kernlab)
install.packages("dplyr")
library(dplyr)
data(spam)
glimpse(spam)

set.seed(123)
training = rbinom(4601,size=1,prob=0.5)
table(training)

train_spam = spam[training==1,]
test_spam = spam[training==0,]

plot(log10(train_spam$capitalAve + 1) ~ train_spam$type)

install.packages("git")
