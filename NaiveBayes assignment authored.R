##---
##title: "Assignment - Naive Bayes DIY"
##author:
##- author: Luc Wolters 
##- reviewer: Ernst Hardeman
##date: 18/03/2022
##---

## Setup
library(tidyverse)
library(caret)
library(tm)
library(stopwords)
library(SnowballC)
library(e1071)

##Business understanding
##Making a model that can identify fake news and apply this to a dataset.

##Data understanding
rawDF <- NB_fakenews

rawDF<- rawDF[-c(1:15000),]

rawDF <- mutate(rawDF, label = recode(label,"1"= "Spam", "0"= "Ham"))
head(rawDF)

rawDF$label <- rawDF$label %>% factor %>% relevel("Spam")
class(rawDF$label)

##Data preperation
rawCorpus <- Corpus(VectorSource(rawDF$text))
inspect(rawCorpus[1:3])

cleanCorpus <- rawCorpus %>% tm_map(tolower) %>% tm_map(removeNumbers)

cleanCorpus <- cleanCorpus %>% tm_map(removeWords, stopwords("en", source = "nltk")) %>% tm_map(removePunctuation)

cleanCorpus <- cleanCorpus %>% tm_map(stripWhitespace)

cleanCorpus <- cleanCorpus %>% tm_map(stemDocument)

tibble(Raw = rawCorpus$content[1:3], Clean = cleanCorpus$content[1:3])

cleanDTM <- cleanCorpus %>% DocumentTermMatrix
inspect(cleanDTM)

trainIndex <- createDataPartition(rawDF$label, p = .75, list = FALSE, times = 1)
head(trainIndex)

trainDF <- rawDF[trainIndex,]
testDF <- rawDF[-trainIndex,]

trainCorpus <- cleanCorpus[trainIndex]
testCorpus <- cleanCorpus[-trainIndex]

trainDTM <- cleanDTM[trainIndex,]
testDTM <- cleanDTM[-trainIndex,]

freqWords <- trainDTM %>% findFreqTerms(5)
trainDTM <- DocumentTermMatrix(trainCorpus, list(dictionary = freqWords))
testDTM <- DocumentTermMatrix(testCorpus, list(dictionary = freqWords))

convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0) %>% factor(levels = c(0,1), labels = c("No", "Yes"))
}

nColsDTM <- dim(trainDTM)[2]

trainDTM <- apply(trainDTM, MARGIN = 2, convert_counts)

testDTM <- apply(testDTM, MARGIN = 2, convert_counts)

head(trainDTM[,1:10])

##Modeling 
nbayesModel <- naiveBayes(trainDTM, trainDF$label, laplace = 1)

predVec <- predict(nbayesModel, testDTM)
confusionMatrix(predVec,testDF$label, positive = "Spam", dnn = c("Prediction", "True"))

##Evaluation and Deployment

