# setwd("C:/Users/farri/OneDrive/Documents/PA/RRP")

# LIBRARIES=====================================================================
library(tidyverse)
library(vroom)
library(tidymodels)
library(discrim)
library(glmnet)
library(rpart)
library(ranger)
library(stacks)
library(recipes)
library(embed) 
library(naivebayes)
library(kknn)
library(themis) # for smote
library(timetk)
library(forecast)
library(modeltime) 
library(bonsai)

test <- vroom("test.csv")
train <- vroom("train.csv")
sampleSubmission <- vroom("sampleSubmission.csv")

# Recipe========================================================================

test$Type[test$Type == "MB"] <- "DT"

recipe2 <- recipe(revenue ~ ., data = train) %>%
  step_mutate(`Open Date` = mdy(`Open Date`),
              Year = year(`Open Date`),
              Month = month(`Open Date`),
              Day = day(`Open Date`),
              DayOfWeek = wday(`Open Date`, label = TRUE)) %>%
  step_rm(`Open Date`) %>%
  step_other(City, threshold = .03) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>% 
  step_rm(Id)

view(bake(prep(recipe2), new_data = train))

# KNN===========================================================================

knn_model <- nearest_neighbor(neighbors=tune()) %>% 
  set_mode("regression") %>%
  set_engine("kknn") 

knn_wf <- workflow() %>% add_recipe(recipe2) %>% add_model(knn_model)

knngrid <- grid_regular(neighbors(), levels = 5)

folds <- vfold_cv(train, v = 6, repeats=1)

CV_results <- knn_wf %>% tune_grid(resamples=folds, grid=knngrid, metrics=metric_set(rmse))

CV_results %>% select_best("rmse")

knn_model <- nearest_neighbor(neighbors=10) %>% 
  set_mode("regression") %>%
  set_engine("kknn")

knn_wf <- workflow() %>% add_recipe(recipe2) %>% add_model(knn_model) %>% fit(data=train) 

knn <- predict(knn_wf, new_data = test) %>% 
  bind_cols(., test) %>% 
  select(Id, .pred) %>% 
  rename(Prediction=.pred)

vroom_write(x=knn, file="./KNN.csv", delim=",")

# Penalized Regression==========================================================

pen_model <- linear_reg(penalty=tune(), mixture=tune()) %>%
  set_engine("glmnet")

pen_wf <- workflow() %>%
add_recipe(recipe2) %>%
add_model(pen_model)

Penalized_Tuning_Grid <- grid_regular(penalty(), mixture(), levels = 5)

folds <- vfold_cv(train, v = 6, repeats=1)

CV_results <- pen_wf %>% tune_grid(resamples=folds, grid=Penalized_Tuning_Grid, metrics=metric_set(rmse))

CV_results %>% select_best("rmse")

pen_model <- linear_reg(penalty=0.0000000001, mixture=0) %>%
  set_engine("glmnet")

pen_wf <- workflow() %>% add_recipe(recipe2) %>% add_model(pen_model) %>% fit(data=train)

pen <- predict(pen_wf, new_data = test) %>% 
  bind_cols(., test) %>% 
  select(Id, .pred) %>% 
  rename(Prediction=.pred)

vroom_write(x=pen, file="./Pen.csv", delim=",")

# Random Forest=================================================================

FOREST_MODEL <- rand_forest(mtry= tune(),
                            min_n=tune(),
                            trees=1000) %>% 
  set_engine("ranger") %>%
  set_mode("regression")

FOREST_GRID <- grid_regular(mtry(range = c(1,42)), min_n(), levels = 5) 

TREEWORKFLOW2 <- workflow() %>% 
  add_recipe(recipe2) %>% add_model(FOREST_MODEL) %>% fit(data=train) 

folds <- vfold_cv(train, v = 6, repeats=1)

CV_results <- TREEWORKFLOW2 %>%
  tune_grid(resamples=folds, grid=FOREST_GRID, metrics=metric_set(rmse))

CV_results %>% select_best("rmse")

FOREST_MODEL <- rand_forest(mtry = 1,
                            min_n=2,
                            trees=1000) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

TREEWORKFLOW <- workflow() %>%
  add_recipe(recipe2) %>% add_model(FOREST_MODEL) %>% fit(data=train) 

ForestAnswer <- predict(TREEWORKFLOW, new_data = test) %>% 
  bind_cols(., test) %>% 
  select(Id, .pred) %>% 
  rename(Prediction=.pred)

vroom_write(x=ForestAnswer, file="./ForestAnswer.csv", delim=",")














