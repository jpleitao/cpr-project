# Joaquim Leitão - 2011150072
# 2016/2017 School Year
# Doctoral Program in Information Science and Technology - Connectivity and Pattern Recognition
# Project

source('/media/jpleitao/Data/PhD/PDCTI/CPR/cpr-project/TimeSeriesProcess/preprocess/missing_values.R')

# Read data, make copy and get number of rows and columns
data <- read.csv('/media/jpleitao/Data/PhD/PDCTI/CPR/cpr-project/TimeSeriesProcess/data/merged_data.csv', sep=';',
                 header=FALSE)

testArimaMissing(data)
testKalmanFilterMissing(data)
data <- fillMissingValuesKalman(data)