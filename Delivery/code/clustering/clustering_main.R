# Joaquim Leitão - 2011150072
# 2016/2017 School Year
# Doctoral Program in Information Science and Technology - Connectivity and Pattern Recognition
# Project

source('/media/jpleitao/Data/PhD/PDCTI/CPR/cpr-project/TimeSeriesProcess/clustering/clustering.R')

# Load data and normalise
imputedDataPath <- '/media/jpleitao/Data/PhD/PDCTI/CPR/cpr-project/TimeSeriesProcess/data/imputed_data.csv'
returnList <- loadImputedFile(imputedDataPath)
time <- returnList$time
readings <- returnList$readings

readings_znormalise <- zNormalise(readings)

# Run TADPole clustering
tadCluster(readings_znormalise)
