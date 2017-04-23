# Joaquim Leitão - 2011150072
# 2016/2017 School Year
# Doctoral Program in Information Science and Technology - Connectivity and Pattern Recognition
# Project

library(forecast)
library(zoo)

full_days <- c(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
               30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 52, 53, 54, 55, 57, 58, 63, 64,
               65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 83, 84, 85, 88, 89, 90, 91, 92, 93, 94,
               95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 111, 113, 115, 116, 117, 118, 119, 120,
               121, 122, 124, 126, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 141, 142, 145, 146, 147,
               148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159, 160, 161, 162, 167, 168, 169, 170, 171, 172, 175,
               176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 190, 191, 192, 193, 194, 197, 198, 201, 202,
               204, 205, 206, 207, 208, 210, 211, 212, 213, 217, 218, 220, 221, 222, 223, 224, 225, 226, 228, 230, 231,
               232, 235, 236, 238, 239, 240, 241, 243, 245, 247, 248, 249, 251, 255, 256, 260, 265, 266, 267, 273, 275,
               276, 281, 282, 286, 289, 291, 292, 294, 295, 296, 297, 299, 301, 302, 303, 304, 305, 306, 307, 308, 309,
               311, 312, 315, 316, 318, 321, 323, 324, 325, 328, 329, 330, 331, 333, 335, 336, 337, 338, 339, 340, 341,
               343, 344, 345, 346, 347, 349, 350, 352, 353, 354, 355, 359, 361, 362, 363, 364)

positions <- list(c(18, 10, 13, 15, 11), c(23, 14, 15, 18, 12), c(9, 16, 8), c(4, 11), c(2, 13),
                  c(8, 18, 11, 19, 22, 14, 9), c(16, 6, 15, 5, 8, 2, 23, 4, 18, 3), c(8, 13, 21), c(16, 0),
                  c(6, 8, 7, 16, 15, 19, 17, 22, 12, 18, 1), c(2, 7, 14), c(15, 23, 0, 16, 7, 5, 18, 11, 12),
                  c(3, 17, 9, 11), c(7, 23, 8, 6), c(1, 4, 22, 5), c(13, 3), c(21, 17, 13, 0, 23, 11),
                  c(4, 5, 22, 11, 7, 8, 3), c(17, 19, 9, 5, 15, 22, 16, 0, 13, 18, 3), c(6, 23, 8, 12),
                  c(9, 15, 23, 22, 4, 10, 16, 5, 19), c(19, 22, 13, 16, 4), c(16, 10), c(0, 6, 4, 8),
                  c(1, 5, 17, 14, 11, 20), c(5, 16, 10, 19, 4, 17, 1, 9), c(14, 7, 16, 15, 8, 17, 10, 9, 3), c(12, 8),
                  c(0), c(11), c(18, 3), c(7, 19, 3, 12, 2, 11), c(3, 15, 7, 22), c(1, 8, 9, 16, 7, 19), c(0, 6, 14),
                  c(18, 5, 0, 3, 9), c(22, 9, 14, 12, 20), c(18, 16, 3, 22), c(14, 19, 23, 8, 6, 17, 7, 16, 9), c(0, 1),
                  c(19, 0, 6, 17, 14, 5, 10, 3, 13, 4), c(18, 10, 21, 11), c(5, 12, 2, 6, 14, 4, 3, 0, 20), c(13),
                  c(3, 20), c(5, 18, 10, 16, 14, 21, 17, 2), c(19, 8, 15, 4, 2, 13),
                  c(13, 5, 8, 21, 14, 3, 22, 12, 19, 18, 11), c(11, 2, 20, 16, 12, 23, 10, 21, 5, 8, 3),
                  c(13, 11, 3, 9, 19), c(0, 11, 12, 14, 17, 7, 4, 5, 18, 8), c(14, 17), c(6, 13), c(9), c(6),
                  c(3, 17, 1, 19, 14), c(17, 19, 8, 16, 2, 4, 13), c(2, 20, 13, 1, 12, 6, 10, 22), c(0, 21, 19),
                  c(9, 14, 2, 1, 6, 8), c(9, 15, 8, 17, 21, 16, 23), c(16, 3, 18), c(2, 12, 10, 9, 16, 11, 23, 0, 1, 6),
                  c(6, 5, 15, 11, 20, 14, 9, 16, 7, 22), c(9, 10, 23, 14, 16, 19, 4), c(19, 9, 12, 20, 8, 18, 22, 15),
                  c(5), c(20, 13, 14, 4, 12, 6, 15, 10, 3, 19, 17), c(18, 9, 16, 4, 17, 21, 8), c(11, 3, 16),
                  c(17, 21, 6, 14, 3, 15, 0, 1, 7), c(22, 20, 17, 19, 10, 6, 7, 13), c(13), c(8, 17, 3, 16), c(2),
                  c(21), c(0, 4, 9, 8, 7, 20, 16, 13, 12), c(1, 11, 2, 9, 10), c(15, 4, 6, 5, 11), c(22),
                  c(4, 21, 13, 11, 6, 10, 16, 8), c(23, 16, 15, 8, 0, 13, 12, 2, 18), c(21, 2, 6, 9, 1, 17, 5, 23),
                  c(8, 21, 0, 22), c(13, 10, 8, 16), c(19, 13), c(21, 22), c(2, 4, 10, 12, 6, 20, 18, 14),
                  c(18, 1, 11, 4, 2, 8, 23), c(8, 16, 3, 18, 17, 11, 20), c(18, 14, 1), c(10, 21, 8, 12, 6, 16, 4),
                  c(20, 16, 6, 19, 18, 15, 10), c(9, 0, 20, 22, 11), c(7, 5, 8, 23, 0, 21, 13, 16, 11),
                  c(4, 21, 19, 2, 1, 22, 5, 3, 17, 20, 8), c(19, 21, 22, 18, 4, 13, 0, 11),
                  c(4, 11, 10, 13, 5, 22, 18, 9, 21, 1), c(17, 19, 6, 15, 11, 21, 3, 5, 8, 1, 10),
                  c(23, 15, 13, 14, 22, 18, 0, 6, 21, 20), c(13, 16, 10, 4, 0, 6, 9, 2), c(6), c(8, 2, 19, 20),
                  c(15, 21, 23, 5, 22, 20, 9, 7, 1, 16), c(9, 10, 17, 8, 19, 16, 2, 21, 3, 23), c(11, 21),
                  c(11, 3, 0, 5, 10, 6, 22), c(0, 23, 6, 15), c(16, 14, 11, 13), c(13, 18, 0, 9, 23, 16, 15, 1, 7),
                  c(19, 23, 21, 22), c(19, 13, 22, 21, 4, 8, 20, 15, 7), c(13, 12, 1, 5, 9, 11),
                  c(16, 2, 18, 7, 1, 17, 14, 9), c(19, 15, 2, 22, 23, 6), c(6, 10, 16, 14, 3, 2, 17, 11, 1, 5),
                  c(16, 19, 3, 14), c(8, 13, 11, 4, 17, 23, 15, 3, 7), c(12, 1, 20, 22, 13, 17, 0, 10, 23, 5, 7),
                  c(17, 1, 19, 15, 12, 9, 3, 11, 10, 21, 13), c(22, 15, 16, 4, 2, 6, 12), c(18, 15, 16, 4, 8),
                  c(23, 12, 6, 22, 13, 10, 11), c(7, 11, 13, 22, 12, 23), c(14, 13, 23, 8, 15, 7, 16, 21, 4, 9, 0),
                  c(6), c(2), c(8, 17, 3, 0, 2, 23, 11, 5, 22, 16, 19), c(12, 4, 3, 9, 22, 18, 8), c(21, 14),
                  c(4, 9, 10, 16, 12, 21, 11), c(0, 17, 21, 8, 13, 12, 2, 14, 9, 7), c(10),
                  c(17, 18, 3, 11, 5, 20, 8, 6, 16, 14), c(15, 22, 10, 18, 11, 9, 4, 17), c(14),
                  c(18, 14, 12, 10, 7, 5, 9, 19, 11, 20, 4), c(14, 9, 22, 0, 13, 10, 5, 23, 20, 18), c(4, 3, 16, 20),
                  c(3, 20), c(19, 17, 8, 13, 0), c(6, 18, 3), c(3, 17, 18), c(19, 1, 17, 22),
                  c(20, 12, 17, 1, 19, 13, 8), c(19, 11), c(11, 22, 7, 19, 18, 9, 1, 14),
                  c(23, 15, 18, 1, 17, 14, 11, 10), c(8, 23, 4, 17, 14), c(18, 2, 13, 4, 10, 16), c(18, 7, 19), c(16),
                  c(10, 11, 17, 23, 22, 21, 2), c(20, 22, 2, 7, 11, 3, 15), c(4, 19, 6, 7, 18, 22, 8, 21, 1, 16),
                  c(3, 1, 5, 23, 22, 13, 14, 18, 20, 10, 6), c(16, 22, 23, 9, 2, 5), c(6), c(23, 16, 5, 12, 10, 11, 22),
                  c(14, 2, 3, 21, 7, 10, 8, 19, 20), c(10, 5), c(17, 6, 22, 1),
                  c(1, 6, 21, 11, 14, 9, 2, 3, 12, 13, 15), c(22, 18, 7, 15, 6, 23, 4, 5, 19, 1),
                  c(11, 4, 20, 15, 21), c(8, 5, 11, 6, 23, 9, 13, 19, 15, 1, 17), c(14, 4, 17, 3, 1, 15, 8, 20), c(5),
                  c(20, 21, 19, 7, 4, 3, 22, 12, 6), c(8, 16), c(15, 14, 22), c(10, 0, 7, 23, 6),
                  c(19, 12, 18, 8, 21, 1, 4, 6, 16), c(17, 16, 10, 3, 20, 14), c(18, 21, 8, 11, 15, 10), c(21),
                  c(21, 4, 20), c(20, 4, 11, 12), c(5, 3, 9, 0), c(13, 2, 11, 5, 1, 3, 17, 6, 21, 4, 15), c(10),
                  c(1, 8, 10), c(15), c(9, 12, 23, 3, 15, 11, 5, 4, 6, 7), c(16, 20), c(10, 19, 18, 17, 16, 2),
                  c(17, 15), c(18, 4, 16, 8, 20), c(0, 2, 5), c(14, 17, 18), c(12), c(19, 17),
                  c(0, 23, 18, 2, 12, 1, 21, 22, 5, 16), c(3, 11, 14, 2, 19), c(9, 10, 1),
                  c(8, 10, 6, 11, 0, 14, 16, 7, 12, 9, 23), c(13, 12, 8, 21, 4, 9, 1, 2, 19, 10), c(0, 14), c(19),
                  c(19, 16, 12, 13, 3, 2), c(16, 5, 11, 13, 10, 1), c(3), c(23, 9, 2, 7, 10, 19), c(0, 15, 20, 6, 12),
                  c(9, 2), c(13, 1, 7, 23, 17, 3), c(4, 9, 19, 3, 18), c(20, 23, 13, 1, 5, 19, 12),
                  c(1, 0, 7, 22, 21, 9, 15, 13), c(15, 3, 8, 1, 10, 9, 23, 7, 14, 19),
                  c(11, 13, 9, 20, 10, 14, 8, 22, 3, 2), c(5, 16, 15, 12, 13, 6), c(3, 22, 6, 13, 16, 4, 11, 15, 8, 23),
                  c(23, 0, 7, 10, 11, 1, 3), c(15, 7, 9, 11, 18, 1, 20), c(4, 20, 11, 9, 22, 1, 23, 17, 12, 2),
                  c(10, 8, 16, 20, 9, 23, 6, 19, 14, 3), c(1, 15, 10, 20, 23), c(13, 2, 9, 16, 5, 20, 12, 22, 10),
                  c(5, 12, 16), c(4, 20, 15, 3), c(18, 21), c(21, 17, 20, 5, 4, 16, 6),
                  c(7, 15, 0, 5, 21, 4, 17, 13, 19, 1, 14), c(19, 8, 11, 2, 21, 9, 15, 23, 6, 13), c(9, 11, 4, 7),
                  c(18, 11, 6, 14, 0, 19), c(1, 21, 23, 7, 8), c(18, 16, 6, 3, 2, 4, 15, 8, 0), c(2, 8, 7, 22, 5),
                  c(8, 3, 5, 23, 16, 17, 10, 13), c(12, 13, 21, 7, 8, 0), c(15, 10), c(22, 0, 23, 7, 5, 6, 8),
                  c(21, 18), c(19, 0, 5, 1, 3, 18, 11, 8), c(10, 22, 12), c(5, 9, 16, 13, 0, 19, 22, 7, 15, 23, 6),
                  c(13, 18, 2, 8, 11, 0, 6), c(11, 8, 20, 21, 5, 14, 17, 23, 22, 18, 1),
                  c(8, 6, 5, 0, 22, 23, 4, 19, 2, 10, 15), c(10, 15, 8, 11, 7, 9, 21, 4, 12), c(17, 14, 20, 22),
                  c(1, 6, 23, 5, 7, 9, 12), c(9), c(1, 4, 0, 8, 19, 18, 7), c(6, 12), c(15, 12, 9, 18, 19),
                  c(11, 13, 18), c(8, 14, 19, 7, 0, 4, 2, 21, 11, 5), c(7, 9, 2, 10, 13, 18, 0, 17),
                  c(5, 0, 8, 18, 7, 21, 22), c(1, 6, 23, 16, 22), c(8, 17, 15, 10, 13, 14, 3, 12, 9),
                  c(1, 18, 2, 3, 8, 6), c(15, 4, 23, 10, 7), c(8, 10, 14, 0, 23), c(8, 12, 3, 15, 7, 4, 13, 21, 10, 9),
                  c(19, 2, 4, 0, 15, 16, 3, 11, 22, 7, 12), c(13, 3, 2, 0, 19, 4, 8, 1), c(11, 22, 15, 8),
                  c(0, 17, 11, 10, 3, 5), c(18, 17, 5), c(23, 4, 19, 11, 20, 21, 6, 16, 5), c(7))

testArimaMissing <- function(data) {
  data_copy <- data
  rows <- nrow(data)
  cols <- ncol(data)
  
  time <- 0:23
  
  for (i in 1:length(full_days)) {
    current_day <- full_days[i] + 1
    # Position for each day
    positions_day <- positions[[i]]
    
    # Set data to NAN
    for (k in positions_day) {
      data[current_day, k+2] <- NA
    }
    
    x <- data[current_day, 2:cols]
    x <- as.double(x)
    
    x_copy <- as.double(data_copy[current_day, 2:cols])
    
    # Fit ARIMA and perform NAN imputation
    x <- imputeArima(x)
    
    # Save plots to disk
    png(paste('/media/jpleitao/Data/PhD/PDCTI/CPR/cpr-project/TimeSeriesProcess/data/missing_data/arima/', full_days[i],
              '.png', sep=''), width=800, height=600)
    plot(time, x_copy, type='l', col='blue')
    lines(time, x, type='l', col='red')
    dev.off();
    
    rmse <- 0
    result <- x - x_copy
    
    for (temp in result) {
      rmse <- rmse + (temp)^2
    }
    rmse <- sqrt(1/length(positions) * rmse)
    
    current_file_line <- paste('[R-ARIMA]DAY: ', full_days[i], ' RMSE: ', rmse, sep='')
    print(current_file_line)
    cat(current_file_line,
        file="/media/jpleitao/Data/PhD/PDCTI/CPR/cpr-project/TimeSeriesProcess/data/missing_data/arima/output.txt",
        append=TRUE, sep='\n')
  }
}

testKalmanFilterMissing <- function(data) {
  # Fill Missing Values with seasonal Kalman Filter
  data_copy <- data
  rows <- nrow(data)
  cols <- ncol(data)
  
  time <- 0:23
  
  for (i in 1:length(full_days)) {
    current_day <- full_days[i] + 1
    # Position for each day
    positions_day <- positions[[i]]
    
    # Set data to NAN
    for (k in positions_day) {
      data[current_day, k+2] <- NA
    }
    
    x <- data[current_day, 2:cols]
    x <- as.double(x)
    
    # First value may not be missing, so replace it with the mean
    if (is.na(x[1])) {
      x[1] <- mean(x, na.rm=TRUE)
    }
    
    # Convert x to a time series object of one day data, collected every hour
    x <- ts(x, start=c(1, 1), end=c(1, 24), frequency=24)
    x_copy <- as.double(data_copy[current_day, 2:cols])
    
    # Fit Model and perform NAN imputation
    x <- na.StructTS(x)
    
    # Save plots to disk
    png(paste('/media/jpleitao/Data/PhD/PDCTI/CPR/cpr-project/TimeSeriesProcess/data/missing_data/kalman/', full_days[i],
              '.png', sep=''), width=800, height=600)
    plot(time, x_copy, type='l', col='blue')
    lines(time, x, type='l', col='red')
    dev.off();
    
    rmse <- 0
    result <- x - x_copy
    
    for (temp in result) {
      rmse <- rmse + (temp)^2
    }
    rmse <- sqrt(1/length(positions) * rmse)
    
    current_file_line <- paste('[R-KALMAN]DAY: ', full_days[i], ' RMSE: ', rmse, sep='')
    print(current_file_line)
    cat(current_file_line,
        file="/media/jpleitao/Data/PhD/PDCTI/CPR/cpr-project/TimeSeriesProcess/data/missing_data/kalman/output.txt",
        append=TRUE, sep='\n')
  }
}

fillMissingValuesKalman <- function(data) {
  # Start by finding the days with nan values, then for each day find the indexes of the nan values
  rows <- nrow(data)
  cols <- ncol(data)
  
  for (i in 0:(rows-1)) {
    if ( !(i %in% full_days) ) {
      # NAN - Get day data
      day_data <- data[i+1, 2:cols]
      day_data <- as.double(day_data)
      
      # Check if first value is missing
      if (is.na(day_data[1])) {
        day_data[1] <- mean(day_data, na.rm=TRUE)
      }
      day_data <- ts(day_data, start=c(1, 1), end=c(1, 24), frequency=24)
      
      # Fit and impute Kalman
      day_data <- na.StructTS(day_data)
      
      # Assign back to data
      data[i+1, 2:cols] <- day_data
    }
  }
  
  # Save to excel (csv) file
  write.table(data, file='/media/jpleitao/Data/PhD/PDCTI/CPR/cpr-project/TimeSeriesProcess/data/imputed_data.csv',
            row.names=FALSE, col.names=FALSE, sep=';', quote=FALSE)
  
  return (data)
}

imputeArima <- function(x) {
  # Based on the code available in the article:
  # "Comparison of different Methods for Univariate Time Series Imputation in R"
  # by Steffen Moritz, Alexis Sardá, Thomas Bartz-Beielstein, Martin Zaefferer and Jörg Stork (2015)
  fit <- auto.arima(x)
  kal <- KalmanRun(x, fit$model)
  tmp <- which(fit$model$Z == 1)
  id <- ifelse (length(tmp) == 1, tmp[1], tmp[2])
  
  id.na <- which(is.na(x))
  x[id.na] <- kal$states[id.na, id]
  
  return(x)
}