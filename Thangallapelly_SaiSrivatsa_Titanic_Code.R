# Load required libraries for data analysis and visualization
library(dplyr)
library(ggplot2)
library(ggcorrplot)
library(plotly)
library(shiny)
library(lubridate)
library(caret)

# Step 1: Define the Research Question and Hypothesis
# Research Question: Can we predict passenger survival on the Titanic based on various features?
# Hypothesis: Certain features like passenger class, gender, age, and family size may influence a passenger's chances of survival.


# Step 2: Data Pre-processing and Cleaning
# Load the Titanic dataset
library(readr)
Titanic_data <- read_csv("Downloads/Semester 3/Predictive Modeling/Project/Buffer docs/titanic.csv")

# Display the first 10 records with all the columns
head(Titanic_data, 10)

# Output the size of the dataset
cat("Size of Titanic_data: ", length(unlist(Titanic_data)), "\n")

# Output the shape of the dataset (number of rows and columns)
cat("Shape of Titanic_data: ", nrow(Titanic_data), " rows and ", ncol(Titanic_data), " columns\n")

# Check the data type of each column
column_data_types <- sapply(Titanic_data, class)
print(column_data_types)

# Step 3: Exploratory Data Analysis (EDA)

# Identify categorical and numerical features
categorical_features <- c()
numerical_features <- c()

for (feature in colnames(Titanic_data)) {
  if (is.character(Titanic_data[[feature]])) {
    categorical_features <- c(categorical_features, feature)
  } else if (is.numeric(Titanic_data[[feature]])) {
    numerical_features <- c(numerical_features, feature)
  }
}

# Print the lists of categorical and numerical features
cat("Categorical Features: ", categorical_features, "\n")
cat("Numerical Features: ", numerical_features, "\n")

# Create a bar plot to visualize categorical vs. numerical variable count
category_names <- c('Categorical Variables', 'Numerical Variables')
category_counts <- c(length(categorical_features), length(numerical_features))
data <- data.frame(Category = category_names, Count = category_counts)

bar_plot <- ggplot(data, aes(x = Category, y = Count, label = Count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  geom_text(vjust = -0.5) +
  labs(x = "Category", y = "Count", title = "Categorical vs Numerical Variable Count")

print(bar_plot)

# Initialize a named vector to store null value counts
null_values_count <- numeric(length(colnames(Titanic_data)))

# Loop through the columns in Titanic_data
for (i in 1:length(colnames(Titanic_data))) {
  # Calculate the number of null values in each column and store it in the named vector
  null_values_count[i] <- sum(is.na(Titanic_data[[i]]))
}
names(null_values_count) <- colnames(Titanic_data)
print(null_values_count)

# Replace missing 'Age' values with the median age
median_age <- median(Titanic_data$Age, na.rm = TRUE)
Titanic_data$Age[is.na(Titanic_data$Age)] <- median_age

# Remove rows with missing values in other columns
Titanic_data_clean <- na.omit(Titanic_data)

# Plot a barplot to visualize the null values count in all features
null_values_count <- sapply(Titanic_data, function(x) sum(is.na(x)))
null_data <- data.frame(Column_Names = names(null_values_count), Null_Value_Count = null_values_count)

library(ggplot2)  # Make sure to load ggplot2 library if not already loaded

null_plot <- ggplot(null_data, aes(x = Column_Names, y = Null_Value_Count)) +
  geom_bar(stat = "identity", fill = "violet") +
  geom_text(aes(label = Null_Value_Count), vjust = -0.5) +
  labs(x = 'Column Names', y = 'Null Value Count') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))  

print(null_plot)


# Display summary statistics of the dataset
summary(Titanic_data)

# Plot histograms for numerical predictor variables
features <- c('Pclass', 'Age', 'SibSp', 'Parch', 'Fare')
par(mfrow = c(3, 2))

for (feature in features) {
  hist(Titanic_data[[feature]], main = paste("Histogram of", feature), xlab = feature, col = 'orange', breaks = 20)
}

par(mfrow = c(1, 1))

# Boxplots for numerical predictor variables
features <- c('Age', 'SibSp', 'Parch', 'Fare')
par(mfrow = c(2, 2))

for (i in 1:length(features)) {
  boxplot(Titanic_data[[features[i]]], main = paste("Box Plot of", features[i]), col = 'orange')
}

par(mfrow = c(1, 1))

# Remove features with high cardinality
cardinalities <- Titanic_data %>%
  summarize_all(n_distinct)
high_cardinality_features <- names(cardinalities)[cardinalities > 100]
Titanic_data <- Titanic_data %>%
  select(-high_cardinality_features)
cardinalities
Titanic_data
# Remove outliers from 'Age' using IQR method
features <- c('Age')

handle_outliers_iqr <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  x[x < lower_bound] <- NA
  x[x > upper_bound] <- NA
  return(x)
}

for (feature in features) {
  Titanic_data[[feature]] <- handle_outliers_iqr(Titanic_data[[feature]])
}

Titanic_data <- na.omit(Titanic_data)

# Remove 'Embarked' column
Titanic_data <- subset(Titanic_data, select = -Embarked)

# Display the first 10 records of the cleaned dataset
head(Titanic_data, 10)

# Step 4: Model Building and Training
# Prepare the data for modeling

# Convert categorical features into factors
Titanic_data$Survived <- as.factor(Titanic_data$Survived)
Titanic_data$Pclass <- as.factor(Titanic_data$Pclass)
Titanic_data$Sex <- as.factor(Titanic_data$Sex)

# Split the data into training and test datasets
set.seed(123)
index_of_train <- sample(1:nrow(Titanic_data), nrow(Titanic_data) * 0.7, replace = FALSE)
train_data <- Titanic_data[index_of_train,]
test_data <- Titanic_data[-index_of_train,]

# Train the Logistic Regression model
set.seed(123)
train_control <- trainControl(method = "cv", number = 20)
Logistic_Model <- train(
  Survived ~ Pclass + Sex + Age + SibSp + Parch,
  data = train_data,
  method = "glm",
  trControl = train_control
)

summary(Logistic_Model)
# Train the Random Forest model
set.seed(123)
Random_Forest_Model <- train(
  Survived ~ Pclass + Sex + Age + SibSp + Parch,
  data = train_data,
  method = "rf",
  trControl = train_control
)
summary(Random_Forest_Model)

# Step 5: Model Evaluation and Conclusion
# Evaluate the models
## Futher Analysis and Conclusions

#Variable Relationships
# Compute the correlation matrix
correlation_matrix <- cor(Titanic_data_clean[, c("Survived", "Pclass", "Age", "SibSp", "Parch", "Fare")])
# Create a heatmap of the correlation matrix
ggcorrplot(correlation_matrix, type = "lower", lab = TRUE)

# Predict using the Logistic Regression model
Logistic_Predictions <- predict(Logistic_Model, test_data)
Logistic_Confusion_Matrix <- confusionMatrix(Logistic_Predictions, test_data$Survived)
Logistic_Confusion_Matrix
# Calculate accuracy and kappa for the Logistic Regression model
Logistic_Accuracy <- Logistic_Confusion_Matrix$overall['Accuracy']
Logistic_Kappa <- Logistic_Confusion_Matrix$overall['Kappa']

# Predict using the Random Forest model
Random_Forest_Predictions <- predict(Random_Forest_Model, test_data)
Random_Forest_Confusion_Matrix <- confusionMatrix(Random_Forest_Predictions, test_data$Survived)
Random_Forest_Confusion_Matrix
# Calculate accuracy and kappa for the Random Forest model
Random_Forest_Accuracy <- Random_Forest_Confusion_Matrix$overall['Accuracy']
Random_Forest_Kappa <- Random_Forest_Confusion_Matrix$overall['Kappa']

# Print results and conclusions
cat("Logistic Regression Model Results:\n")
cat("Accuracy: ", Logistic_Accuracy, "\n")
cat("Kappa: ", Logistic_Kappa, "\n")

cat("Random Forest Model Results:\n")
cat("Accuracy: ", Random_Forest_Accuracy, "\n")
cat("Kappa: ", Random_Forest_Kappa, "\n")

# Detailed interpretation of results and insights from EDA

# The dataset was preprocessed to handle missing values, remove outliers, and drop columns with high cardinality.
# EDA revealed that passenger class (Pclass), gender (Sex), and age (Age) may influence a passenger's chances of survival.
# The Logistic Regression and Random Forest models were trained and evaluated.
# The Random Forest model performed slightly better in terms of accuracy and kappa.
# Further feature engineering and model tuning can be explored to improve predictions.

# Document the reasoning behind preprocessing steps
# - Missing 'Age' values were replaced with the median age to maintain data integrity and preserve valuable information.
# - Rows with missing values in other columns were removed as these rows couldn't contribute to modeling.
# - Outliers in the 'Age' column were removed using the IQR method to avoid distortion of the models.
# - Columns with high cardinality were removed to simplify modeling and prevent overfitting.

