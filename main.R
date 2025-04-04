#chargement des librairies

library(recommenderlab)
library(ggplot2)
library(data.table)
library(reshape2)
library(dplyr)
library(tidyr)


# Chargement des donnes
movies <- fread("ml-25m/movies.csv", stringsAsFactors = FALSE)
ratings <- fread("ml-25m/ratings.csv")

#Affichage de la data 
glimpse(movies)
summary(movies)

glimpse(ratings)
summary(ratings)



genres <- c("Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
"Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
"Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western")

#Creer matrice de genre
genre_matrix <- movies %>%
  separate_rows(genres, sep = "\\|") %>%
  mutate(value = 1) %>%
  pivot_wider(names_from = genres, values_from = value, values_fill = 0) %>%
  select(-genres)  # Remove the original genres column

# Verifier que toute la matrice sout remplie 
for (g in genres) {
  if (!g %in% colnames(genre_matrix)) {
    genre_matrix[[g]] <- 0
  }
}

#Matrice des avis comparer au utilisateurs qui l'ont donnee
rating_matrix <- dcast(ratings, userId ~ movieId, value.var = "rating", fill = NA)
rating_matrix <- as.matrix(rating_matrix[, -1]) %>% 
  as("realRatingMatrix")

# 4. Data Exploration and Visualization -----------------------------------

# Movie popularity analysis
movie_views <- colCounts(rating_matrix)
top_movies <- data.frame(
  movieId = names(movie_views),
  views = movie_views,
  stringsAsFactors = FALSE
) %>%
  left_join(movies, by = "movieId") %>%
  arrange(desc(views))

# Visualize top 10 movies
ggplot(top_movies[1:10, ], aes(x = reorder(title, views), y = views)) +
  geom_col(fill = "steelblue") +
  geom_text(aes(label = views), hjust = -0.1) +
  coord_flip() +
  labs(title = "Top 10 Most Rated Movies", x = "Movie Title", y = "Number of Ratings") +
  theme_minimal()

# Rating distribution
rating_dist <- ratings %>%
  group_by(rating) %>%
  summarise(count = n())

ggplot(rating_dist, aes(x = rating, y = count)) +
  geom_col(fill = "steelblue") +
  labs(title = "Distribution of Movie Ratings", x = "Rating", y = "Count") +
  theme_minimal()

#Garde seulement les films avec 50 avis et les utilisateurs qui ont au moins mis 50 avis 
#Le but est de garder seulement les avis interessant
min_movies <- 50
min_users <- 50
filtered_ratings <- rating_matrix[
  rowCounts(rating_matrix) > min_movies,
  colCounts(rating_matrix) > min_users
]

# Normalize the data
normalized_ratings <- normalize(filtered_ratings)

# 6. Model Training and Evaluation ----------------------------------------

set.seed(123)
eval_scheme <- evaluationScheme(
  filtered_ratings,
  method = "split",
  train = 0.8,
  given = 10,
  goodRating = 3.5
)

#Creation du modele via recommenderlab
ibcf_model <- Recommender(
  getData(eval_scheme, "train"),
  method = "IBCF",
  parameter = list(k = 30, method = "cosine")
)

# Make predictions
predictions <- predict(
  ibcf_model,
  getData(eval_scheme, "known"),
  type = "ratings"
)

#Evaluation
error_metrics <- calcPredictionAccuracy(
  predictions,
  getData(eval_scheme, "unknown")
)
print(error_metrics)

#Fonction pour recupere des recommendations
get_recommendations <- function(user_id, n = 10) {
  user_ratings <- filtered_ratings[user_id, ]
  pred <- predict(ibcf_model, user_ratings, n = n)
  
  movie_ids <- as.integer(pred@itemLabels[pred@items[[1]]])
  
  movies %>%
    filter(movieId %in% movie_ids) %>%
    select(title, genres) %>%
    mutate(predicted_rating = pred@ratings[[1]])
}

#Recommendation pour utilisateur 1
user1_recs <- get_recommendations(1)
print(user1_recs)



# Tune parameters using cross-validation
parameters <- list(
  k = c(20, 30, 40),
  method = c("cosine", "pearson")
)

# Evaluate different parameter combinations
results <- evaluate(
  eval_scheme,
  method = "IBCF",
  type = "ratings",
  parameter = parameters
)

# Affichage
plot(results, annotate = TRUE)


best_model <- results %>%
  getConfusionMatrix() %>%
  as.data.frame() %>%
  arrange(desc(TPR)) %>%
  head(1)


# Train final model with best parameters
final_model <- Recommender(
  filtered_ratings,
  method = "IBCF",
  parameter = list(k = best_model$k, method = best_model$method)
)

# Sauvegarde du model pour des utilisations futur
saveRDS(final_model, "movie_recommender.rds")