library(tidyverse)

# --- Read one IR .txt file ---
read_thermal_txt <- function(file) {
  lines <- readLines(file)
  data_start <- which(lines == "[Data]") + 1
  dat <- lines[data_start:length(lines)]
  
  mat <- dat %>%
    str_replace_all(",", ".") %>%
    read.table(text = ., sep = "\t", stringsAsFactors = FALSE)
  
  as.matrix(sapply(mat, as.numeric))
}

# --- Load your file ---
file <- "D:/Masterarbeit/Data/Data_TXT_formatted_aligned/Paradiestal_12-13.08.25/2025-08-12_1110.txt"
img <- read_thermal_txt(file)

# --- Convert matrix to dataframe with numeric indices ---
df <- as.data.frame(as.table(img))
colnames(df) <- c("y", "x", "temp")

# Here x and y are characters → we replace them with numeric indices
df$x <- as.numeric(rep(1:ncol(img), each = nrow(img)))
df$y <- rep(nrow(img):1, times = ncol(img))  # flip Y so image is oriented correctly

# --- Plot ---
ggplot(df, aes(x = x, y = y, fill = temp)) +
  geom_raster() +
  scale_fill_viridis_c(option = "inferno") +
  coord_fixed() +
  labs(
    title = "Thermal Image",
    x = "X (pixels)", y = "Y (pixels)", fill = "°C"
  ) +
  theme_void()