# === CONFIG ===
input_folder  <- "D:/Masterarbeit/Data/Data_TXT/Karwendel_12-13.11.25"   # <- set this
output_folder <- "D:/Masterarbeit/Data/Data_TXT_formatted/Karwendel_12-13.11.25"  # <- set this
if (!dir.exists(output_folder)) dir.create(output_folder, recursive = TRUE)
year <- format(Sys.Date(), "%Y")               
start_time <- "13:30"                          
tz <- "Europe/Berlin"                          
# ===============

files <- list.files(input_folder, pattern = "\\.txt$", full.names = TRUE)

# Parse metadata for sorting
meta <- data.frame(
  path = files,
  fname = basename(files),
  stringsAsFactors = FALSE
)

meta$month <- as.integer(substr(meta$fname, 3, 4))
meta$day   <- as.integer(substr(meta$fname, 5, 6))
meta$meas  <- as.integer(substr(meta$fname, nchar(meta$fname)-1, nchar(meta$fname)))

# order files by date (month+day) and measurement number
meta <- meta[order(meta$month, meta$day, meta$meas), ]
rownames(meta) <- NULL

# Build continuous timestamps
start_dt <- as.POSIXct(sprintf("%s-%02d-%02d %s", year, meta$month[1], meta$day[1], start_time), tz = tz)
meta$new_dt <- start_dt + (seq_len(nrow(meta)) - 1) * 10 * 60

# Rename files
for (i in seq_len(nrow(meta))) {
  f <- meta$path[i]
  new_name <- paste0(format(meta$new_dt[i], "%Y-%m-%d_%H%M"), ".txt")
  new_path <- file.path(output_folder, new_name)
  
  # avoid overwriting by appending suffix
  if (file.exists(new_path)) {
    j <- 1
    repeat {
      new_name2 <- paste0(format(meta$new_dt[i], "%Y-%m-%d_%H%M"), sprintf("_%02d.txt", j))
      new_path2 <- file.path(output_folder, new_name2)
      if (!file.exists(new_path2)) { new_path <- new_path2; break }
      j <- j + 1
    }
  }
  
  file.copy(f, new_path, overwrite = FALSE)
  message("Copied: ", basename(f), " -> ", basename(new_path))
}

