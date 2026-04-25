# Inital Data Set Builder pm25_dataset_builder.py
Includes UTC time, date, AirNOW measure, and GEOS-CF PM2.5 forecast (t+1, t+2, t+3)
We pulled monthly data for the LA-Long Beach-Anaheim, CA Urban Area for the 2023 calender year

Example use: 
python pm25_dataset_builder.py \
  --location-name "Los Angeles--Long Beach--Anaheim, CA Urban Area" \
  --start 2023-11-01T00:00:00Z \
  --end 2023-11-30T23:00:00Z \
  --output-csv data/11_23.csv \
  --output-meta-json data/11_23.json

# Combine Data combine_cvs.py (TODO)
For convenience, combine into a single CSV (expecting ~365*24 data points)

# Additional Feature Scripts (TODO)
- Forecasts (physical models, etc)
- Temperature
- Humidity 

