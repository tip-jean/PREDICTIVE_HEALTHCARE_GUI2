import matplotlib.pyplot as plt
import pandas as pd
from disease_preprocess import count_cases
from disease_outbreak import detect_outbreak_per_day, predict_future_outbreaks
from disease_detect import check_symptoms

# so eto muna kunwari gui dedetect ng disease, after non maa-add siya sa 'dataset/sample_user_data.csv'
# accessible yung last entry ng user sa `latest_entry` variable if ever na need
latest_entry = check_symptoms()

# let's say na lang na may master list na tayo ng entry ng users of their symptoms and kasama na dun yung predicted disease
# bale manggagaling 'to sa user input ng symptoms nila
df = pd.read_csv('dataset/sample_user_data.csv')

# for every entry, ippredict yung disease, then isasave sa list sa gantong format ng csv
disease_counts = count_cases(df)

print('Disease counts:\n', disease_counts)

# then let's say overtime malaki na yung dataset natin, pag naglalagay sila ng bagong entry, pede na
# tayo gumawa ng detection ng outbreaks, based sa average cases per prognosis per day
# (or not necessarily namang malaki na yung dataset, basta may historical data tayo ng mga cases at
# naga-add sila ng entries ganun)
detected_outbreaks = detect_outbreak_per_day(disease_counts)

print('\nDetected outbreaks:\n', detected_outbreaks)

# then pede na rin gumawa ng forecast ng disease counts and outbreaks based sa historical data
# let's say, for the next 7 days
forecasted_outbreaks = predict_future_outbreaks(disease_counts, days=7)

print('\nForecasted outbreaks:\n', forecasted_outbreaks)

# Determine the split date between historical and forecasted data
split_date = detected_outbreaks['date'].max()

# Add 'source' column to forecasted_outbreaks based on date
forecasted_outbreaks['source'] = forecasted_outbreaks['date'].apply(
    lambda d: 'historical' if pd.to_datetime(
        d) <= pd.to_datetime(split_date) else 'forecasted'
)

total_outbreaks = forecasted_outbreaks.copy()

# Ensure date is datetime
total_outbreaks['date'] = pd.to_datetime(total_outbreaks['date'])

# Plot for each disease
for disease in total_outbreaks['prognosis'].unique():
    disease_data = total_outbreaks[total_outbreaks['prognosis'] == disease]
    plt.figure(figsize=(10, 5))
    # Plot historical
    hist = disease_data[disease_data['source'] == 'historical']
    plt.plot(hist['date'], hist['cases'],
             label='Historical', marker='o', color='blue')
    # Plot forecasted
    forecast = disease_data[disease_data['source'] == 'forecasted']
    if not forecast.empty:
        plt.plot(forecast['date'], forecast['cases'],
                 label='Forecasted', marker='o', color='orange')
    # Mark outbreaks
    outbreak_dates = disease_data[disease_data['outbreak'] == 1]['date']
    outbreak_cases = disease_data[disease_data['outbreak'] == 1]['cases']
    plt.scatter(outbreak_dates, outbreak_cases,
                color='red', label='Outbreak', zorder=5)
    plt.title(f'Disease Cases and Outbreaks: {disease}')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    plt.tight_layout()
    plt.show()