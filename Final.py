"""
Class: CS230--Section 7
Name: Matt Bouzakis
Description: (Give a brief description for Exercise name--See
below)
I pledge that I have completed the programming assignment
independently.
I have not copied the code from a student or any source.
I have not given my code to any student.
"""

# import modules
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# create sidebar for navigation
st.sidebar.markdown('''
# Analysis
- [Crashes accross different months](#section-1)
- [WeekDays VS Weekends](#section-2)
- [Crash comparison between Old and Young](#section-3)
- [Top 10 Cities](#section-4)
- [Map](#section-5)
''', unsafe_allow_html=True)

st.title("2017 Motor Vehicle Crash Analysis")


def load_clean_data():
    """ this method load the dataset and clean it"""

    # load data
    data = pd.read_csv("2017_Crashes_10000_sample.csv")
    # keep only useful columns
    imp_data = data[['CRASH_NUMB',
                     'CITY_TOWN_NAME',
                     'CRASH_DATE_TEXT',
                     'CRASH_TIME',
                     'CRASH_HOUR',
                     'CRASH_DATETIME',
                     'WEATH_COND_DESCR',
                     'AMBNT_LIGHT_DESCR',
                     'LON',
                     'LAT',
                     'AGE_DRVR_OLDEST',
                     'AGE_DRVR_YNGST',
                     'CNTY_NAME', ]]
    # change data types
    imp_data['CRASH_DATE_TEXT'] = pd.to_datetime(imp_data['CRASH_DATE_TEXT'])
    imp_data['CRASH_DATETIME'] = pd.to_datetime(imp_data['CRASH_DATETIME'])

    # create new columns
    imp_data['CRASH_MONTH'] = imp_data['CRASH_DATETIME'].dt.month
    imp_data['TIME_OF_DAY'] = pd.to_datetime(imp_data['CRASH_TIME']).dt.hour.apply(
        lambda x: 'Morning' if 5 <= x < 12 else 'Afternoon' if 12 <= x < 17
        else 'Evening' if 17 <= x < 21 else 'Night')

    # calculate null percentage
    null_percentages = (imp_data.isnull().sum() / len(imp_data)) * 100

    # Drop columns where the percentage of null values is less than 5%
    columns_to_drop = null_percentages[null_percentages < 5].index
    imp_data.dropna(subset=columns_to_drop, inplace=True)

    # drop duplicates
    imp_data.drop_duplicates()

    # sort values
    imp_data.sort_values(by="CRASH_DATETIME", inplace=True, ascending=True)

    # handle spelling mistakes
    imp_data.replace({"Not reported": "Unknown", "Other": "Unknown"}, inplace=True)
    return imp_data


def crashes_in_different_time(imp_data, county="All"):

    if county not in [None, "All"]:
        imp_data = imp_data[imp_data["CNTY_NAME"] == county]

    # Aggregate the number of crashes by month
    crashes_by_month = imp_data.groupby('CRASH_MONTH')['CRASH_NUMB'].count().reset_index()

    # Aggregate crashes by month and time of day
    crashes_by_month_time = imp_data.groupby(['CRASH_MONTH', 'TIME_OF_DAY']).size().unstack(fill_value=0).reset_index()

    # Plotting line plots for each time of day
    plt.figure(figsize=(12, 6))
    crashes_by_month_time.sort_values(by="CRASH_MONTH", inplace=True)
    for time in ['Morning', 'Afternoon', 'Evening', 'Night']:
        plt.plot(crashes_by_month_time['CRASH_MONTH'], crashes_by_month_time[time], marker='o', label=time)

    plt.xlabel('Month')
    plt.ylabel('Number of Crashes')
    plt.title('Distribution of Crashes Across Different Times of Day for Each Month')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt


def average_crashes_by_day_time(imp_data):

    imp_data['DAY_OF_WEEK'] = imp_data['CRASH_DATE_TEXT'].dt.dayofweek  # Mon to Fri
    imp_data['WEEKEND'] = (imp_data['DAY_OF_WEEK'] >= 5)  # Saturday (5) and Sunday (6)

    # Filter data for weekdays and weekends
    weekday_data = imp_data[imp_data['WEEKEND'] == False]
    weekend_data = imp_data[imp_data['WEEKEND'] == True]

    # Aggregate the number of crashes by time of day for weekdays and weekends
    crashes_by_time_weekday = (weekday_data.groupby('TIME_OF_DAY')['CRASH_NUMB'].count() / 5).reset_index()
    crashes_by_time_weekend = (weekend_data.groupby('TIME_OF_DAY')['CRASH_NUMB'].count() / 2).reset_index()

    # Plotting
    plt.figure(figsize=(12, 6))

    # Bar plot for average number of crashes by time of day for weekdays

    sns.barplot(x='TIME_OF_DAY', y='CRASH_NUMB', data=crashes_by_time_weekday, color='skyblue', label="Weekday")
    plt.xlabel('Time of Day')
    plt.ylabel('Average Number of Crashes')
    plt.title('Average Crashes by Time of Day')

    # Bar plot for average number of crashes by time of day for weekends
    sns.barplot(x='TIME_OF_DAY', y='CRASH_NUMB', data=crashes_by_time_weekend, color='salmon', label="Weekend")
    plt.xlabel('Time of Day')
    plt.ylabel('Average Number of Crashes')
    plt.title('Average Crashes by Time of Day')

    plt.tight_layout()
    return plt


def crash_dist_of_day(imp_data):

    monthly_data = imp_data[imp_data['CRASH_MONTH'] == 1]

    # Aggregate crashes by time of day for January
    crashes_by_time = monthly_data['TIME_OF_DAY'].value_counts()

    # Plotting pie chart for distribution of crashes by time of day in January
    plt.figure(figsize=(8, 8))
    plt.pie(crashes_by_time, labels=crashes_by_time.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Crashes by Time of Day')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    return plt


def crash_by_age(imp_data):

    # Calculate the total number of crashes
    total_crashes = len(imp_data)

    # Calculate the proportion of crashes for each age group (oldest and youngest)
    oldest = imp_data.groupby('AGE_DRVR_OLDEST')["CRASH_NUMB"].count().reset_index(name='CRASH_COUNT')
    oldest['CRASH_PROPORTION'] = oldest['CRASH_COUNT'] / total_crashes * 100

    youngest = imp_data.groupby('AGE_DRVR_YNGST')["CRASH_NUMB"].count().reset_index(name='CRASH_COUNT')
    youngest['CRASH_PROPORTION'] = youngest['CRASH_COUNT'] / total_crashes * 100

    # Plotting the proportion of crashes involving drivers of different age groups
    plt.figure(figsize=(10, 6))

    # Plotting the bar chart for oldest drivers
    plt.bar(oldest['AGE_DRVR_OLDEST'], oldest['CRASH_PROPORTION'], color='skyblue', label='Oldest Driver')

    # Plotting the bar chart for youngest drivers
    plt.bar(youngest['AGE_DRVR_YNGST'], youngest['CRASH_PROPORTION'], color='orange', label='Youngest Driver',
            alpha=0.8)

    plt.xlabel('Driver Age Group')
    plt.ylabel('Proportion of Crashes (%)')
    plt.title('Proportion of Crashes Involving Drivers of Different Age Groups')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    return plt


def top_cities(imp_data, how, month=None, ):

    if month:
        months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                  'Nov': 11, 'Dec': 12}

        imp_data = imp_data[imp_data["CRASH_MONTH"] == months[month]]

    crashes_by_city = imp_data.groupby('CITY_TOWN_NAME').size().reset_index(name='CRASH_COUNT')

    if how == "high":
        # Sorting the cities or towns based on the crash count in descending order
        crashes_by_city_sorted = crashes_by_city.sort_values(by='CRASH_COUNT', ascending=False)
    else:
        crashes_by_city_sorted = crashes_by_city.sort_values(by='CRASH_COUNT', ascending=True)
    plt.figure(figsize=(10, 6))

    # Plotting the bar chart
    plt.barh(crashes_by_city_sorted['CITY_TOWN_NAME'].head(10), crashes_by_city_sorted['CRASH_COUNT'].head(10),
             color='skyblue')

    # Adding labels and title
    plt.xlabel('Number of Crashes')
    plt.ylabel('City or Town')
    if how == "high":
        plt.title('Top 10 Cities/Towns with the Highest Number of Crashes')
    else:
        plt.title('Top 10 Cities/Towns with the Least Number of Crashes')

    # Displaying the plot
    plt.gca().invert_yaxis()  # Invert y-axis to have the city with the most crashes on top
    return plt


clean_data = load_clean_data()

st.header('Section 1')
st.header("Distribution of Crashes across different Months")
county_names = ["All"] + list(clean_data["CNTY_NAME"].unique())
option = st.selectbox(
    "Do you want to see distribution for any particular county?",
    county_names,
    index=None,
    placeholder="Select county name....",
)
crashes_in_different_time_fig = crashes_in_different_time(clean_data, option)
st.pyplot(crashes_in_different_time_fig)

st.header("Section 2")
st.header('WeekDays Vs Weekends')
average_crashes_by_day_time_fig = average_crashes_by_day_time(clean_data)
st.pyplot(crashes_in_different_time_fig)

on = st.toggle('Do you want to see percentage of crashes in different times of day?')
if on:
    st.pyplot(crash_dist_of_day(clean_data))

st.header("Section 3")
st.header('Crash comparison between Old and Young')
crash_by_age_fig = crash_by_age(clean_data)
st.pyplot(crash_by_age_fig)

st.header("Section 4")
st.header('Top 10 Cities/Towns with the Highest Number of Crashes')
month = st.select_slider(
    'Filter by Months',
    options=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
print(month)
top_high_cities_fig = top_cities(clean_data, "high", month)
st.pyplot(top_high_cities_fig)

st.header('Top 10 Cities/Towns with the Lowest Number of Crashes')
top_low_cities_fig = top_cities(clean_data, "low", month)
st.pyplot(top_low_cities_fig)

st.header("Section 6")
st.header('Crashes Map')
chart_data = clean_data[["LON", "LAT"]]
chart_data["colors"] = np.random.rand(9181, 4).tolist()
chart_data["size"] = [np.random.randn()*100 for x in range(9181)]
st.map(chart_data, latitude="LAT", longitude="LON", color='colors', zoom=10, size="size")
