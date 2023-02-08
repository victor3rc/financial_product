"""
Logic to measure user engagement and retention in app.

Date Ranges, queried in Google BigQuery:

user_activity_daily.date:
    Min: 2016-01-22 00:00:00 UTC
    Max: 2018-06-14 00:00:00 UTC

users.account_activation:
    Min: 2016-01-22 13:27:47.493862 UTC
    Max: 2018-06-15 07:59:43.946732 UTC

transactions.date:
    Min: 2017-06-03 11:32:58.998000 UTC
    Max: 2018-06-15 15:34:02.801000 UTC

Patterns in transactions data:
    Drop in unique daily users transacting on Sundays.
    Jump in unique daily users transacting on Mondays.
"""
import csv
import logging

from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Date in which more than 5 daily active users were transacting
TRANSACTIONS_START_DATE = datetime(2017, 11, 1)

# File containing user data
USERS_FILE = 'data/users.csv'

# File containing transactions
TRANSACTIONS_FILE = 'data/transactions.csv'


def load_timestamp_ms(ts: str):
    """
    Load timestamp from a string in the following format, to a datetime object:

    YYYY-MM-DD HH:MM:SS.ssssss ZZZ
    """
    return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f %Z')


def load_timestamp(ts: str):
    """
    Load timestamp from a string in the following format, to a datetime object:

    YYYY-MM-DD HH:MM:SS ZZZ
    """
    return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S %Z')


def load_timestamp_date(ts: str):
    """
    Load timestamp from a string in the following format, to a datetime object:

    YYYY-MM-DD
    """
    return datetime.strptime(ts, '%Y-%m-%d').date()


def date_key(dte: datetime):
    """
    Get the date key for a given datetime. This can be used in dictionaries.
    """
    return f'{dte.year}-{dte.month:02}-{dte.day:02}'


def load_csv_data(filename: str):
    """
    Load data efficiently (memory-wise) from a CSV file using a generator.
    """
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            yield line


def age_category(age: int):
    """
    Return the age category for a given age. Categories are:
    - Under 18 (1)
    - 18 to 24 (2)
    - 25 to 34 (3)
    - 35 to 44 (4)
    - 45 to 54 (5)
    - 55 to 64 (6)
    - 65 and over (7)
    """
    if age < 18:
        return 1
    elif age < 25: 
        return 2
    elif age < 35:
        return 3
    elif age < 45:
        return 4
    elif age < 55:
        return 5
    elif age < 65:
        return 6
    else:
        return 7


def load_user_data(filename: str, target_age_categories: list=None, target_android_pay: bool=None, target_overdraft: bool=None):
    """
    Load users data from a CSV file, and provide age categories overview as well as the average user age.
    
    params:
    :param filename: The name of the file to load users.
    :param age_categories: The age categories the user can belong to.
    :param android_pay: Whether the user has adroid pay activated or not.
    :param overdraft: Whether the user has been offered an overdraft or not.
    """
    users = {}
    age_categories, android_pay, overdraft = defaultdict(int), defaultdict(int), defaultdict(int)
    total_age = 0
    for u in load_csv_data(filename):
        age = int(u['age'])
        total_age += age
        age_cat = age_category(age)
        age_categories[age_cat] += 1
        android_pay[age_cat] += 1 if u['android_pay_activated'] else 0
        overdraft[age_cat] += 0 if not u['offered_overdraft'] else int(u['offered_overdraft'])

        if target_age_categories and age_cat not in target_age_categories:
            # ignore user not in age categories specified.
            continue
        
        if target_android_pay and u['android_pay_activated']:
            # ignore users that have not activated android pay, if indicated to do so.
            continue
        
        if target_overdraft and int(u['offered_overdraft']) != target_overdraft:
            # ignore users which have or have not been offered an overdraft, as indicated.
            continue

        users[u['user_id']] = (load_timestamp_ms(u['account_activation']), age_cat)

    return users, age_categories, total_age/len(users), android_pay, overdraft


def total_users_timeline(users: dict):
    """
    Load active users count over time
    """
    active_users = defaultdict(int)
    for _, user_info in users.items():
        activation, _ = user_info
        active_users[date_key(activation)] += 1
    
    # Min and Max dates retrieved from the database
    start_date = datetime(2016, 1, 22)
    end_date = datetime(2018, 6, 15)
    
    total, total_active_users = 0, {}
    delta = timedelta(days=1)
    while start_date <= end_date:
        if start_date < TRANSACTIONS_START_DATE:
            start_date += delta
            continue

        key = date_key(start_date)
        if key in active_users:
            total += active_users[key]
        
        total_active_users[key] = total    
        start_date += delta

    return total_active_users


def calculate_daily_active_transactions(users: dict):
    """
    Calculate daily unique users transactions.
    """
    logging.info('Loading transactions.')
    timeline = defaultdict(set)
    for transaction in load_csv_data(TRANSACTIONS_FILE):
        if transaction['user_id'] not in users:
            # ignore users that are not being assessed.
            continue

        if not transaction['amount']:  #  or float(transaction['amount']) > 0:
            continue
        
        try:
            # timestamp communicated with and without miliseconds information
            ts = load_timestamp_ms(transaction['timestamp'])
        except ValueError as ex:
            ts = load_timestamp(transaction['timestamp'])

        if ts < TRANSACTIONS_START_DATE:
            continue

        timeline[date_key(ts)].add(transaction['user_id'])

    logging.info('Gathering daily transactions counts.')
    total_users = {}
    for dte in sorted(timeline):
        # calculate on a daily basis how many users come back to the app
        users = timeline[dte]
        total_users[dte] = len(users)

    return total_users


def calculate_daily_active_users(activity_file: str, users: dict):
    """
    Calculate daily active users based on the number of users opening the app.
    """
    timeline = defaultdict(set)
    for activity in load_csv_data(activity_file):
        if activity['user_id'] not in users:
            # ignore users that are not being assessed.
            continue
        if int(activity['app_opens']) == 0:
            continue

        timeline[load_timestamp(activity['date'])].add(activity['user_id'])

    yesterday_users = set()
    daily_gained, daily_dropped, total_users = {}, {}, {}
    for dte in sorted(timeline):
        # calculate on a daily basis how many users come back to the app
        users = timeline[dte]
        daily_gained[dte] = sum([1 for u in users if u not in yesterday_users])
        daily_dropped[dte] = sum([1 for u in yesterday_users if u not in users])
        total_users[dte] = len(users)
        yesterday_users = users

    return total_users, daily_gained, daily_dropped


def calculate_monthly_active_users(activity_file: str, users: dict):
    """
    Calculate daily active users based on the number of users opening the app.
    """
    timeline = defaultdict(set)
    for activity in load_csv_data(activity_file):
        if activity['user_id'] not in users:
            # ignore users that are not being assessed.
            continue
        if int(activity['app_opens']) == 0:
            continue
        
        ts = load_timestamp(activity['date'])
        key = f'{ts.year}-{ts.month:02}'

        timeline[key].add(activity['user_id'])

    last_month_users = set()
    monthly_gained, monthly_dropped, total_users = {}, {}, {}
    for dte in sorted(timeline):
        # calculate on a daily basis how many users come back to the app
        users = timeline[dte]
        monthly_gained[dte] = sum([1 for u in users if u not in last_month_users])
        monthly_dropped[dte] = sum([1 for u in last_month_users if u not in users])
        total_users[dte] = len(users)
        last_month_users = users

    return total_users, monthly_gained, monthly_dropped


def plot_active_users(users_gained: list, users_dropped: list, active_users: list, total_users: list, date_range: list, spacing: int=10):
    """Plot daily active users"""
    df = pd.DataFrame({
        'Users Gained': users_gained,
        'Users Dropped': users_dropped,
        'Active Users': active_users,
        'Total Users': total_users
    })

    df[['Users Gained','Users Dropped']].plot(kind='bar')
    df['Active Users'].plot(kind='line', color='green', label='Active Users')
    df['Total Users'].plot(kind='line', color='yellow', label='Total Users')
    # df['total_users'].plot(secondary_y=True)

    ax = plt.gca()
    fmt = '{x:,}'
    tick = matplotlib.ticker.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)

    labels = []
    for i, label in enumerate(date_range):
        if i%spacing != 0:
            labels.append('')
        else:
            labels.append(label)

    ax.set_xticklabels(labels)
    plt.xticks(rotation=75)
    plt.show()


def save_daily_active_users_to_csv(active_users: dict):
    """
    Save the daily active users information to a CSV file, which is used in the presentation.
    """
    with open('data/daily_active_users.csv', 'w') as f:
        f.write('Date, Daily Active Users\n')
        for dte in sorted(active_users):
            f.write(f'{dte},{active_users[dte]}\n')


def save_daily_active_users_wow_growth_to_csv(active_users: dict):
    """
    Calculate Week-on-Week growth for the daily active users, and save them to a CSV file.
    """
    weeks = defaultdict(int)
    for dte in sorted(active_users):
        count = active_users[dte]
        dte = load_timestamp_date(dte)
        weeks[f'{dte.year} - week {dte.isocalendar().week:02}'] += count
    
    with open('data/wow_growth_daily_active_users.csv', 'w') as f:
        f.write('Week, Daily Active Users Growth\n')
        last_week = 0
        for week in sorted(weeks):
            if last_week == 0:
                last_week = weeks[week]
                continue
    
            growth = ((weeks[week]-last_week)/last_week)*100
            f.write(f'{week},{growth:.2f}%\n')
            last_week = weeks[week]


def save_daily_engagement_rate_to_csv(total_users: dict, active_users: dict):
    """
    Calculate Daily Engagement Rate, and save to a CSV file.
    """
    with open('data/der_total.csv', 'w') as f:
        with open('data/der_proportion.csv', 'w') as g:
            f.write('Date, Daily Active Users, Total Users\n')
            g.write('Date, Daily Engagement Rate\n')
            for dte in sorted(active_users):
                der = (active_users[dte]/total_users[dte])*100
                f.write(f'{dte},{active_users[dte]},{total_users[dte]}\n')
                g.write(f'{dte},{der:.2f}%\n')


def save_daily_engagement_rate_per_age_category(rates: dict):
    """
    Save retation rate per age category to a CSV file.
    """
    with open('data/users_retention_per_age_category.csv', 'w') as f:
        f.write('Date, 18 to 24, 25 to 34, 35 to 44, 45 to 54, 55 to 64, 65 and Over\n')
        for dte in sorted(rates[2]):
            f.write(
                f'{dte},{rates[2][dte]:.2f}%,{rates[3][dte]:.2f}%,{rates[4][dte]:.2f}%,{rates[5][dte]:.2f}%,{rates[6][dte]:.2f}%,{rates[7][dte]:.2f}%\n'
            )


def get_daily_engagement_rate_per_age_category():
    """
    Calculate Daily Engagement Rate per age category.
    """
    der = {}
    for i in range(2, 8):
        logging.info(f'Processing DER for category {i}')
        # store total users and daily active users per age category
        der[i] = {}
        users, _, _, _, _ = load_user_data(USERS_FILE, target_age_categories=[i])
        total_users = total_users_timeline(users)
        active_users = calculate_daily_active_transactions(TRANSACTIONS_FILE, users)

        rate = 0
        for dte in sorted(total_users):
            if total_users[dte] > 0 and dte in active_users:
                rate = (active_users[dte]/total_users[dte])*100
            
            der[i][dte] = rate
        
    save_daily_engagement_rate_per_age_category(der)


def get_first_transaction_per_user():
    """
    Get the first transaction date for each user.
    """
    logging.info('Gathering first transaction for each user.')
    
    last_tx = datetime(2018, 6, 15) - timedelta(days=30)
    first_tx = {}
    for transaction in load_csv_data(TRANSACTIONS_FILE):
        uid = transaction['user_id']

        try:
            # timestamp communicated with and without miliseconds information
            ts = load_timestamp_ms(transaction['timestamp'])
        except ValueError as ex:
            ts = load_timestamp(transaction['timestamp'])

        if ts > last_tx:
            # only keep users which have a first transaction early enough
            # to assess all 3 windows
            continue

        if (uid in first_tx and ts < first_tx[uid]) or uid not in first_tx:
            first_tx[uid] = ts

    return first_tx


def define_retention_windows(first_tx: dict):
    """
    Define windows that will be used to assess the retention 
    rate for users.
    """
    logging.info('Defining retention windows.')
    # windows to be used to evaluate user retention. This is ugly, we need an object but we're in a rush.
    windows = {
        u: {
            'w1': [None, None, False],  # date window opens, date it closes and 
            'w2': [None, None, False],  # bool indicating whether there is a transaction
            'w3': [None, None, False]
        } for u in first_tx
    }
    
    for u, tx in first_tx.items():
        # first window range, 7-8 days after the first transaction
        windows[u]['w1'][0] = tx.date() + timedelta(days=7)
        windows[u]['w1'][1] = tx.date() + timedelta(days=7)

        # second window range, 13-14 days after the first transaction
        windows[u]['w2'][0] = tx.date() + timedelta(days=14)
        windows[u]['w2'][1] = tx.date() + timedelta(days=14)

        # third window range, 28-30 days after the first transaction
        windows[u]['w3'][0] = tx.date() + timedelta(days=28)
        windows[u]['w3'][1] = tx.date() + timedelta(days=28)
    
    return windows


def load_user_categories_map():
    """
    Load dictionary of user IDs (value) and age categories (key)
    """
    categories = defaultdict(list)
    for u in load_csv_data(USERS_FILE):
        age_cat = age_category(int(u['age']))
        categories[age_cat].append(u['user_id'])

    return categories


def calculate_retention_rate(windows: dict):
    """
    Calculate the retention window specific for each user.
    """
    logging.info('Calculating retention rates.')
    for transaction in load_csv_data(TRANSACTIONS_FILE):
        uid = transaction['user_id']
        if uid not in windows:
            continue

        try:
            # timestamp communicated with and without miliseconds information
            ts = load_timestamp_ms(transaction['timestamp'])
        except ValueError as ex:
            ts = load_timestamp(transaction['timestamp'])

        if windows[uid]['w1'][0] <= ts.date() <= windows[uid]['w1'][1]:
            # transaction performed in first window
            windows[uid]['w1'][2] = True 
        if windows[uid]['w2'][0] <= ts.date() <= windows[uid]['w2'][1]:
            # transaction performed in first window
            windows[uid]['w2'][2] = True
        if windows[uid]['w3'][0] <= ts.date() <= windows[uid]['w3'][1]:
            # transaction performed in first window
            windows[uid]['w3'][2] = True


def get_retention_rate():
    """
    Calculates the retention rate for each user by evaluating 
    the first time a user performs a transaction and whether 
    the user is still actively transacting 7-8 days, 13-14 days 
    and 28-30 days after the initial transaction.
    """
    age_categories = load_user_categories_map()
    first_tx = get_first_transaction_per_user()
    windows = define_retention_windows(first_tx)
    calculate_retention_rate(windows)
    
    logging.info(f'Total Users Assessed: {len(first_tx)}')
    logging.info(f"Total Users retained 7 days after first transaction: {sum([1 for u in first_tx if windows[u]['w1'][2] is True])}")
    logging.info(f"Total Users retained 14 days after first transaction: {sum([1 for u in first_tx if windows[u]['w2'][2] is True])}")
    logging.info(f"Total Users retained 28 days after first transaction: {sum([1 for u in first_tx if windows[u]['w3'][2] is True])}")

    for i in range(2, 8):
        logging.info(f'Age category {i} Users Assessed: {sum([1 for u in first_tx if u in age_categories[i]])}')
        logging.info(f"Age category {i} Users retained 7 days after first transaction: {sum([1 for u in first_tx if u in age_categories[i] and windows[u]['w1'][2] is True])}")
        logging.info(f"Age category {i} Users retained 14 days after first transaction: {sum([1 for u in first_tx if u in age_categories[i] and windows[u]['w2'][2] is True])}")
        logging.info(f"Age category {i} Users retained 28 days after first transaction: {sum([1 for u in first_tx if u in age_categories[i] and windows[u]['w3'][2] is True])}")


if __name__ == '__main__':
    # get_daily_engagement_rate_per_age_category()

    get_retention_rate()

    # users, age_categories, average_user_age, android, overdraft = load_user_data(users_file, target_age_categories=None)
    # total_users = total_users_timeline(users)
    
    # active_users, gained, dropped = calculate_daily_active_transactions(users)

    # save_daily_active_users_to_csv(active_users) 
    # save_daily_active_users_wow_growth_to_csv(active_users)
    # save_daily_engagement_rate_to_csv(total_users, active_users, users)

    # plot_active_users(
    #     [gained[i] for i in sorted(gained)],
    #     [dropped[i] for i in sorted(dropped)],
    #     [active_users[i] for i in sorted(active_users)],
    #     [total_users[i] for i in sorted(active_users)],
    #     sorted(active_users), spacing=3
    # )
