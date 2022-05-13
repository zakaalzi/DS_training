import re
import statistics
from scipy import stats
import pandas as pd
import category_encoders as ce
from math import isnan, radians, cos, sin, asin, sqrt
import numpy as np
import pgeocode
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class DataCleaner():

    def __init__(self, df, inplace=True):
        if not inplace:
            self.df = df.copy()
        elif inplace:
            self.df = df
        self.sid_obj = SentimentIntensityAnalyzer()
        self.original_columns = df.columns

    def get_df(self):
        return self.df
    
    def set_df(self, new_df):
        self.df = new_df

    # Util Functions
    @staticmethod
    def _currency_to_integer(x):
        try:
            currency_symbols = ['£', '$', '€']
            for symbol in currency_symbols:
                x = x.replace(symbol, '')
            x = x.replace(',', '')
            return int(x)
        except:
            return x

    @staticmethod
    def _extract_manufactured_year(x):
        if isinstance(x, str):
            return int(x[:4])
        return x


    @staticmethod
    def _reg_checker(reg_no):
        if isinstance(reg_no, str):
            # Returns True if plat matches modern standards (from 2001 September)
            pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$')
            reg_return = pattern.findall(reg_no)
            if reg_return:
                return True
            return False
        return False

    def _is_private_plate(self, reg, year):
        if year >= 2002: # new systen
            if not self._reg_checker(reg):
                return 1
            return 0
        else:
            return 0

    
    def group_on_quartile(self, x, column, df=None):
        """
        UTIL
        """
        if df is None:
            df = self.df

        try:
            percentile = stats.percentileofscore(df[column], x, kind='rank')
        except Exception:
            return 0
        if percentile < 25:
            return 1
        elif 25 <= percentile < 50:
            return 2
        elif 50 <= percentile < 75:
            return 3
        else:
            return 4


    def _encode_mileage_deviation(self, x):
        if isinstance(x, str):
            if 'ABOVE' in x:
                return 1
            elif 'EQUAL' in x:
                return 0
            elif 'BELOW' in x:
                return -1
        return 0

    @staticmethod
    def _find_rare(col, threshold=0.01):
        """
        Find the uncommon labels in a set and group them as 'rare'
        """
        vcs = col.value_counts().astype(int) # This is behaving weirdly
        n = len(col)
        rare_labels = []
        for k in vcs.keys():
            vcs[k] /= n
            if vcs[k] <= threshold:
                rare_labels.append(k)
        return rare_labels

    @staticmethod
    def _remove_whitespace_and_symbols(x):
        x = str(x).lower()
        x = re.sub(r'\W', '', x) # Symbols + whitespace
        return x

    def encode_with_hashing(self, df, x, n_components=8):
        local_copy = df.copy()
        hash_encoder = ce.HashingEncoder(n_components=n_components, hash_method='sha256', cols=[x])
        dfbin = hash_encoder.fit_transform(local_copy[x])
        for col in dfbin.columns:
            dfbin.rename(columns={col: f"{x}_{col}"}, inplace=True)
        local_copy = pd.concat([df, dfbin], axis=1)
        local_copy.drop(columns=[x], inplace=True)
        return local_copy


    @staticmethod
    def _extract_integer(pattern, x):
        if isinstance(x, str):
            x = re.sub(pattern, '', x)
            x = x.replace(',', '')
            return int(x)
        return x

    
    @staticmethod
    def _group_by_num_owners(x):
        try:
            x = int(x) # np.nan can't be converted to int
        except ValueError:
            return 'unknown'
        if x == 1:
            return '1 owner'
        elif 2 <= x <= 4:
            return '2to4 owners'
        elif 5 <= x <= 9:
            return '5to9 owners'
        else:
            return '10+ owners'
    
    @staticmethod
    def _engine_to_numerical(x):
            try:
                x = x[:-1]
                return float(x)
            except:
                return x

    
    def categorise_engine_size(self, x):
        x = self._engine_to_numerical(x)
        if x <= 1.0:
            return 'small'
        elif 1.0 < x < 3.0:
            return 'medium'
        elif 3.0 <= x < 15.0:
            return 'large'
        elif 15.0 < x < 500: # Assume error in input
            x /= 10
            return self.categorise_engine_size(x)
        elif x >= 500: # Assume engine is in CC
            x /= 1000
            return self.categorise_engine_size(x)
        else:
            return 'unknown'

    @staticmethod
    def _co2_to_numerical(x):
        if isinstance(x, str):
            return int(x[:-4])
        else:
            return x
    
    def categorise_co2_emissions(self, x):
        x = self._co2_to_numerical(x)
        if x == np.nan:
            return 'unknown'
        elif x == 0:
            return 'band1'
        elif 1 <= x <= 50:
            return 'band2'
        elif 51 <= x <= 75:
            return 'band3'
        elif 76 <= x <= 90:
            return 'band4'
        elif 91 <= x <= 100:
            return 'band5'
        elif 101 <= x <= 110:
            return 'band6'
        elif 111 <= x <= 130:
            return 'band7'
        elif 131 <= x <= 150:
            return 'band8'
        elif 151 <= x <= 170:
            return 'band9'
        elif 171 <= x <= 190:
            return 'band10'
        elif 191 <= x <= 225:
            return 'band11'
        elif 226 <= x <= 255:
            return 'band12'
        else:
            return 'band13'


    @staticmethod
    def _single_pt_haversine(lat, lng, decimal_degrees=True):
        """
        'Single-point' Haversine: Calculates the great circle distance
        between a point on Earth and the (0, 0) lat-long coordinate
        """
        r = 6371 # Earth's radius (km). Have r = 3956 if you want miles

        # Convert decimal degrees to radians
        if decimal_degrees:
            lat, lng = map(radians, [lat, lng])

        # 'Single-point' Haversine formula
        a = sin(lat/2)**2 + cos(lat) * sin(lng/2)**2
        d = 2 * r * asin(sqrt(a)) 

        return d

    @staticmethod
    def get_sentiment(text, sid_obj, raw=False):
        """
        Extra column could be the raw sentiment
        """
        if isinstance(text, float):
            text = ""

        sentiment_dict = sid_obj.polarity_scores(text)

        compound = sentiment_dict['compound']
        if compound >= 0.05:
            return 1
        elif compound <= -0.05:
            return -1
        else:
            return 0




    # def remove_skewed_outliers(self, col):
    #     """
    #     Remove based on IQR range
    #     - A multiplier of 1.5 represents a +- 2.7 SD. +- 3 SD for a Gaussian is 99% of the data 
    #     """
    #     x = np.array(self.df[col])
    #     q3, q1 = np.percentile(x, [75 ,25])
    #     iqr_const = (q3 - q1) * 1.5
    #     upper_bound = q3 + iqr_const
    #     lower_bound = q1 - iqr_const

    #     mask = (x < lower_bound) & (x > upper_bound)
    #     return self.df[mask]


    # Main Functions
    def clean_price(self):
        self.df['price'] = [self._currency_to_integer(x) for x in self.df.price]
    
    def add_service_history_flag(self):
        self.df['has_service_history'] = [1 if x is not None else 0 for x in self.df.service_history] 

    def add_imported_flag(self):
        self.df['is_imported'] = [1 if x else 0 for x in self.df.imported]

    def add_website_flag(self):
        self.df['has_website'] = [1 if x is not None else 0 for x in self.df.sellerwebsite]

    def add_trim_flag(self):
        self.df['has_trim'] = [1 if x is not None else 0 for x in self.df.trim]

    def add_ulez_flag(self):
        self.df['is_ulez'] = [1 if x == 'ULEZ' else 0 for x in self.df.emission_scheme]

    def add_convertible_flag(self):
        self.df['is_convertible'] = [1 if x == 'Convertible' else 0 for x in self.df.body_type]

    def add_vrm_flag(self):
        self.df['known_reg_plate'] = [1 if x else 0 for x in self.df.vrm]

    def add_manufactured_year(self):
        self.df['year_of_manufacture'] = [self._extract_manufactured_year(x) for x in self.df.manufactured_year]

    def add_private_plate_flag(self):
        try:
            self.df['is_private_plate'] = [self._is_private_plate(reg, year) for reg, year in zip(self.df.vrm, self.df.year_of_manufacture)]
        except Exception:
            print("year of manufacture column not created yet")  

    def calculate_vehicle_age(self):
        self.df['year_of_scrape'] = [int(x[-4:]) for x in self.df.todaysdate]
        self.df['vehicle_age'] = [x - y  if np.isnan(x, where=False) and np.isnan(y, where=False) else 1000000 for x, y in zip(self.df.year_of_scrape, self.df.year_of_manufacture)]

    def add_mileage_deviation(self):
        self.df['mileage_deviation_encoded'] = [self._encode_mileage_deviation(x) for x in self.df.mileageDeviation]

    def group_care_makes(self):
        rare_labels = self._find_rare(self.df.make)
        self.df['make_grouped'] = ['rare' if x in rare_labels else self._remove_whitespace_and_symbols(x) for x in self.df.make]
    
    def add_manual_flag(self):
        self.df['is_manual'] = [0 if x == 'Automatic' else 1 for x in self.df.transmission]

    def group_fuel_types(self):
        fuel_rare_labels = self._find_rare(self.df.fuel_type)
        # Using this list is not flexible at all
        to_exclude = ["Hybrid – Diesel/Electric", "Hybrid – Diesel/Electric Plug-in"]
        for exclude in to_exclude:
            fuel_rare_labels.remove(exclude)

        def _remove_plugin_from_fuel(x):
            x = re.sub(r'plugin', '', x)
            return x
            
        self.df['fuel_type_grouped'] = ['other' if x in fuel_rare_labels else _remove_plugin_from_fuel(self._remove_whitespace_and_symbols(x)) for x in self.df.fuel_type]

    def group_doors(self):
        self.df['doors'].fillna('unknown', inplace=True)
        door_rare_labels = self._find_rare(self.df.doors)
        self.df['doors_grouped'] = ['>5 doors' if x in door_rare_labels and x != 'unknown' else x for x in self.df.doors]


    def group_seats(self):
        self.df['seats'].fillna('unknown', inplace=True)
        seat_rare_labels = self._find_rare(self.df.seats)
        self.df['seats_grouped'] = ['>6 seats' if x in seat_rare_labels and x != 'unknown' else x for x in self.df.seats]

    def mileage_to_integer(self):
        self.df['mileage'] = [self._extract_integer(r'mile[s]?', x) for x in self.df.mileage]

    def group_owners(self):
        self.df['owners'] = [self._extract_integer(r'owner[s]?', x) for x in self.df.owners]
        self.df['owners_grouped'] = [self._group_by_num_owners(x) for x in self.df.owners]

    def group_engine_size(self):
        self.df['engine_size_grouped'] = [self.categorise_engine_size(x) for x in self.df.engine_size]

    def add_new_flag(self):
        self.df['is_new'] = [1 if x == 'New' else 0 for x in self.df.condition]
    
    def group_co2(self):
        self.df['co2_grouped'] = [self.categorise_co2_emissions(x) for x in self.df.co2Emissions]

    def count_images(self):
        self.df['image_count'] = [x.count('https') if isinstance(x, str) else 0 for x in self.df.images]

    def add_postcode_flag(self):
            self.df['has_postcode'] = [1 if x else 0 for x in self.df.sellerpostcode]

    def extract_lat_long_from_postcode(self):
        nomi = pgeocode.Nominatim('gb')
        postcodes = list(self.df.sellerpostcode)
        geo_df = nomi.query_postal_code(postcodes)

        self.df['long_lat_feature'] = [self._single_pt_haversine(lat, long, decimal_degrees=True) for long, lat in zip(geo_df.longitude, geo_df.latitude)]

    def group_annual_tax(self):
        self.df['annual_tax_grouped'] = [self.group_on_quartile(x, 'annual_tax') for x in self.df.annual_tax]
    
    def encode_adverttitle(self):
        self.df['advert_title_sentiment'] = [self.get_sentiment(x, self.sid_obj) for x in self.df.adverttitle]

    def encode_advert(self):
        self.df['advert_sentiment'] = [self.get_sentiment(x, self.sid_obj) for x in self.df.advert]

    def load_sentiment_from_file(self):
        sentiment_df = pd.read_csv("tmp_sentiment_file.csv")
        self.df['advert_sentiment'] = sentiment_df['advert_sentiment']
        self.df['advert_title_sentiment'] = sentiment_df['advert_title_sentiment']



    def clean_data(self):
        ## Clean target: Price
        self.clean_price()
        self.add_service_history_flag()
        self.add_imported_flag()
        self.add_imported_flag
        self.add_website_flag()
        self.add_trim_flag()
        self.add_ulez_flag()
        self.add_convertible_flag()
        self.add_vrm_flag()
        self.add_manufactured_year()
        self.add_private_plate_flag()
        self.calculate_vehicle_age()
        self.add_mileage_deviation()
        self.add_manual_flag()
        self.add_new_flag()
        self.add_postcode_flag()

        self.mileage_to_integer()
        self.count_images()
        # self.extract_lat_long_from_postcode() # This introduced too many nulls that are difficult to impute
        self.load_sentiment_from_file() # temporary to save time
        
        self.group_care_makes()
        self.group_fuel_types()
        self.group_doors()
        self.group_seats()
        self.group_owners()
        self.group_engine_size()
        self.group_co2()

        self.df = self.encode_with_hashing(self.df, 'model', n_components=16) # takes about 30s


    def drop_columns(self, df=None):
        if df is None:
            df = self.df
        cols_to_drop = ['year_of_scrape', 'year_of_manufacture']
        full_col_list = list(self.original_columns) + cols_to_drop
        for col in full_col_list:
            try:
                df.drop(columns=col, inplace=True)   
            except Exception:
                continue


    def _get_ordinal(self, x, X_train, y_train):
        X_train_copy = X_train.copy()
        X_train_copy['price'] = y_train
        med_dict = X_train_copy.groupby(x)['price'].median()
        med_dict_sorted = {k: v for k, v in sorted(med_dict.items(), key=lambda item: item[1])}
        label = 0
        for k, v in med_dict_sorted.items():
            med_dict_sorted[k] = label
            label += 1
        return med_dict_sorted

    def _map_to_ordinal(self, col, mapping, df=None):
        if df is None:
            df=self.df
        df[col] = df[col].map(mapping)

    def to_ordinal(self, col, X_train, y_train, X_test):
        col_mapping = self._get_ordinal(col, X_train, y_train)
        self._map_to_ordinal(col, col_mapping, df=X_train)
        self._map_to_ordinal(col, col_mapping, df=X_test)


    def convert_columns_to_ordinal(self, columns, X_train, y_train, X_test):
        for col in columns:
            self.to_ordinal(col, X_train, y_train, X_test)
