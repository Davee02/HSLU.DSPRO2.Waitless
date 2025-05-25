#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import date, datetime
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def get_weather_forecast(dates, location="Rust, Germany"):
    """Get weather forecast for future dates using free API."""
    import requests
    from datetime import datetime
    
    weather_data = []
    
    print(f"Fetching weather for {location}...")
    
    for date_obj in dates:
        try:
            date_str = date_obj.strftime("%Y-%m-%d")
            
            url = f"https://wttr.in/{location.replace(' ', '+')}?format=j1&date={date_str}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'weather' in data and len(data['weather']) > 0:
                    day_weather = data['weather'][0]
                    
                    temp_c = float(day_weather.get('avgtempC', 20))
                    
                    precip_mm = float(day_weather.get('totalSnow_cm', 0)) + float(day_weather.get('precipMM', 0))
                    
                    wind_kmh = float(day_weather.get('uvIndex', 10))
                    if 'hourly' in day_weather and len(day_weather['hourly']) > 0:
                        wind_kmh = float(day_weather['hourly'][0].get('windspeedKmph', 10))
                    
                    weather_data.append({
                        'date': pd.to_datetime(date_obj),
                        'temperature': temp_c,
                        'rain': precip_mm,
                        'wind': wind_kmh
                    })
                    
                    print(f"  {date_str}: {temp_c}°C, {precip_mm}mm rain, {wind_kmh}km/h wind")
                
                else:
                    weather_data.append({
                        'date': pd.to_datetime(date_obj),
                        'temperature': 20,
                        'rain': 0,
                        'wind': 10
                    })
                    print(f"  {date_str}: Using defaults (API data incomplete)")
            
            else:
                weather_data.append({
                    'date': pd.to_datetime(date_obj),
                    'temperature': 20,
                    'rain': 0,
                    'wind': 10
                })
                print(f"  {date_str}: API error, using defaults")
        
        except Exception as e:
            weather_data.append({
                'date': pd.to_datetime(date_obj),
                'temperature': 20,
                'rain': 0, 
                'wind': 10
            })
            print(f"  {date_str}: Error ({e}), using defaults")
    
    return pd.DataFrame(weather_data)


# In[3]:


def load_and_split_data(file_path, test_year=2023):
    """Load data and split by year to avoid data leaks."""
    df = pd.read_parquet(file_path)
    
    print(f"Loaded data with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    ride_name_cols = [col for col in df.columns if col.startswith('ride_name_') and not any(x in col for x in ['sin', 'cos'])]
    if ride_name_cols:
        df['ride_name'] = df[ride_name_cols].idxmax(axis=1).str.replace('ride_name_', '')
    else:
        print("Warning: No ride_name columns found")
        df['ride_name'] = 'unknown'
    
    year_cols = [col for col in df.columns if col.startswith('year_') and col.replace('year_', '').isdigit()]
    if year_cols:
        df['year'] = df[year_cols].idxmax(axis=1).str.replace('year_', '').astype(int)
    else:
        print("Warning: No year columns found")
        if 'timestamp' in df.columns:
            df['year'] = pd.to_datetime(df['timestamp']).dt.year
        else:
            df['year'] = 2022
    
    month_cols = [col for col in df.columns if col.startswith('month_') and 
                  col.replace('month_', '').isdigit()]
    if month_cols:
        df['month'] = df[month_cols].idxmax(axis=1).str.replace('month_', '').astype(int)
    else:
        print("Warning: No month columns found")
        if 'timestamp' in df.columns:
            df['month'] = pd.to_datetime(df['timestamp']).dt.month
        else:
            df['month'] = 6
    
    day_cols = [col for col in df.columns if col.startswith('day_') and 
                col.replace('day_', '').isdigit()]
    if day_cols:
        df['day'] = df[day_cols].idxmax(axis=1).str.replace('day_', '').astype(int)
    else:
        print("Warning: No day columns found")
        if 'timestamp' in df.columns:
            df['day'] = pd.to_datetime(df['timestamp']).dt.day
        else:
            df['day'] = 15
    
    hour_cols = [col for col in df.columns if col.startswith('hour_') and 
                 col.replace('hour_', '').isdigit()]
    if hour_cols:
        df['hour'] = df[hour_cols].idxmax(axis=1).str.replace('hour_', '').astype(int)
    else:
        print("Warning: No hour columns found")
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        else:
            df['hour'] = 12
    
    try:
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    except Exception as e:
        print(f"Error creating date column: {e}")
        print(f"Year range: {df['year'].min()}-{df['year'].max()}")
        print(f"Month range: {df['month'].min()}-{df['month'].max()}")
        print(f"Day range: {df['day'].min()}-{df['day'].max()}")

        df['month'] = df['month'].clip(1, 12)
        df['day'] = df['day'].clip(1, 28)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    try:
        df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    except Exception as e:
        print(f"Error creating timestamp column: {e}")
        df['hour'] = df['hour'].clip(0, 23)
        df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    
    train_mask = df['year'] < test_year
    test_mask = df['year'] == test_year
    
    train_data = df[train_mask].copy()
    test_data = df[test_mask].copy()
    
    print(f"Train data: {len(train_data)} samples")
    print(f"Test data: {len(test_data)} samples")
    
    if len(train_data) > 0:
        print(f"Date range - Train: {train_data['date'].min()} to {train_data['date'].max()}")
    if len(test_data) > 0:
        print(f"Date range - Test: {test_data['date'].min()} to {test_data['date'].max()}")
    
    return train_data, test_data


# In[4]:


def check_data_quality(df):
    """Check for data quality issues."""
    print("=== Data Quality Check ===")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"Hours per day (avg): {len(df) / df['date'].nunique():.1f}")
    
    dow_counts = df['day_of_week'].value_counts().sort_index()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print("\nRecords by day of week:")
    for i, day in enumerate(days):
        count = dow_counts.get(i, 0)
        print(f"  {day}: {count} records")
    
    hour_counts = df['hour'].value_counts().sort_index()
    print(f"\nHour range: {df['hour'].min()}-{df['hour'].max()}")
    print(f"Hours with <100 records: {(hour_counts < 100).sum()}")
    
    print(f"\nWait time stats:")
    print(f"  Min: {df['wait_time'].min():.1f} minutes")
    print(f"  Max: {df['wait_time'].max():.1f} minutes")
    print(f"  Mean: {df['wait_time'].mean():.1f} minutes")
    print(f"  Zero wait times: {(df['wait_time'] == 0).sum()} records")
    
    return df


# In[5]:


def calculate_hourly_attendance_metrics(ride_data):
    """Calculate attendance metrics using only external factors (no wait time dependency)."""
    print(f"Calculating attendance metrics for {len(ride_data)} data points...")
    
    log_wait = np.log1p(ride_data['wait_time'])
    ride_data['attendance_score'] = (log_wait.rank(pct=True) * 100).round(1)
    
    ride_data['year'] = ride_data['timestamp'].dt.year
    ride_data['month'] = ride_data['timestamp'].dt.month
    ride_data['day_of_month'] = ride_data['timestamp'].dt.day
    ride_data['day_of_week'] = ride_data['timestamp'].dt.dayofweek
    ride_data['day_of_year'] = ride_data['timestamp'].dt.dayofyear
    ride_data['hour'] = ride_data['timestamp'].dt.hour
    ride_data['is_weekend'] = (ride_data['day_of_week'] >= 5).astype(int)
    
    season_map = {
        'spring': (ride_data['month'].isin([3, 4, 5])),
        'summer': (ride_data['month'].isin([6, 7, 8])),
        'fall': (ride_data['month'].isin([9, 10, 11])),
        'winter': (ride_data['month'].isin([12, 1, 2]))
    }
    
    ride_data['season'] = 'unknown'
    for season, mask in season_map.items():
        ride_data.loc[mask, 'season'] = season
    
    season_intensity = {
        'summer': 1.2,    # 20% boost for summer
        'spring': 1.1,    # 10% boost for spring  
        'fall': 1.0,      # Normal for fall
        'winter': 0.8     # 20% reduction for winter
    }
    ride_data['season_intensity'] = ride_data['season'].map(season_intensity).fillna(1.0)
    
    ride_data['month_sin'] = np.sin(2 * np.pi * ride_data['month'] / 12)
    ride_data['month_cos'] = np.cos(2 * np.pi * ride_data['month'] / 12)
    ride_data['day_of_week_sin'] = np.sin(2 * np.pi * ride_data['day_of_week'] / 7)
    ride_data['day_of_week_cos'] = np.cos(2 * np.pi * ride_data['day_of_week'] / 7)
    ride_data['hour_sin'] = np.sin(2 * np.pi * ride_data['hour'] / 24)
    ride_data['hour_cos'] = np.cos(2 * np.pi * ride_data['hour'] / 24)
    
    weather_features = ['temperature', 'rain', 'wind']
    for feature in weather_features:
        if feature not in ride_data.columns:
            ride_data[feature] = 20 if feature == 'temperature' else 0
    
    ride_data['temp_comfort'] = 100 - ((ride_data['temperature'] - 22) ** 2)
    ride_data['rain_impact'] = np.where(ride_data['rain'] > 0, -ride_data['rain'] * 10, 0)
    ride_data['wind_impact'] = np.where(ride_data['wind'] > 15, -(ride_data['wind'] - 15) * 2, 0)
    
    holiday_cols = [col for col in ride_data.columns if 'holiday' in col.lower()]
    if holiday_cols:
        ride_data['is_any_holiday'] = ride_data[holiday_cols].max(axis=1)
    else:
        ride_data['is_any_holiday'] = 0
    
    ride_data['holiday_boost'] = ride_data['is_any_holiday'] * 1.3
    
    print(f"Created attendance metrics for {len(ride_data)} data points")
    print(f"Weather features: {[f for f in weather_features if f in ride_data.columns]}")
    print(f"Holiday columns found: {holiday_cols}")
    
    return ride_data


# In[6]:


def train_attendance_prediction_model(hourly_attendance_data, target_metric='attendance_score'):
    """Train attendance model using only external factors (no wait time features)."""
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    print(f"Training attendance prediction model for {target_metric}...")
    
    cat_features = ['season']
    
    num_features = [
        'month', 'day_of_month', 'day_of_week', 'hour', 'is_weekend',
        'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos', 'hour_sin', 'hour_cos',
        'season_intensity',
        'temperature', 'rain', 'wind', 'temp_comfort', 'rain_impact', 'wind_impact',
        'is_any_holiday', 'holiday_boost'
    ]
    
    available_features = [f for f in num_features if f in hourly_attendance_data.columns]
    print(f"Available features: {available_features}")
    
    hourly_attendance_data = hourly_attendance_data.sort_values('timestamp')
    split_idx = int(len(hourly_attendance_data) * 0.8)
    train_data = hourly_attendance_data.iloc[:split_idx]
    test_data = hourly_attendance_data.iloc[split_idx:]
    
    print(f"Training: {len(train_data)} samples, Test: {len(test_data)} samples")
    
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_features),
            ('num', num_transformer, available_features)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ))
    ])
    
    X_train = train_data[cat_features + available_features]
    y_train = train_data[target_metric]
    
    for col in X_train.columns:
        if X_train[col].isna().any():
            if X_train[col].dtype in ['object', 'category']:
                X_train[col] = X_train[col].fillna('unknown')
            else:
                X_train[col] = X_train[col].fillna(X_train[col].median())
    
    model.fit(X_train, y_train)
    
    X_test = test_data[cat_features + available_features]
    y_test = test_data[target_metric]
    
    for col in X_test.columns:
        if X_test[col].isna().any():
            if X_test[col].dtype in ['object', 'category']:
                X_test[col] = X_test[col].fillna('unknown')
            else:
                X_test[col] = X_test[col].fillna(X_test[col].median())
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Attendance Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    if hasattr(model['regressor'], 'feature_importances_'):
        feature_names = []
        for name, transformer, features in preprocessor.transformers_:
            if name == 'cat':
                if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                    encoded_names = transformer.named_steps['onehot'].get_feature_names_out(features)
                    feature_names.extend(encoded_names)
            else:
                feature_names.extend(features)
        
        if len(feature_names) == len(model['regressor'].feature_importances_):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model['regressor'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 features for attendance:")
            for _, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return {
        'model': model,
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'features': {'cat_features': cat_features, 'num_features': available_features},
        'target_metric': target_metric,
        'test_predictions': {'y_test': y_test, 'y_pred': y_pred, 'test_data': test_data}
    }


# In[7]:


def predict_attendance_for_timestamps(timestamps, attendance_model, historical_wait_times=None):
    """Predict attendance using only external factors."""
    
    cat_features = attendance_model['features']['cat_features']
    num_features = attendance_model['features']['num_features']
    
    predictions = []
    
    for timestamp in timestamps:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
            
        date_obj = timestamp.date()
        hour = timestamp.hour
        
        temp, rain, wind = 20, 0, 10
            
        features = {
            'month': date_obj.month,
            'day_of_month': date_obj.day,
            'day_of_week': date_obj.weekday(),
            'hour': hour,
            'is_weekend': 1 if date_obj.weekday() >= 5 else 0,
            
            'month_sin': np.sin(2 * np.pi * date_obj.month / 12),
            'month_cos': np.cos(2 * np.pi * date_obj.month / 12),
            'day_of_week_sin': np.sin(2 * np.pi * date_obj.weekday() / 7),
            'day_of_week_cos': np.cos(2 * np.pi * date_obj.weekday() / 7),
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            
            'season': 'summer' if date_obj.month in [6,7,8] else 
                     'spring' if date_obj.month in [3,4,5] else
                     'fall' if date_obj.month in [9,10,11] else 'winter',
            
            'season_intensity': 1.2 if date_obj.month in [6,7,8] else
                              1.1 if date_obj.month in [3,4,5] else  
                              1.0 if date_obj.month in [9,10,11] else 0.8,
            
            'temperature': temp,
            'rain': rain,
            'wind': wind,
            'temp_comfort': 100 - ((temp - 22) ** 2),
            'rain_impact': -rain * 10 if rain > 0 else 0,
            'wind_impact': -(wind - 15) * 2 if wind > 15 else 0,
            
            'is_any_holiday': 0,
            'holiday_boost': 0
        }
        
        features_df = pd.DataFrame([features])
        
        for feature in cat_features + num_features:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        X_pred = features_df[cat_features + num_features]
        attendance_pred = attendance_model['model'].predict(X_pred)[0]
        
        predictions.append({
            'timestamp': timestamp,
            'attendance_score': attendance_pred
        })
    
    return pd.DataFrame(predictions)


# In[8]:


def prepare_features_with_attendance(df, attendance_predictions=None):
    """Prepare features for wait time prediction, including attendance if available."""
    
    exclude_cols = ['wait_time', 'ride_name', 'year', 'date', 'month', 'day', 'hour', 'timestamp']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X_temp = df[feature_cols]
    numeric_cols = X_temp.select_dtypes(include=[np.number]).columns.tolist()
    
    X = X_temp[numeric_cols].copy()
    y = df['wait_time']
    
    if 'attendance_score' in df.columns:
        print(f"Found attendance_score column with {df['attendance_score'].notna().sum()} valid values")
        if 'attendance_score' not in X.columns:
            X['attendance_score'] = df['attendance_score']
    
    elif attendance_predictions is not None:
        df_with_attendance = df.merge(attendance_predictions, on='date', how='left')
        
        if 'attendance_score' in df_with_attendance.columns:
            X['attendance_score'] = df_with_attendance['attendance_score']
            print(f"Added attendance_score feature with {X['attendance_score'].notna().sum()} valid values")
    
    if 'attendance_score' in X.columns and X['attendance_score'].isna().any():
        median_attendance = X['attendance_score'].median()
        missing_count = X['attendance_score'].isna().sum()
        X['attendance_score'] = X['attendance_score'].fillna(median_attendance)
        print(f"Filled {missing_count} missing attendance values with median: {median_attendance:.1f}")
    
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"NaN values found: {nan_count}")
        for col in X.columns:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
    
    if y.isna().sum() > 0:
        print(f"Dropping {y.isna().sum()} rows with NaN target values...")
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
    
    print(f"Selected {len(X.columns)} features: {list(X.columns)}")
    print(f"Final feature shape: {X.shape}")
    
    return X, y


# In[9]:


def train_wait_time_model(X_train, y_train):
    """Train wait time prediction model with Gradient Boosting."""
    from sklearn.ensemble import GradientBoostingRegressor
    
    scaler = StandardScaler()
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled, y_train)
    
    return scaler, model


# In[10]:


def evaluate_wait_time_model(scaler, model, X_test, y_test):
    """Evaluate wait time prediction model."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred = np.maximum(y_pred, 0)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    epsilon = 1e-8
    abs_pct_errors = np.abs(y_test - y_pred) / (np.abs(y_test) + epsilon)
    non_zero_mask = y_test > 0
    mape = np.mean(abs_pct_errors[non_zero_mask]) * 100
    
    print(f"Wait Time Model - Test MAE: {mae:.2f} minutes")
    print(f"Wait Time Model - Test RMSE: {rmse:.2f} minutes")
    print(f"Wait Time Model - Test R²: {r2:.4f}")
    print(f"Wait Time Model - Test MAPE: {mape:.2f}%")
    
    wait_ranges = [(0, 10), (10, 30), (30, 60), (60, float('inf'))]
    for min_wait, max_wait in wait_ranges:
        mask = (y_test >= min_wait) & (y_test < max_wait)
        if np.sum(mask) > 0:
            range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            print(f"MAE for {min_wait}-{max_wait if max_wait != float('inf') else '∞'} min: {range_mae:.2f} (n={np.sum(mask)})")
    
    return mae, rmse, r2, mape, y_pred


# In[11]:


def visualize_attendance_predictions(attendance_model, hourly_attendance_data):
    """Visualize attendance prediction results."""
    test_data = attendance_model['test_predictions']['test_data']
    y_test = attendance_model['test_predictions']['y_test']
    y_pred = attendance_model['test_predictions']['y_pred']
    target_metric = attendance_model['target_metric']
    
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel(f'Actual {target_metric}')
    plt.ylabel(f'Predicted {target_metric}')
    plt.title(f'Attendance Model: Actual vs Predicted {target_metric}')
    plt.grid(True, linestyle=':')
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    if len(test_data) > 10:
        plt.figure(figsize=(12, 6))
        
        comparison_df = pd.DataFrame({
            'timestamp': test_data['timestamp'].values,
            'actual': y_test.values,
            'predicted': y_pred
        }).sort_values('timestamp')
        
        if len(comparison_df) > 500:
            comparison_df = comparison_df.iloc[::len(comparison_df)//500]
        
        plt.plot(comparison_df['timestamp'], comparison_df['actual'], 'o-', label='Actual', alpha=0.7, markersize=2)
        plt.plot(comparison_df['timestamp'], comparison_df['predicted'], 's-', label='Predicted', alpha=0.7, markersize=2)
        plt.xlabel('Timestamp')
        plt.ylabel(target_metric)
        plt.title(f'Attendance Model: {target_metric} Over Time')
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    features = attendance_model['features']['num_features']
    temporal_features = [f for f in features if any(temp in f for temp in ['lag', 'rolling', 'trend', 'morning', 'peak'])]
    
    if temporal_features:
        print(f"\nTemporal features used in attendance model:")
        for feature in temporal_features:
            print(f"  - {feature}")
    
    if 'hour' in hourly_attendance_data.columns and 'day_of_week' in hourly_attendance_data.columns:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        hourly_avg = hourly_attendance_data.groupby('hour')['attendance_score'].mean()
        plt.plot(hourly_avg.index, hourly_avg.values, 'o-')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Attendance Score')
        plt.title('Attendance Patterns by Hour')
        plt.grid(True, linestyle=':')
        
        plt.subplot(1, 2, 2)
        daily_avg = hourly_attendance_data.groupby('day_of_week')['attendance_score'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        plt.bar(range(len(daily_avg)), daily_avg.values)
        plt.xlabel('Day of Week')
        plt.ylabel('Average Attendance Score')
        plt.title('Attendance Patterns by Day of Week')
        plt.xticks(range(len(days)), days)
        plt.grid(True, linestyle=':')
        
        plt.tight_layout()
        plt.show()
    
    print(f"\nTop 5 highest attendance hours (actual):")
    top_hours = hourly_attendance_data.nlargest(5, 'attendance_score')[['timestamp', 'attendance_score', 'wait_time']]
    for _, row in top_hours.iterrows():
        print(f"  {row['timestamp']}: Score {row['attendance_score']:.0f}, Wait Time {row['wait_time']:.1f}min")
    
    print(f"\nTop 5 lowest attendance hours (actual):")
    bottom_hours = hourly_attendance_data.nsmallest(5, 'attendance_score')[['timestamp', 'attendance_score', 'wait_time']]
    for _, row in bottom_hours.iterrows():
        print(f"  {row['timestamp']}: Score {row['attendance_score']:.0f}, Wait Time {row['wait_time']:.1f}min")


# In[12]:


def visualize_wait_time_results(y_test, y_pred, test_data, title_suffix=""):
    """Visualize wait time prediction results."""
    y_pred = np.maximum(y_pred, 0)
    
    ride_name_cols = [col for col in test_data.columns if col.startswith('ride_name_')]
    if ride_name_cols:
        test_data_viz = test_data.copy()
        test_data_viz['ride_name'] = test_data_viz[ride_name_cols].idxmax(axis=1).str.replace('ride_name_', '')
        ride_names = test_data_viz['ride_name'].values
    else:
        ride_names = ['unknown'] * len(y_test)
    
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': y_pred - y_test,
        'ride_name': ride_names
    })

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    axes[0].scatter(results_df['Actual'], results_df['Predicted'], alpha=0.5)
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    axes[0].plot([0, max_val], [0, max_val], 'r--')
    axes[0].set_xlabel('Actual Wait Time (minutes)')
    axes[0].set_ylabel('Predicted Wait Time (minutes)')
    axes[0].set_title(f'Wait Time Model: Actual vs Predicted {title_suffix}')
    axes[0].grid(True, linestyle=':')

    mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
    rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Predicted']))
    r2 = r2_score(results_df['Actual'], results_df['Predicted'])
    
    metrics_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}"
    axes[0].text(0.05, 0.95, metrics_text, transform=axes[0].transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    sample_size = min(2000, len(results_df))
    results_sample = results_df.sample(n=sample_size, random_state=42)
    
    axes[1].scatter(range(len(results_sample)), results_sample['Actual'], 
                   label='Actual', alpha=0.7, s=1)
    axes[1].scatter(range(len(results_sample)), results_sample['Predicted'], 
                   label='Predicted', alpha=0.7, s=1)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Wait Time (minutes)')
    axes[1].set_title(f'Wait Time Model: Sample Predictions {title_suffix}')
    axes[1].grid(True, linestyle=':')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    
    bins = [0, 10, 20, 30, 40, 50, 60, np.inf]
    labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60+']
    
    results_df['wait_bin'] = pd.cut(results_df['Actual'], bins=bins, labels=labels, include_lowest=True)
    
    bin_metrics = results_df.groupby('wait_bin').agg({
        'Error': ['mean', 'std'],
        'Actual': 'count'
    })
    
    bin_metrics.columns = ['Mean Error', 'Std Error', 'Count']
    bin_metrics['Abs Mean Error'] = results_df.groupby('wait_bin')['Error'].apply(lambda x: np.abs(x).mean())
    
    bin_metrics = bin_metrics[bin_metrics['Count'] > 0]
    
    plt.bar(range(len(bin_metrics)), bin_metrics['Abs Mean Error'], alpha=0.7)
    plt.xlabel('Actual Wait Time Range (minutes)')
    plt.ylabel('Mean Absolute Error (minutes)')
    plt.title(f'Error by Wait Time Range {title_suffix}')
    plt.xticks(range(len(bin_metrics)), bin_metrics.index)
    
    for i, count in enumerate(bin_metrics['Count']):
        plt.text(i, bin_metrics['Abs Mean Error'].iloc[i] + 0.5, f"n={count}", 
                 ha='center', va='bottom', fontsize=8)
    
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()
    
    return results_df


# In[13]:


def visualize_future_predictions(predictions_df, title="Future Wait Time Predictions"):
    """Visualize future wait time predictions."""
    
    pivot_table = predictions_df.pivot_table(
        index='date',
        columns='hour',
        values='predicted_wait'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt='.1f', 
        cmap='YlOrRd',
        linewidths=.5
    )
    plt.title(title)
    plt.xlabel('Hour of Day')
    plt.ylabel('Date')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    for date in predictions_df['date'].unique():
        date_data = predictions_df[predictions_df['date'] == date]
        plt.plot(date_data['hour'], date_data['predicted_wait'], 'o-', 
                label=str(date), alpha=0.8, markersize=4)
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Predicted Wait Time (minutes)')
    plt.title(f'{title} - By Hour')
    plt.grid(True, linestyle=':')
    plt.legend(title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# In[14]:


def predict_attendance_for_timestamps(timestamps, attendance_model, historical_wait_times=None):
    """Predict attendance using the SAME feature engineering as training."""
    
    cat_features = attendance_model['features']['cat_features']
    num_features = attendance_model['features']['num_features']
    
    # Convert timestamps to DataFrame format that matches training data structure
    timestamp_data = []
    for timestamp in timestamps:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
            
        timestamp_data.append({
            'timestamp': timestamp,
            'wait_time': 15.0  # Dummy wait time for feature calculation (not used as feature)
        })
    
    # Create DataFrame
    temp_df = pd.DataFrame(timestamp_data)
    
    # Apply THE SAME feature engineering as training
    temp_df_with_features = calculate_hourly_attendance_metrics(temp_df)
    
    # Now extract just the features needed for prediction
    prediction_data = []
    
    for _, row in temp_df_with_features.iterrows():
        features = {}
        
        # Extract categorical features
        for cat_feature in cat_features:
            if cat_feature in row:
                features[cat_feature] = row[cat_feature]
            else:
                features[cat_feature] = 'unknown'
        
        # Extract numerical features  
        for num_feature in num_features:
            if num_feature in row:
                features[num_feature] = row[num_feature]
            else:
                features[num_feature] = 0
        
        # Create prediction DataFrame
        features_df = pd.DataFrame([features])
        
        # Make prediction using the trained model
        try:
            attendance_pred = attendance_model['model'].predict(features_df[cat_features + num_features])[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            print(f"Features: {features_df.columns.tolist()}")
            print(f"Expected: {cat_features + num_features}")
            attendance_pred = 50  # Fallback
        
        prediction_data.append({
            'timestamp': row['timestamp'],
            'attendance_score': attendance_pred
        })
    
    return pd.DataFrame(prediction_data)


# In[15]:


def main():
    """Main function to run the full pipeline."""
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    
    data_path = "../data/processed/ep/final_cleaned_processed_wait_times.parquet"
    test_year = 2023
    
    print("=== Attendance-Based Wait Time Prediction Model (No Data Leaks) ===")
    
    # Step 1: Load and split data
    print("\n1. Loading and splitting data...")
    train_data, test_data = load_and_split_data(data_path, test_year)
    
    # Step 2: Calculate attendance metrics from training data ONLY
    print("\n2. Calculating attendance metrics from TRAINING data only...")
    hourly_attendance_train = calculate_hourly_attendance_metrics(train_data)
    print(f"Attendance metrics calculated for {len(hourly_attendance_train)} training samples")
    
    # Step 3: Train attendance prediction model
    print("\n3. Training attendance prediction model...")
    attendance_model = train_attendance_prediction_model(hourly_attendance_train)
    
    # Step 4: Visualize attendance model
    print("\n4. Visualizing attendance prediction results...")
    visualize_attendance_predictions(attendance_model, hourly_attendance_train)
    
    # Step 5: Predict attendance for test timestamps
    print("\n5. Predicting attendance for test timestamps...")
    test_timestamps = test_data['timestamp'].unique()
    attendance_predictions_test = predict_attendance_for_timestamps(test_timestamps, attendance_model)
    print(f"Generated attendance predictions for {len(attendance_predictions_test)} test timestamps")
    print(f"Test attendance range: {attendance_predictions_test['attendance_score'].min():.1f} - {attendance_predictions_test['attendance_score'].max():.1f}")
    
    # Get attendance predictions for training data (using the trained model)
    train_timestamps = train_data['timestamp'].unique()
    attendance_predictions_train = predict_attendance_for_timestamps(train_timestamps, attendance_model)
    print(f"Generated attendance predictions for {len(attendance_predictions_train)} training timestamps")
    
    # Get attendance predictions for test data
    test_timestamps = test_data['timestamp'].unique()
    attendance_predictions_test = predict_attendance_for_timestamps(test_timestamps, attendance_model)
    print(f"Generated attendance predictions for {len(attendance_predictions_test)} test timestamps")
    print(f"Test attendance range: {attendance_predictions_test['attendance_score'].min():.1f} - {attendance_predictions_test['attendance_score'].max():.1f}")
    
    # Step 6: Prepare features for wait time model (both train and test will have attendance_score)
    print("\n6. Preparing features for wait time prediction...")
    
    # Merge training data with predicted attendance
    train_data_with_attendance = train_data.merge(
        attendance_predictions_train, 
        on='timestamp', 
        how='left'
    )
    X_train, y_train = prepare_features_with_attendance(train_data_with_attendance)
    
    # Merge test data with predicted attendance
    test_data_with_attendance = test_data.merge(
        attendance_predictions_test, 
        on='timestamp', 
        how='left'
    )
    X_test, y_test = prepare_features_with_attendance(test_data_with_attendance)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Check if attendance_score was added
    if 'attendance_score' in X_test.columns:
        print(f"✓ Attendance score successfully added as feature")
    else:
        print("⚠ Warning: Attendance score not found in features")
    
    # Step 7: Train wait time prediction model
    print("\n7. Training wait time prediction model...")
    scaler, wait_time_model = train_wait_time_model(X_train, y_train)
    
    # Step 8: Evaluate wait time model
    print("\n8. Evaluating wait time model...")
    mae, rmse, r2, mape, y_pred = evaluate_wait_time_model(scaler, wait_time_model, X_test, y_test)
    
    # Step 9: Visualize wait time results
    print("\n9. Visualizing wait time prediction results...")
    results_df = visualize_wait_time_results(y_test, y_pred, test_data, "(Attendance-Based)")
    
    # Step 10: Demonstrate future predictions
    print("\n10. Demonstrating future predictions...")
    future_dates = [date(2025, 5, 15 + i) for i in range(7)]  # May 15-21, 2025
    future_hours = list(range(10, 20))  # 10am to 7pm
    
    print("Future dates for prediction:")
    for d in future_dates:
        print(f"  - {d}")
    
    sample_features = X_test.iloc[:1]
    future_predictions = predict_future_wait_times(
        scaler, wait_time_model, attendance_model, future_dates, future_hours, sample_features
    )
    
    if not future_predictions.empty:
        print(f"\nGenerated {len(future_predictions)} future predictions")
        print("\nSample future predictions:")
        print(future_predictions.head(10))
        
        print(f"\nFuture attendance scores: {future_predictions['attendance_score'].min():.1f} - {future_predictions['attendance_score'].max():.1f}")
        print(f"Future wait times: {future_predictions['predicted_wait'].min():.1f} - {future_predictions['predicted_wait'].max():.1f} minutes")
        
        visualize_future_predictions(future_predictions, "Future Wait Time Predictions")
    
    print("\n=== Pipeline completed successfully! ===")
    
    return {
        'attendance_model': attendance_model,
        'wait_time_model': wait_time_model,
        'scaler': scaler,
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape},
        'predictions': y_pred,
        'results_df': results_df,
        'future_predictions': future_predictions
    }


# In[16]:


results = main()

