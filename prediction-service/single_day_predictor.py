#!/usr/bin/env python3
"""
Single Day Prediction System for Queue Times

This handles predicting a single day while incorporating previous predictions
into the historical data window for the TCN model.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from google.cloud import firestore

# Import your existing classes
from main import (
    ModelManager, WeatherService, HolidayService, 
    FirestoreDataFetcher, LOCAL_RIDE_MAPPINGS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleDayPredictor:
    """Handles single day predictions for all attractions with historical data integration"""
    
    def __init__(self):
        self.db = firestore.Client()
        self.model_manager = ModelManager()
        self.data_fetcher = FirestoreDataFetcher()
        
        # All attractions we have models for
        self.attractions = [mapping["name"] for mapping in LOCAL_RIDE_MAPPINGS]
        
    def predict_single_day_all_attractions(
        self, 
        target_date: datetime.date, 
        previous_predictions: Dict[str, List[Dict]] = None
    ) -> Dict[str, Dict]:
        """
        Predict wait times for a single day for all attractions.
        
        Args:
            target_date: Date to predict (must be future date)
            previous_predictions: Dict mapping attraction names to list of previous predictions
            
        Returns:
            Dictionary with results for each attraction
        """
        if previous_predictions is None:
            previous_predictions = {}
            
        logger.info(f"Starting single day prediction for {target_date}")
        logger.info(f"Previous predictions available for {len(previous_predictions)} attractions")
        
        results = {}
        
        for attraction_name in self.attractions:
            try:
                logger.info(f"🔄 Processing single day prediction for {attraction_name}")
                
                # Get previous predictions for this attraction
                attraction_prev_predictions = previous_predictions.get(attraction_name, [])
                
                # Generate predictions for this attraction
                predictions = self._predict_attraction_single_day(
                    attraction_name, 
                    target_date,
                    attraction_prev_predictions
                )
                
                if predictions:
                    # Save predictions to Firestore
                    self._save_predictions_to_firestore(attraction_name, predictions)
                    results[attraction_name] = {
                        "status": "success",
                        "predictions_count": len(predictions),
                        "target_date": target_date.isoformat(),
                        "used_previous_predictions": len(attraction_prev_predictions)
                    }
                    logger.info(f"✅ Successfully processed {attraction_name}: {len(predictions)} predictions")
                else:
                    results[attraction_name] = {
                        "status": "failed",
                        "error": "No predictions generated",
                        "target_date": target_date.isoformat()
                    }
                    logger.warning(f"❌ No predictions generated for {attraction_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {attraction_name}: {e}", exc_info=True)
                results[attraction_name] = {
                    "status": "error",
                    "error": str(e),
                    "target_date": target_date.isoformat()
                }
        
        # Log summary
        successful = sum(1 for r in results.values() if r["status"] == "success")
        logger.info(f"Single day prediction summary: {successful}/{len(self.attractions)} attractions successful for {target_date}")
        
        return results
    
    def _predict_attraction_single_day(
        self, 
        attraction_name: str, 
        target_date: datetime.date,
        previous_predictions: List[Dict] = None
    ) -> List[Dict]:
        """Generate single day predictions for one attraction with enhanced historical data integration"""
        if previous_predictions is None:
            previous_predictions = []
            
        try:
            logger.info(f"🔄 Starting single day prediction for '{attraction_name}' on {target_date}")
    
            # Fetch historical data
            try:
                historical_df = self.data_fetcher.fetch_historical_data(attraction_name, hours_back=72)
                if historical_df.empty:
                    logger.error(f"❌ No historical data for {attraction_name}")
                    return []
                logger.info(f"✅ Historical data found for '{attraction_name}': {len(historical_df)} records")
            except Exception as e:
                logger.error(f"❌ Historical data fetch failed for '{attraction_name}': {e}")
                return []
    
            # Integrate previous predictions into historical data
            if previous_predictions:
                logger.info(f"🔗 Integrating {len(previous_predictions)} previous predictions into historical data")
                historical_df = self._integrate_previous_predictions(historical_df, previous_predictions)
                logger.info(f"✅ Enhanced historical data: {len(historical_df)} records")
    
            # Preprocess data
            try:
                processed_df = self.data_fetcher.preprocess_for_prediction(historical_df, attraction_name)
                if processed_df.empty:
                    logger.error(f"❌ No data after preprocessing for {attraction_name}")
                    return []
                logger.info(f"✅ Data preprocessing completed for '{attraction_name}': {len(processed_df)} records")
            except Exception as e:
                logger.error(f"❌ Data preprocessing failed for '{attraction_name}': {e}")
                return []
    
            # Get model
            try:
                model = self.model_manager.get_model(attraction_name)
                logger.info(f"✅ Model loaded successfully for '{attraction_name}'")
            except Exception as e:
                logger.error(f"❌ Model loading failed for '{attraction_name}': {e}")
                return []
    
            # Prepare data for prediction
            try:
                prediction_df = model.preprocess_input(processed_df)
                logger.info(f"✅ Model preprocessing completed for '{attraction_name}': {len(prediction_df)} records")
            except Exception as e:
                logger.error(f"❌ Model preprocessing failed for '{attraction_name}': {e}")
                return []
    
            # Calculate prediction parameters for single day (18 intervals: 09:00-17:30 CEST)
            intervals_per_day = 18  # 09:00 to 17:30 in 30-minute intervals
            total_prediction_steps = intervals_per_day
    
            # Set start time to target date at 9 AM CEST
            import pytz
            cest_tz = pytz.timezone('Europe/Zurich')
            
            # Create target date at 9 AM CEST
            start_time_cest = cest_tz.localize(
                datetime.combine(target_date, datetime.strptime("09:00:00", "%H:%M:%S").time())
            )
    
            # Convert to UTC for consistent handling
            start_timestamp = start_time_cest.astimezone(pytz.UTC)
            start_timestamp = pd.Timestamp(start_timestamp).tz_localize(None)
    
            logger.info(f"🎯 Generating {total_prediction_steps} predictions for '{attraction_name}' on {target_date} starting {start_timestamp} (9 AM CEST)")
        
            # Generate future data for single day
            try:
                future_df = self._generate_single_day_future_data(start_timestamp, total_prediction_steps, attraction_name)
                if future_df.empty:
                    logger.error(f"❌ No future data generated for {attraction_name}")
                    return []
                logger.info(f"✅ Future data generated for '{attraction_name}': {len(future_df)} records")
            except Exception as e:
                logger.error(f"❌ Future data generation failed for '{attraction_name}': {e}")
                return []
    
            # Prepare future static features
            try:
                future_features = model.preprocess_input(future_df)
                future_static_features = future_features[model.static_feature_cols].values
                logger.info(f"✅ Future features prepared for '{attraction_name}': {future_static_features.shape}")
            except Exception as e:
                logger.error(f"❌ Future feature preparation failed for '{attraction_name}': {e}")
                return []
    
            # Get recent sequence for autoregressive prediction (48 timesteps = 24 hours)
            try:
                recent_data = prediction_df.tail(48).copy()  # TCN needs 48 timesteps
        
                # Handle insufficient data
                if len(recent_data) < 48:
                    avg_wait_time = processed_df['wait_time'].mean() if not processed_df.empty else 15.0
                    padding_needed = 48 - len(recent_data)
            
                    logger.info(f"⚠️ Padding {padding_needed} records for '{attraction_name}' (avg wait time: {avg_wait_time:.1f})")
            
                    last_timestamp = recent_data['timestamp'].iloc[-1] if not recent_data.empty else pd.Timestamp.now()
                    for i in range(padding_needed):
                        padding_data = {
                            'timestamp': [last_timestamp - pd.Timedelta(minutes=30*(i+1))],
                            'wait_time': [avg_wait_time]
                        }
                        for col in model.static_feature_cols:
                            if col in recent_data.columns:
                                padding_data[col] = [recent_data[col].iloc[-1]]
                            else:
                                padding_data[col] = [0]
                
                        recent_data = pd.concat([recent_data, pd.DataFrame(padding_data)], ignore_index=True)
            except Exception as e:
                logger.error(f"❌ Sequence preparation failed for '{attraction_name}': {e}")
                return []
    
            # Build initial sequence
            try:
                initial_history = []
                gb_model = model.gb_model
        
                for _, row in recent_data.tail(48).iterrows():  # Use exactly 48 timesteps
                    wait_time = row['wait_time']
                    static_features = np.array([row[col] for col in model.static_feature_cols])
                    gb_pred = gb_model.predict(static_features.reshape(1, -1))[0]
                    residual = wait_time - gb_pred
                    initial_history.append((wait_time, residual))
        
                logger.info(f"✅ Initial sequence built for '{attraction_name}': {len(initial_history)} steps")
            except Exception as e:
                logger.error(f"❌ Initial sequence building failed for '{attraction_name}': {e}")
                return []
    
            # Generate predictions
            try:
                initial_static = future_static_features[0] if len(future_static_features) > 0 else np.zeros(len(model.static_feature_cols))
        
                predictions = model.predict_sequence(
                    initial_static_features=initial_static,
                    initial_history=initial_history,
                    future_static_features=future_static_features,
                    horizon=min(total_prediction_steps, len(future_static_features))
                )
        
                logger.info(f"✅ Predictions generated for '{attraction_name}': {len(predictions)} predictions")
            except Exception as e:
                logger.error(f"❌ Prediction generation failed for '{attraction_name}': {e}")
                return []
    
            # Format predictions with metadata
            formatted_predictions = []
            for i, pred in enumerate(predictions):
                if i < len(future_df):
                    formatted_predictions.append({
                        "timestamp": future_df.iloc[i]['timestamp'],
                        "predicted_wait_time": max(0, pred["wait_time_prediction"]),
                        "baseline_prediction": pred["baseline_prediction"],
                        "residual_prediction": pred["residual_prediction"],
                        "temperature": future_df.iloc[i]['temperature'],
                        "rain": future_df.iloc[i]['rain'],
                        "is_holiday": any([
                            future_df.iloc[i]['is_german_holiday'],
                            future_df.iloc[i]['is_swiss_holiday'],
                            future_df.iloc[i]['is_french_holiday']
                        ]),
                        "prediction_created_at": datetime.now(),
                        "model_version": "cached_scheduled_sampling_tcn_v1",
                        "prediction_type": "single_day_incremental"
                    })
    
            logger.info(f"🎉 Successfully completed single day prediction for '{attraction_name}': {len(formatted_predictions)} formatted predictions")
            return formatted_predictions
    
        except Exception as e:
            logger.error(f"❌ Unexpected error generating single day predictions for {attraction_name}: {e}", exc_info=True)
            return []
    
    def _integrate_previous_predictions(self, historical_df: pd.DataFrame, previous_predictions: List[Dict]) -> pd.DataFrame:
        """Integrate previous predictions into historical data for TCN model input"""
        if not previous_predictions:
            return historical_df
        
        logger.info(f"Integrating {len(previous_predictions)} previous predictions into historical data")
        
        # Convert previous predictions to DataFrame format
        prediction_records = []
        for pred in previous_predictions:
            try:
                timestamp = pd.to_datetime(pred['timestamp'])
                prediction_records.append({
                    'timestamp': timestamp,
                    'wait_time': pred['predicted_wait_time'],
                    'closed': 0,  # Assume open during predictions
                    'temperature': pred.get('temperature', 20.0),
                    'rain': pred.get('rain', 0.0),
                    'temperature_unscaled': pred.get('temperature', 20.0),
                    'rain_unscaled': pred.get('rain', 0.0),
                    'is_german_holiday': 1 if pred.get('is_holiday', False) else 0,
                    'is_swiss_holiday': 1 if pred.get('is_holiday', False) else 0,
                    'is_french_holiday': 1 if pred.get('is_holiday', False) else 0,
                    'data_source': 'previous_prediction'
                })
            except Exception as e:
                logger.warning(f"Skipping invalid prediction record: {e}")
                continue
        
        if not prediction_records:
            logger.warning("No valid prediction records to integrate")
            return historical_df
        
        # Create DataFrame from predictions
        predictions_df = pd.DataFrame(prediction_records)
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        
        # Add data source marker to historical data
        historical_df = historical_df.copy()
        historical_df['data_source'] = 'historical'
        
        # Combine historical and prediction data
        combined_df = pd.concat([historical_df, predictions_df], ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates based on timestamp (prefer historical data)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')
        
        logger.info(f"Combined data: {len(historical_df)} historical + {len(predictions_df)} predictions = {len(combined_df)} total records")
        
        return combined_df
    
    def _generate_single_day_future_data(self, start_timestamp: pd.Timestamp, total_steps: int, attraction_name: str) -> pd.DataFrame:
        """Generate future data for single day predictions (09:00-17:30 CEST)"""
        import pytz
        
        # Define CEST timezone
        cest_tz = pytz.timezone('Europe/Zurich')
        
        future_data = []
        current_time = start_timestamp
        steps_generated = 0

        logger.info(f"🕐 Generating single day data from {start_timestamp} for {total_steps} steps (09:00-17:30 CEST)")

        while steps_generated < total_steps:
            # Convert to CEST for hour checking
            current_time_cest = current_time.tz_localize('UTC').astimezone(cest_tz) if current_time.tz is None else current_time.astimezone(cest_tz)
            hour = current_time_cest.hour
            minute = current_time_cest.minute

            # Check if time is within operating hours: 09:00 to 17:30 CEST
            is_operating_hours = (hour >= 9 and hour < 17) or (hour == 17 and minute <= 30)

            if is_operating_hours:
                # Get weather data for this timestamp
                weather_data = self.data_fetcher.weather_service.fetch_weather_for_timestamp(current_time)

                # Get holiday information
                holiday_data = self.data_fetcher.holiday_service.check_holidays_for_date(current_time.date())

                future_data.append({
                    'timestamp': current_time,
                    'wait_time': 0,  # Will be predicted
                    'closed': 0,     # Assume open during operating hours
                    **weather_data,
                    **holiday_data
                })

                steps_generated += 1
                logger.debug(f"Generated step {steps_generated}/{total_steps} at {current_time_cest.strftime('%H:%M')} CEST")

            # Move to next 30-minute interval
            current_time += pd.Timedelta(minutes=30)

            # Stop if we've moved past the target day
            if current_time_cest.date() != start_timestamp.tz_localize('UTC').astimezone(cest_tz).date():
                break

        df = pd.DataFrame(future_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info(f"Generated {len(df)} future data points for single day prediction")
        if len(df) > 0:
            logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Apply same preprocessing
        processed_df = self.data_fetcher.preprocess_for_prediction(df, attraction_name)

        return processed_df
    
    def _save_predictions_to_firestore(self, attraction_name: str, predictions: List[Dict]) -> None:
        """Save predictions to Firestore in predictedQueueTimes collection"""
        if not predictions:
            return
        
        logger.info(f"Saving {len(predictions)} single day predictions for {attraction_name} to Firestore")
        
        # Get reference to predictedQueueTimes subcollection
        collection_ref = self.db.collection("attractions").document(attraction_name).collection("predictedQueueTimes")
        
        # Use batched writes for efficiency
        batch = self.db.batch()
        batch_size = 0
        max_batch_size = 500  # Firestore limit
        
        try:
            for prediction in predictions:
                # Create document ID from timestamp
                timestamp = prediction["timestamp"]
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)
                
                doc_id = timestamp.strftime("%Y%m%d%H%M%S")
                
                # Prepare document data
                doc_data = {
                    "timestamp": timestamp,
                    "predicted_wait_time": float(prediction["predicted_wait_time"]),
                    "baseline_prediction": float(prediction["baseline_prediction"]),
                    "residual_prediction": float(prediction["residual_prediction"]),
                    "temperature": float(prediction["temperature"]),
                    "rain": float(prediction["rain"]),
                    "is_holiday": bool(prediction["is_holiday"]),
                    "prediction_created_at": prediction["prediction_created_at"],
                    "model_version": prediction["model_version"],
                    "prediction_type": prediction.get("prediction_type", "single_day_incremental")
                }
                
                # Add to batch
                batch.set(collection_ref.document(doc_id), doc_data)
                batch_size += 1
                
                # Commit batch if it reaches the limit
                if batch_size >= max_batch_size:
                    batch.commit()
                    logger.debug(f"Committed batch with {batch_size} predictions for {attraction_name}")
                    batch = self.db.batch()
                    batch_size = 0
            
            # Commit any remaining predictions
            if batch_size > 0:
                batch.commit()
                logger.debug(f"Committed final batch with {batch_size} predictions for {attraction_name}")
            
            logger.info(f"✅ Successfully saved all single day predictions for {attraction_name}")
            
        except Exception as e:
            logger.error(f"Error saving single day predictions for {attraction_name}: {e}")
            raise


def main():
    """Main function for testing single day prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Single Day Queue Time Predictions")
    parser.add_argument('--target-date', required=True, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--attractions', nargs='*', help='Specific attractions to predict (default: all)')
    
    args = parser.parse_args()
    
    # Parse target date
    target_date = datetime.strptime(args.target_date, "%Y-%m-%d").date()
    
    predictor = SingleDayPredictor()
    
    # Override attractions if specified
    if args.attractions:
        predictor.attractions = args.attractions
    
    # Run predictions
    results = predictor.predict_single_day_all_attractions(target_date)
    
    # Print summary
    print("\n" + "="*50)
    print(f"SINGLE DAY PREDICTION SUMMARY - {target_date}")
    print("="*50)
    
    for attraction, result in results.items():
        status = result["status"]
        if status == "success":
            print(f"✅ {attraction}: {result['predictions_count']} predictions")
        else:
            print(f"❌ {attraction}: {result.get('error', 'Unknown error')}")
    
    successful = sum(1 for r in results.values() if r["status"] == "success")
    print(f"\nTotal: {successful}/{len(results)} attractions successful")


if __name__ == "__main__":
    main()