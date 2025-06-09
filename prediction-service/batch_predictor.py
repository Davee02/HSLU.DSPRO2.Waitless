#!/usr/bin/env python3
"""
Batch Prediction System for Queue Times

This script runs weekly predictions for all attractions and saves them to Firestore.
It's designed to be triggered by Firebase Functions or Cloud Scheduler.
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from google.cloud import firestore
from collections import defaultdict

# Import your existing classes
from main import (
    ModelManager, WeatherService, HolidayService, 
    FirestoreDataFetcher, LOCAL_RIDE_MAPPINGS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchPredictor:
    """Handles batch predictions for all attractions"""
    
    def __init__(self):
        self.db = firestore.Client()
        self.model_manager = ModelManager()
        self.data_fetcher = FirestoreDataFetcher()
        
        # All attractions we have models for
        self.attractions = [mapping["name"] for mapping in LOCAL_RIDE_MAPPINGS]
        
    def run_weekly_predictions(self, prediction_days: int = 7) -> Dict[str, Dict]:
        """
        Run predictions for all attractions for the upcoming week
        
        Args:
            prediction_days: Number of days to predict (default: 7)
            
        Returns:
            Dictionary with results for each attraction
        """
        logger.info(f"Starting weekly predictions for {len(self.attractions)} attractions")
        
        results = {}
        
        for attraction_name in self.attractions:
            try:
                logger.info(f"🔄 Processing predictions for {attraction_name}")
                
                # Generate predictions for this attraction
                predictions = self._predict_attraction_week(attraction_name, prediction_days)
                
                if predictions:
                    # Save predictions to Firestore
                    self._save_predictions_to_firestore(attraction_name, predictions)
                    results[attraction_name] = {
                        "status": "success",
                        "predictions_count": len(predictions),
                        "date_range": f"{predictions[0]['timestamp']} to {predictions[-1]['timestamp']}"
                    }
                    logger.info(f"✅ Successfully processed {attraction_name}: {len(predictions)} predictions")
                else:
                    results[attraction_name] = {
                        "status": "failed",
                        "error": "No predictions generated"
                    }
                    logger.warning(f"❌ No predictions generated for {attraction_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {attraction_name}: {e}", exc_info=True)
                results[attraction_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Log summary
        successful = sum(1 for r in results.values() if r["status"] == "success")
        logger.info(f"Batch prediction summary: {successful}/{len(self.attractions)} attractions successful")
        
        return results
    
    def _predict_attraction_week(self, attraction_name: str, prediction_days: int) -> List[Dict]:
        """Generate weekly predictions for a single attraction with enhanced debugging"""
        try:
            logger.info(f"🔄 Starting prediction for '{attraction_name}'")
        
            # Fetch historical data with enhanced error handling
            try:
                historical_df = self.data_fetcher.fetch_historical_data(attraction_name, hours_back=72)
                if historical_df.empty:
                    logger.error(f"❌ No historical data for {attraction_name}")
                    return []
                logger.info(f"✅ Historical data found for '{attraction_name}': {len(historical_df)} records")
            except Exception as e:
                logger.error(f"❌ Historical data fetch failed for '{attraction_name}': {e}")
                return []
        
            # Preprocess data with enhanced error handling
            try:
                processed_df = self.data_fetcher.preprocess_for_prediction(historical_df, attraction_name)
                if processed_df.empty:
                    logger.error(f"❌ No data after preprocessing for {attraction_name}")
                    return []
                logger.info(f"✅ Data preprocessing completed for '{attraction_name}': {len(processed_df)} records")
            except Exception as e:
                logger.error(f"❌ Data preprocessing failed for '{attraction_name}': {e}")
                return []
        
            # Get model with enhanced error handling
            try:
                model = self.model_manager.get_model(attraction_name)
                logger.info(f"✅ Model loaded successfully for '{attraction_name}'")
            except Exception as e:
                logger.error(f"❌ Model loading failed for '{attraction_name}': {e}")
                return []
        
            # Prepare data for prediction with enhanced error handling
            try:
                prediction_df = model.preprocess_input(processed_df)
                logger.info(f"✅ Model preprocessing completed for '{attraction_name}': {len(prediction_df)} records")
            except Exception as e:
                logger.error(f"❌ Model preprocessing failed for '{attraction_name}': {e}")
                return []
        
            # Calculate prediction parameters
            # Only predict during park hours (9 AM to 7:30 PM) = 21 intervals per day (30-min intervals)
            intervals_per_day = 21  # 09:00, 09:30, 10:00, ..., 19:00, 19:30
            total_prediction_steps = prediction_days * intervals_per_day
        
            # FIXED: Start predictions from tomorrow at 9 AM in Europe/Zurich timezone
            import pytz
            zurich_tz = pytz.timezone('Europe/Zurich')
        
            # Get tomorrow in Zurich timezone
            tomorrow_utc = datetime.now(pytz.UTC) + timedelta(days=1)
            tomorrow_zurich = tomorrow_utc.astimezone(zurich_tz)
        
            # Set to 9 AM Zurich time
            start_time_zurich = tomorrow_zurich.replace(hour=9, minute=0, second=0, microsecond=0)
        
            # Convert back to UTC for consistent handling
            start_timestamp = start_time_zurich.astimezone(pytz.UTC)
            start_timestamp = pd.Timestamp(start_timestamp).tz_localize(None)  # Remove timezone for pandas
        
            logger.info(f"🎯 Generating {total_prediction_steps} predictions for '{attraction_name}' starting {start_timestamp} (9 AM Zurich time)")
            # Generate future data with weather and holidays - PASS ATTRACTION NAME!
            try:
                future_df = self._generate_weekly_future_data(start_timestamp, total_prediction_steps, attraction_name)
                if future_df.empty:
                    logger.error(f"❌ No future data generated for {attraction_name}")
                    return []
                logger.info(f"✅ Future data generated for '{attraction_name}': {len(future_df)} records")
            except Exception as e:
                logger.error(f"❌ Future data generation failed for '{attraction_name}': {e}")
                return []
        
            # Prepare future static features with enhanced error handling
            try:
                future_features = model.preprocess_input(future_df)
                future_static_features = future_features[model.static_feature_cols].values
                logger.info(f"✅ Future features prepared for '{attraction_name}': {future_static_features.shape}")
            except Exception as e:
                logger.error(f"❌ Future feature preparation failed for '{attraction_name}': {e}")
                return []
        
            # Get recent sequence for autoregressive prediction with enhanced error handling
            try:
                recent_data = prediction_df.tail(model.seq_length).copy()
            
                # Handle insufficient data
                if len(recent_data) < model.seq_length:
                    avg_wait_time = processed_df['wait_time'].mean() if not processed_df.empty else 15.0
                    padding_needed = model.seq_length - len(recent_data)
                
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
        
            # Build initial sequence with enhanced error handling
            try:
                initial_history = []
                gb_model = model.gb_model
            
                for _, row in recent_data.tail(model.seq_length).iterrows():
                    wait_time = row['wait_time']
                    static_features = np.array([row[col] for col in model.static_feature_cols])
                    gb_pred = gb_model.predict(static_features.reshape(1, -1))[0]
                    residual = wait_time - gb_pred
                    initial_history.append((wait_time, residual))
            
                logger.info(f"✅ Initial sequence built for '{attraction_name}': {len(initial_history)} steps")
            except Exception as e:
                logger.error(f"❌ Initial sequence building failed for '{attraction_name}': {e}")
                return []
        
            # Generate predictions with enhanced error handling
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
                        "model_version": "cached_scheduled_sampling_tcn_v1"
                    })
        
            logger.info(f"🎉 Successfully completed prediction for '{attraction_name}': {len(formatted_predictions)} formatted predictions")
            return formatted_predictions
        
        except Exception as e:
            logger.error(f"❌ Unexpected error generating predictions for {attraction_name}: {e}", exc_info=True)
            return []
    
    def _generate_weekly_future_data(self, start_timestamp: pd.Timestamp, total_steps: int, attraction_name: str) -> pd.DataFrame:
        """Generate future data for weekly predictions (only during park hours 09:00-19:30) - FIXED"""
        future_data = []
        current_time = start_timestamp
        steps_generated = 0

        logger.info(f"🕐 Starting time generation from {start_timestamp} for {total_steps} steps")

        while steps_generated < total_steps:
            hour = current_time.hour
            minute = current_time.minute

            # FIXED: Check if time is within operating hours: 09:00 to 19:30
            is_operating_hours = (hour >= 9 and hour < 19) or (hour == 19 and minute <= 30)

            logger.debug(f"Time {current_time.strftime('%H:%M')}: operating_hours={is_operating_hours}")

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
                logger.debug(f"Generated step {steps_generated}/{total_steps} at {current_time.strftime('%H:%M')}")

            # Move to next 30-minute interval
            current_time += pd.Timedelta(minutes=30)

            # FIXED: Skip to next day if we're past operating hours
            if hour >= 19 and minute > 30:
                next_day = current_time.date() + timedelta(days=1)
                current_time = pd.Timestamp.combine(next_day, pd.Timestamp("09:00:00").time())
                logger.info(f"🌅 Moving to next day: {current_time.strftime('%Y-%m-%d %H:%M')}")

        df = pd.DataFrame(future_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info(f"Generated {len(df)} future data points for weekly prediction")
        if len(df) > 0:
            logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            # Log first and last few timestamps to verify
            logger.info(f"First 3 times: {df['timestamp'].head(3).dt.strftime('%H:%M').tolist()}")
            logger.info(f"Last 3 times: {df['timestamp'].tail(3).dt.strftime('%H:%M').tolist()}")

        # Apply same preprocessing - PASS THE ATTRACTION NAME!
        processed_df = self.data_fetcher.preprocess_for_prediction(df, attraction_name)

        return processed_df
    
    def _save_predictions_to_firestore(self, attraction_name: str, predictions: List[Dict]) -> None:
        """Save predictions to Firestore in predictedQueueTimes collection"""
        if not predictions:
            return
        
        logger.info(f"Saving {len(predictions)} predictions for {attraction_name} to Firestore")
        
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
                    "model_version": prediction["model_version"]
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
            
            logger.info(f"✅ Successfully saved all predictions for {attraction_name}")
            
        except Exception as e:
            logger.error(f"Error saving predictions for {attraction_name}: {e}")
            raise
    
    def get_predictions_for_attraction(self, attraction_name: str, date: datetime.date) -> List[Dict]:
        """Retrieve predictions for a specific attraction and date"""
        try:
            date_string = date.strftime("%Y%m%d")
            
            collection_ref = self.db.collection("attractions").document(attraction_name).collection("predictedQueueTimes")
            
            query = collection_ref.where(
                "__name__", ">=", date_string
            ).where(
                "__name__", "<", date_string + "z"
            ).order_by("__name__")
            
            docs = query.stream()
            
            predictions = []
            for doc in docs:
                doc_data = doc.to_dict()
                predictions.append(doc_data)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error retrieving predictions for {attraction_name} on {date}: {e}")
            return []


def main():
    """Main function for batch prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Queue Time Predictions")
    parser.add_argument('--prediction-days', type=int, default=7, 
                       help='Number of days to predict (default: 7)')
    parser.add_argument('--attractions', nargs='*',
                       help='Specific attractions to predict (default: all)')
    
    args = parser.parse_args()
    
    predictor = BatchPredictor()
    
    # Override attractions if specified
    if args.attractions:
        predictor.attractions = args.attractions
    
    # Run predictions
    results = predictor.run_weekly_predictions(args.prediction_days)
    
    # Print summary
    print("\n" + "="*50)
    print("BATCH PREDICTION SUMMARY")
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