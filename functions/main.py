import firebase_functions.scheduler_fn as scheduler_fn
from firebase_functions import https_fn
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import json
import logging
from datetime import datetime

# Initialize logging
logger = logging.getLogger(__name__)

# Your existing API URL for queue times
API_URL = 'https://api.allorigins.win/get?url=https://queue-times.com/parks/51/queue_times.json'

# Your Cloud Run prediction service URL
PREDICTION_SERVICE_URL = 'https://queue-prediction-service-holfg7665a-oa.a.run.app'

# ============================================================================
# EXISTING FUNCTION - Keep this exactly as it is
# ============================================================================

@scheduler_fn.on_schedule(schedule="every 5 minutes synchronized", region="europe-west6")
def fetch_realtime_queue_times(event: scheduler_fn.ScheduledEvent) -> None:
    """
    Fetches real-time queue times and saves them to Firestore if the attraction is open.
    Triggered by a scheduled event every 5 minutes.
    """
    print("Fetching real-time queue times...")

    # Initialize Firestore client HERE, inside the function
    try:
        # The SDK will be initialized by the Functions environment, so just get the app
        app = firebase_admin.get_app()
        db = firestore.client(app=app)
    except ValueError:
        # If for some reason the app isn't initialized, initialize it (less common in Functions)
        if not firebase_admin._apps:
             firebase_admin.initialize_app(credentials.ApplicationDefault())
             app = firebase_admin.get_app()
             db = firestore.client(app=app)
        else:
             # This case should ideally not happen in the Functions environment
             print("Error: Firebase app not initialized and cannot be initialized within this context.")
             return

    try:
        response = requests.get(API_URL)
        response.raise_for_status()

        proxy_data = response.json()

        if proxy_data['status']['http_code'] != 200:
            print(f"API returned status: {proxy_data['status']['http_code']}")
            return

        data = json.loads(proxy_data['contents'])

        all_api_rides = []
        if data.get('lands'):
            for land in data['lands']:
                if land.get('rides'):
                    for ride in land['rides']:
                        all_api_rides.append(ride)

        if not all_api_rides:
            print("No ride data found in API response.")
            return

        batch = db.batch()
        batch_size = 0
        max_batch_size = 500

        for ride in all_api_rides:
            if ride.get('is_open') and ride['is_open']:
                record_data = {
                    "timestamp": datetime.fromisoformat(ride['last_updated'].replace('Z', '+00:00')),
                    "wait_time": float(ride['wait_time'])
                }

                attraction_ref = db.collection('attractions').document(str(ride['name']))
                queue_times_collection_ref = attraction_ref.collection('queueTimes')

                # Use a timestamp as the document ID to ensure uniqueness and ordering
                doc_id = record_data["timestamp"].strftime("%Y%m%d%H%M%S%f")
                batch.set(queue_times_collection_ref.document(doc_id), record_data)
                batch_size += 1

                if batch_size >= max_batch_size:
                    batch.commit()
                    print(f"Committed batch with {batch_size} writes.")
                    batch = db.batch()
                    batch_size = 0

        if batch_size > 0:
            batch.commit()
            print(f"Committed final batch with {batch_size} writes.")

        print(f"Successfully saved data for {batch_size} open rides to Firestore.")

    except requests.exceptions.RequestException as e:
        print(f"Error making API call: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# ============================================================================
# NEW FUNCTIONS - Add these for prediction automation
# ============================================================================

@scheduler_fn.on_schedule(schedule="0 2 * * *", timezone="Europe/Zurich", region="europe-west6")
def run_nightly_predictions(event: scheduler_fn.ScheduledEvent) -> None:
    """
    Scheduled function that runs nightly to generate weekly predictions.
    Runs at 2 AM CET daily.
    """
    logger.info("Starting nightly batch predictions")
    
    try:
        # Call the batch prediction endpoint on your Cloud Run service
        response = requests.post(
            f"{PREDICTION_SERVICE_URL}/batch-predict",
            json={
                "prediction_days": 7,
                "trigger_source": "scheduled_function",
                "timestamp": datetime.now().isoformat()
            },
            timeout=3600  # 1 hour timeout for batch processing
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Batch predictions completed successfully")
            logger.info(f"Result: {result.get('successful_attractions', 0)}/{result.get('total_attractions', 0)} attractions successful")
            print(f"✅ Nightly predictions completed: {result.get('successful_attractions', 0)} attractions")
        else:
            logger.error(f"Batch predictions failed: {response.status_code} - {response.text}")
            print(f"❌ Nightly predictions failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error triggering batch predictions: {e}")
        print(f"❌ Error in nightly predictions: {e}")

@https_fn.on_request(region="europe-west6")
def manual_batch_predict(req) -> dict:
    """
    HTTP function to manually trigger batch predictions.
    
    Usage: POST https://your-region-your-project.cloudfunctions.net/manual_batch_predict
    Body: {"prediction_days": 7}
    """
    if req.method != "POST":
        return {"error": "Only POST method allowed"}, 405
    
    try:
        data = req.get_json() or {}
        prediction_days = data.get("prediction_days", 7)
        
        logger.info(f"Manual batch prediction triggered for {prediction_days} days")
        print(f"🚀 Manual batch prediction triggered for {prediction_days} days")
        
        # Call the batch prediction endpoint
        response = requests.post(
            f"{PREDICTION_SERVICE_URL}/batch-predict",
            json={
                "prediction_days": prediction_days,
                "trigger_source": "manual_function",
                "timestamp": datetime.now().isoformat()
            },
            timeout=3600
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Manual batch predictions completed: {result}")
            return {
                "status": "success", 
                "result": result,
                "message": f"Predictions completed for {result.get('successful_attractions', 0)} attractions"
            }
        else:
            logger.error(f"Manual batch predictions failed: {response.status_code}")
            return {"status": "error", "message": response.text}, 500
            
    except Exception as e:
        logger.error(f"Error in manual batch prediction: {e}")
        return {"status": "error", "message": str(e)}, 500

@https_fn.on_request(region="europe-west6")
def get_predictions_proxy(req) -> dict:
    """
    HTTP function to retrieve predictions for a specific attraction and date.
    
    Usage: GET https://your-region-your-project.cloudfunctions.net/get_predictions_proxy?attraction_id=silver-star&date=2025-06-09
    """
    if req.method != "GET":
        return {"error": "Only GET method allowed"}, 405
    
    try:
        attraction_id = req.args.get("attraction_id")
        date_str = req.args.get("date")  # Format: YYYY-MM-DD
        
        if not attraction_id or not date_str:
            return {"error": "attraction_id and date parameters required"}, 400
        
        # Call the prediction service
        response = requests.get(
            f"{PREDICTION_SERVICE_URL}/predictions/{attraction_id}",
            params={"date": date_str},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to retrieve predictions"}, response.status_code
            
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}")
        return {"error": str(e)}, 500

@https_fn.on_request(region="europe-west6")
def prediction_service_health(req) -> dict:
    """
    HTTP function to check the health of the prediction service.
    
    Usage: GET https://your-region-your-project.cloudfunctions.net/prediction_service_health
    """
    if req.method != "GET":
        return {"error": "Only GET method allowed"}, 405
    
    try:
        # Check if the prediction service is healthy
        response = requests.get(f"{PREDICTION_SERVICE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            return {
                "status": "healthy",
                "prediction_service": health_data,
                "firebase_function_timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "unhealthy",
                "error": f"Prediction service returned {response.status_code}",
                "firebase_function_timestamp": datetime.now().isoformat()
            }, 500
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "firebase_function_timestamp": datetime.now().isoformat()
        }, 500