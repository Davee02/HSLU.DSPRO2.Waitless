import firebase_functions.scheduler_fn as scheduler_fn
from firebase_functions import https_fn
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import json
import logging
from datetime import datetime, timedelta
from firebase_functions.options import SupportedRegion

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
    print("🌙 Starting nightly batch predictions at 2 AM CET")
    logger.info("Starting nightly batch predictions")
    
    try:
        # First, clear old predictions (optional - helps keep collection clean)
        print("🧹 Clearing old predictions...")
        clear_result = clear_old_predictions()
        print(f"Cleared old predictions: {clear_result}")
        
        # Call the batch prediction endpoint on your Cloud Run service
        print("🤖 Triggering ML predictions...")
        response = requests.post(
            f"{PREDICTION_SERVICE_URL}/batch-predict",
            json={
                "prediction_days": 7,
                "trigger_source": "scheduled_function_nightly",
                "timestamp": datetime.now().isoformat()
            },
            timeout=3600  # 1 hour timeout for batch processing
        )
        
        if response.status_code == 200:
            result = response.json()
            successful = result.get('successful_attractions', 0)
            total = result.get('total_attractions', 0)
            
            logger.info(f"Nightly batch predictions completed successfully: {successful}/{total}")
            print(f"✅ Nightly predictions completed: {successful}/{total} attractions successful")
            
            # Log detailed results for monitoring
            print(f"📊 Prediction Summary:")
            print(f"   - Total attractions: {total}")
            print(f"   - Successful: {successful}")
            print(f"   - Failed: {total - successful}")
            print(f"   - Prediction days: {result.get('prediction_days', 7)}")
            
            return {"status": "success", "successful_attractions": successful, "total_attractions": total}
            
        else:
            error_msg = f"Batch predictions failed: {response.status_code} - {response.text}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}
            
    except Exception as e:
        error_msg = f"Error triggering batch predictions: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return {"status": "error", "message": str(e)}

def clear_old_predictions() -> dict:
    """
    Helper function to clear old predictions from predictedQueueTimes collections.
    Clears predictions older than 1 day to keep only fresh weekly predictions.
    """
    try:
        app = firebase_admin.get_app()
        db = firestore.client(app=app)
    except ValueError:
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.ApplicationDefault())
            app = firebase_admin.get_app()
            db = firestore.client(app=app)
        else:
            return {"error": "Firebase app not initialized"}
    
    try:
        # Get cutoff time (1 day ago)
        cutoff_time = datetime.now() - timedelta(days=1)
        cutoff_doc_id = cutoff_time.strftime("%Y%m%d%H%M%S")
        
        # Get all attractions
        attractions = db.collection('attractions').stream()
        
        total_deleted = 0
        attractions_processed = 0
        
        for attraction_doc in attractions:
            attraction_name = attraction_doc.id
            
            try:
                # Get old predictions for this attraction
                predictions_ref = attraction_doc.reference.collection('predictedQueueTimes')
                
                # FIX: Use the correct Firestore query syntax
                # Instead of where('__name__', '<', cutoff_doc_id) which requires an index
                # Use end_before with document ID filtering
                old_predictions = predictions_ref.order_by('__name__').end_before([cutoff_doc_id]).limit(500).stream()
                
                # Delete old predictions in batches
                batch = db.batch()
                batch_size = 0
                
                for pred_doc in old_predictions:
                    batch.delete(pred_doc.reference)
                    batch_size += 1
                    
                    if batch_size >= 500:  # Firestore batch limit
                        batch.commit()
                        total_deleted += batch_size
                        batch = db.batch()
                        batch_size = 0
                
                # Commit remaining deletions
                if batch_size > 0:
                    batch.commit()
                    total_deleted += batch_size
                
                attractions_processed += 1
                
            except Exception as e:
                print(f"Error clearing predictions for {attraction_name}: {e}")
                continue
        
        return {
            "status": "success",
            "total_deleted": total_deleted,
            "attractions_processed": attractions_processed
        }
        
    except Exception as e:
        print(f"Error in clear_old_predictions: {e}")
        return {"status": "error", "message": str(e)}

@https_fn.on_request(
    region="europe-west6",
    timeout_sec=540,  # 9 minutes (max for HTTP functions)
    memory=1024       # More memory for better performance
)
def manual_batch_predict(req) -> dict:
    """
    HTTP function with increased timeout for batch predictions.
    """
    print("🚀 Manual batch predict function called")
    
    # Handle CORS manually
    if req.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        return ("", 204, headers)
    
    if req.method != "POST":
        print(f"❌ Wrong method: {req.method}")
        response_data = {"error": "Only POST method allowed"}
        headers = {"Access-Control-Allow-Origin": "*"}
        return (response_data, 405, headers)
    
    try:
        print("📝 Parsing request data...")
        data = req.get_json() or {}
        prediction_days = data.get("prediction_days", 1)  # Default to 1 day for testing
        
        print(f"✅ Request parsed - prediction_days: {prediction_days}")
        
        # Test prediction service health with longer timeout
        print("🔍 Testing prediction service health...")
        try:
            health_response = requests.get(f"{PREDICTION_SERVICE_URL}/health", timeout=30)
            print(f"Health check status: {health_response.status_code}")
            
            if health_response.status_code != 200:
                print(f"❌ Prediction service health check failed: {health_response.status_code}")
                response_data = {
                    "status": "error", 
                    "message": f"Prediction service unhealthy: {health_response.status_code}"
                }
                headers = {"Access-Control-Allow-Origin": "*"}
                return (response_data, 500, headers)
                
        except Exception as health_e:
            print(f"❌ Cannot reach prediction service: {health_e}")
            response_data = {
                "status": "error",
                "message": f"Cannot reach prediction service: {str(health_e)}"
            }
            headers = {"Access-Control-Allow-Origin": "*"}
            return (response_data, 500, headers)
        
        print("✅ Prediction service is healthy")
        
        # Skip clearing predictions for now to speed up testing
        print("⏭️ Skipping clear predictions for faster testing...")
        
        # Call batch prediction with reduced timeout for Firebase Function
        print("🤖 Calling batch prediction endpoint...")
        payload = {
            "prediction_days": prediction_days,
            "trigger_source": "manual_function_fast",
            "timestamp": datetime.now().isoformat()
        }
        print(f"Payload: {payload}")
        
        # Use shorter timeout since Firebase Function has 9-minute limit
        response = requests.post(
            f"{PREDICTION_SERVICE_URL}/batch-predict",
            json=payload,
            timeout=480  # 8 minutes (leave 1 minute buffer)
        )
        
        print(f"Batch prediction response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            
            response_data = {
                "status": "success", 
                "result": result,
                "message": f"Predictions completed for {result.get('successful_attractions', 0)} attractions",
                "debug_info": {
                    "service_url": PREDICTION_SERVICE_URL,
                    "prediction_days": prediction_days,
                    "function_timestamp": datetime.now().isoformat()
                }
            }
            headers = {"Access-Control-Allow-Origin": "*"}
            return (response_data, 200, headers)
        else:
            error_text = response.text
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"Error response: {error_text}")
            
            response_data = {
                "status": "error", 
                "message": f"Batch prediction failed: {response.status_code}",
                "error_details": error_text
            }
            headers = {"Access-Control-Allow-Origin": "*"}
            return (response_data, 500, headers)
            
    except requests.exceptions.Timeout as timeout_e:
        print(f"❌ Timeout calling prediction service: {timeout_e}")
        response_data = {
            "status": "error",
            "message": "Prediction service timeout (>8 minutes)",
            "error_type": "timeout"
        }
        headers = {"Access-Control-Allow-Origin": "*"}
        return (response_data, 500, headers)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        response_data = {
            "status": "error", 
            "message": str(e),
            "error_type": "unexpected_error"
        }
        headers = {"Access-Control-Allow-Origin": "*"}
        return (response_data, 500, headers)

@https_fn.on_request(
    region="europe-west6",
    timeout_sec=60
)
def test_connection(req) -> dict:
    """
    Test connection with proper timeout.
    """
    print("🔍 Testing connection to prediction service...")
    
    if req.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        return ("", 204, headers)
    
    try:
        # Test with longer timeout
        response = requests.get(f"{PREDICTION_SERVICE_URL}/health", timeout=30)
        
        result = {
            "status": "success",
            "service_url": PREDICTION_SERVICE_URL,
            "response_status": response.status_code,
            "timestamp": datetime.now().isoformat()
        }
        
        if response.status_code == 200:
            result["health_data"] = response.json()
            print("✅ Connection test successful")
        else:
            result["error"] = f"Service returned {response.status_code}"
            print(f"⚠️ Service returned {response.status_code}")
        
        headers = {"Access-Control-Allow-Origin": "*"}
        return (result, 200, headers)
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        result = {
            "status": "error",
            "error": str(e),
            "service_url": PREDICTION_SERVICE_URL,
            "timestamp": datetime.now().isoformat()
        }
        headers = {"Access-Control-Allow-Origin": "*"}
        return (result, 500, headers)

@https_fn.on_request(region="europe-west6")  # FIXED: Removed cors=True
def get_predictions_proxy(req) -> dict:
    """
    HTTP function to retrieve predictions for a specific attraction and date.
    """
    # Handle CORS manually
    if req.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        return ("", 204, headers)
    
    if req.method != "GET":
        response_data = {"error": "Only GET method allowed"}
        headers = {"Access-Control-Allow-Origin": "*"}
        return (response_data, 405, headers)
    
    try:
        attraction_id = req.args.get("attraction_id")
        date_str = req.args.get("date")  # Format: YYYY-MM-DD
        
        if not attraction_id or not date_str:
            response_data = {"error": "attraction_id and date parameters required"}
            headers = {"Access-Control-Allow-Origin": "*"}
            return (response_data, 400, headers)
        
        # Call the prediction service
        response = requests.get(
            f"{PREDICTION_SERVICE_URL}/predictions/{attraction_id}",
            params={"date": date_str},
            timeout=30
        )
        
        headers = {"Access-Control-Allow-Origin": "*"}
        
        if response.status_code == 200:
            return (response.json(), 200, headers)
        else:
            response_data = {"error": "Failed to retrieve predictions", "details": response.text}
            return (response_data, response.status_code, headers)
            
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}")
        response_data = {"error": str(e)}
        headers = {"Access-Control-Allow-Origin": "*"}
        return (response_data, 500, headers)

@https_fn.on_request(region="europe-west6")  # FIXED: Removed cors=True
def prediction_service_health(req) -> dict:
    """
    HTTP function to check the health of the prediction service.
    """
    # Handle CORS manually
    if req.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        return ("", 204, headers)
    
    if req.method != "GET":
        response_data = {"error": "Only GET method allowed"}
        headers = {"Access-Control-Allow-Origin": "*"}
        return (response_data, 405, headers)
    
    try:
        # Check if the prediction service is healthy
        response = requests.get(f"{PREDICTION_SERVICE_URL}/health", timeout=10)
        
        headers = {"Access-Control-Allow-Origin": "*"}
        
        if response.status_code == 200:
            health_data = response.json()
            response_data = {
                "status": "healthy",
                "prediction_service": health_data,
                "firebase_function_timestamp": datetime.now().isoformat(),
                "service_url": PREDICTION_SERVICE_URL
            }
            return (response_data, 200, headers)
        else:
            response_data = {
                "status": "unhealthy",
                "error": f"Prediction service returned {response.status_code}",
                "firebase_function_timestamp": datetime.now().isoformat(),
                "service_url": PREDICTION_SERVICE_URL
            }
            return (response_data, 500, headers)
            
    except Exception as e:
        response_data = {
            "status": "error",
            "error": str(e),
            "firebase_function_timestamp": datetime.now().isoformat(),
            "service_url": PREDICTION_SERVICE_URL
        }
        headers = {"Access-Control-Allow-Origin": "*"}
        return (response_data, 500, headers)
    
@https_fn.on_request(region="europe-west6")
def create_missing_attraction_documents_corrected(req) -> dict:
    """
    Create missing attraction documents using EXACT names from Firestore screenshots.
    This replaces the previous function with correct document names.
    """
    # Handle CORS
    if req.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        return ("", 204, headers)
    
    try:
        app = firebase_admin.get_app()
        db = firestore.client(app=app)
    except ValueError:
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.ApplicationDefault())
            app = firebase_admin.get_app()
            db = firestore.client(app=app)
        else:
            headers = {"Access-Control-Allow-Origin": "*"}
            return ({"error": "Firebase app not initialized"}, 500, headers)
    
    try:
        # EXACT document names from your Firestore screenshots
        exact_firestore_names = [
            "ARTHUR",
            "Alpine Express 'Enzian'",
            "Arena of Football - Be Part of It!",
            "Atlantica SuperSplash",
            "Atlantis Adventure",
            "Ba-a-a Express",
            "Castello dei Medici",
            "Dancing Dingie",
            "Euro-Mir",
            "Euro-Tower",
            "Eurosat - CanCan Coaster",
            "Eurosat Coastiality",
            "Fjord-Rafting",
            "GRAND PRIX EDventure",
            "Jim Button – Journey through Morrowland",
            "Josefina’s Magical Imperial Journey",
            "Kolumbusjolle",
            "Madame Freudenreich Curiosités",
            "Matterhorn-Blitz",
            "Old Mac Donald's Tractor Fun",
            "Pegasus",
            "Pirates in Batavia",
            "Poppy Towers",
            "Silver Star",
            "Snorri Touren",
            "Swiss Bob Run",
            "Tirol Log Flume",
            "Vienna Wave Swing - 'Glückspilz'",
            "Vindjammer",
            "VirtualLine: Euro-Mir",
            "VirtualLine: Pirates in Batavia",
            "VirtualLine: Voletarium",
            "VirtualLine: Voltron Nevera powered by Rimac",
            "VirtualLine: WODAN - Timburcoaster",
            "VirtualLine: Water Rollercoaster Poseidon",
            "VirtualLine: blue fire Megacoaster",
            "Voletarium",
            "Volo da Vinci",
            "Voltron Nevera powered by Rimac",
            "WODAN - Timburcoaster",
            "Water rollercoaster Poseidon",
            "Whale Adventures - Northern Lights",
            "blue fire Megacoaster"
        ]
        
        created_count = 0
        skipped_existing = 0
        batch = db.batch()
        batch_size = 0
        results = []
        
        for attraction_name in exact_firestore_names:
            try:
                attraction_ref = db.collection('attractions').document(attraction_name)
                
                # Check if document already exists
                existing_doc = attraction_ref.get()
                if existing_doc.exists:
                    skipped_existing += 1
                    results.append(f"EXISTS: {attraction_name}")
                    continue
                
                # Check if this attraction has queueTimes data (to verify it's a real attraction)
                queue_times_ref = attraction_ref.collection('queueTimes')
                queue_docs = list(queue_times_ref.limit(1).stream())
                
                if len(queue_docs) > 0:
                    # Create document with basic metadata
                    attraction_data = {
                        "name": attraction_name,
                        "created_at": datetime.now(),
                        "is_active": True,
                        "source": "corrected_creation_from_screenshots",
                        "has_queue_times": True
                    }
                    
                    batch.set(attraction_ref, attraction_data)
                    batch_size += 1
                    created_count += 1
                    results.append(f"CREATED: {attraction_name}")
                    
                    # Commit batch if it gets too large
                    if batch_size >= 500:
                        batch.commit()
                        batch = db.batch()
                        batch_size = 0
                else:
                    results.append(f"NO_QUEUE_DATA: {attraction_name}")
                    
            except Exception as e:
                results.append(f"ERROR: {attraction_name} - {str(e)}")
        
        # Commit remaining documents
        if batch_size > 0:
            batch.commit()
        
        result = {
            "status": "success",
            "created_documents": created_count,
            "skipped_existing": skipped_existing,
            "total_processed": len(exact_firestore_names),
            "details": results,
            "timestamp": datetime.now().isoformat(),
            "message": f"Created {created_count} documents, skipped {skipped_existing} existing"
        }
        
        headers = {"Access-Control-Allow-Origin": "*"}
        return (result, 200, headers)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        headers = {"Access-Control-Allow-Origin": "*"}
        return (error_result, 500, headers)