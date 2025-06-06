import firebase_functions.scheduler_fn as scheduler_fn
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import json
from datetime import datetime

# Initialize Firebase Admin SDK (this will be done automatically in the Cloud Functions environment)
# We can keep this commented out as the environment handles it.
# if not firebase_admin._apps:
#     firebase_admin.initialize_app(credentials.ApplicationDefault())

# Initialize Firestore client - MOVE THIS INSIDE THE FUNCTION
# db = firestore.client()

API_URL = 'https://api.allorigins.win/get?url=https://queue-times.com/parks/51/queue_times.json'

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
                    "timestamp": datetime.fromisoformat(ride['last_updated'].replace('Z', '+00:00')),\
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
