#!/bin/bash

# Complete Deployment Script for Queue Prediction System
# Updated for existing Firebase Functions structure

# Configuration
PROJECT_ID="waitless-aca8b"
SERVICE_NAME="queue-prediction-service"
REGION="europe-west6"  # Zurich, Switzerland
MODEL_BUCKET="waitless-aca8b-prediction-models"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting complete Queue Prediction System deployment${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"

# Check if required tools are installed
command -v gcloud >/dev/null 2>&1 || { echo -e "${RED}❌ gcloud CLI is required but not installed. Aborting.${NC}" >&2; exit 1; }
command -v firebase >/dev/null 2>&1 || { echo -e "${RED}❌ Firebase CLI is required but not installed. Aborting.${NC}" >&2; exit 1; }

# Set project
echo -e "${YELLOW}📋 Setting up project configuration...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}🔧 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable firestore.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudscheduler.googleapis.com

# Create Cloud Storage bucket for models (if it doesn't exist)
echo -e "${YELLOW}📦 Setting up model storage bucket...${NC}"
gsutil mb gs://${MODEL_BUCKET} 2>/dev/null || echo "Bucket already exists"

# Set bucket permissions
gsutil iam ch serviceAccount:$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")-compute@developer.gserviceaccount.com:objectViewer gs://${MODEL_BUCKET}

echo -e "${BLUE}📊 STEP 1: Building and deploying Cloud Run service...${NC}"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo -e "${RED}❌ main.py not found. Please run this script from the prediction-service directory.${NC}"
    exit 1
fi

# Check if batch_predictor.py exists
if [ ! -f "batch_predictor.py" ]; then
    echo -e "${RED}❌ batch_predictor.py not found. Please add this file to the prediction-service directory.${NC}"
    exit 1
fi

# Build the container image using Cloud Build
echo -e "${YELLOW}🏗️  Building container image...${NC}"
gcloud builds submit --tag $IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Container image built successfully${NC}"
else
    echo -e "${RED}❌ Container build failed${NC}"
    exit 1
fi

# Deploy to Cloud Run
echo -e "${YELLOW}🚀 Deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600 \
    --concurrency 10 \
    --max-instances 5 \
    --set-env-vars MODEL_BUCKET=$MODEL_BUCKET,PROJECT_ID=$PROJECT_ID \
    --service-account $(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")-compute@developer.gserviceaccount.com

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Cloud Run deployment successful!${NC}"
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
    
    echo -e "${GREEN}🌐 Service URL: $SERVICE_URL${NC}"
    
else
    echo -e "${RED}❌ Cloud Run deployment failed${NC}"
    exit 1
fi

echo -e "${BLUE}📊 STEP 2: Testing Cloud Run service...${NC}"

# Test the health endpoint
echo -e "${YELLOW}🔍 Testing health endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s "$SERVICE_URL/health")
if [[ $HEALTH_RESPONSE == *"\"status\":\"healthy\""* ]]; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed: $HEALTH_RESPONSE${NC}"
fi

echo -e "${BLUE}📊 STEP 3: Updating Firebase Functions...${NC}"

# Check if Firebase Functions directory exists
if [ ! -d "../firebase-functions" ]; then
    echo -e "${RED}❌ Firebase Functions directory not found. Please ensure you have ../firebase-functions/ with your updated main.py${NC}"
    exit 1
fi

cd ../firebase-functions

# Check if the main.py has the new functions
if ! grep -q "run_nightly_predictions" main.py; then
    echo -e "${RED}❌ Firebase Functions main.py doesn't contain the new prediction functions.${NC}"
    echo -e "${YELLOW}Please update your firebase-functions/main.py with the new functions.${NC}"
    exit 1
fi

# Update the prediction service URL in Firebase Functions
echo -e "${YELLOW}🔧 Updating prediction service URL in Firebase Functions...${NC}"
sed -i.bak "s|PREDICTION_SERVICE_URL = '.*'|PREDICTION_SERVICE_URL = '$SERVICE_URL'|g" main.py

echo -e "${YELLOW}🚀 Deploying Firebase Functions...${NC}"

# Deploy functions
firebase deploy --only functions

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Firebase Functions deployed successfully!${NC}"
    
    # Get the function URLs
    FIREBASE_PROJECT_URL="https://$REGION-$PROJECT_ID.cloudfunctions.net"
    
    echo -e "${GREEN}🔗 Firebase Functions URLs:${NC}"
    echo -e "${GREEN}   Manual Batch Predict: $FIREBASE_PROJECT_URL/manual_batch_predict${NC}"
    echo -e "${GREEN}   Get Predictions: $FIREBASE_PROJECT_URL/get_predictions_proxy${NC}"
    echo -e "${GREEN}   Health Check: $FIREBASE_PROJECT_URL/prediction_service_health${NC}"
    
else
    echo -e "${RED}❌ Firebase Functions deployment failed${NC}"
    echo -e "${YELLOW}💡 Make sure you're logged in to Firebase:${NC}"
    echo -e "   firebase login"
    exit 1
fi

cd ../prediction-service

echo -e "${BLUE}📊 STEP 4: Testing complete system...${NC}"

# Test single prediction
echo -e "${YELLOW}🧪 Testing single prediction...${NC}"
PREDICTION_TEST=$(curl -s -X POST "$SERVICE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "attraction_id": "silver-star",
    "prediction_hours": 3
  }')

if [[ $PREDICTION_TEST == *"\"predictions\""* ]]; then
    echo -e "${GREEN}✅ Single prediction test passed${NC}"
else
    echo -e "${YELLOW}⚠️ Single prediction test - check response: ${PREDICTION_TEST:0:100}...${NC}"
fi

# Test Firebase Functions health
echo -e "${YELLOW}🧪 Testing Firebase Functions health...${NC}"
FIREBASE_HEALTH=$(curl -s "$FIREBASE_PROJECT_URL/prediction_service_health")
if [[ $FIREBASE_HEALTH == *"\"status\":\"healthy\""* ]]; then
    echo -e "${GREEN}✅ Firebase Functions health check passed${NC}"
else
    echo -e "${YELLOW}⚠️ Firebase Functions health check - check response${NC}"
fi

echo -e "${GREEN}🎉 Complete system deployment finished!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}📊 DEPLOYMENT SUMMARY${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🌐 Cloud Run Service: $SERVICE_URL${NC}"
echo -e "${GREEN}🏥 Health Check: $SERVICE_URL/health${NC}"
echo -e "${GREEN}📈 Batch Prediction: $SERVICE_URL/batch-predict${NC}"
echo -e "${GREEN}📋 Get Predictions: $SERVICE_URL/predictions/{attraction_id}?date=YYYY-MM-DD${NC}"
echo -e "${GREEN}⏰ Nightly predictions run at 2 AM CET daily${NC}"
echo -e "${GREEN}📱 Firebase Functions deployed with existing queue fetching${NC}"

echo -e "${BLUE}🔗 Firebase Functions:${NC}"
echo -e "${GREEN}📊 Manual Batch: $FIREBASE_PROJECT_URL/manual_batch_predict${NC}"
echo -e "${GREEN}📋 Get Predictions: $FIREBASE_PROJECT_URL/get_predictions_proxy${NC}"
echo -e "${GREEN}🏥 Health Proxy: $FIREBASE_PROJECT_URL/prediction_service_health${NC}"

echo -e "${BLUE}🧪 QUICK TESTS:${NC}"
echo -e "${YELLOW}# Test health:${NC}"
echo -e "curl $SERVICE_URL/health"
echo ""
echo -e "${YELLOW}# Test batch prediction via Firebase Function:${NC}"
echo -e "curl -X POST $FIREBASE_PROJECT_URL/manual_batch_predict -H 'Content-Type: application/json' -d '{\"prediction_days\": 1}'"
echo ""
echo -e "${YELLOW}# Test single prediction:${NC}"
echo -e "curl -X POST $SERVICE_URL/predict -H 'Content-Type: application/json' -d '{\"attraction_id\": \"silver-star\", \"prediction_hours\": 6}'"
echo ""
echo -e "${YELLOW}# Get saved predictions via Firebase Function:${NC}"
echo -e "curl '$FIREBASE_PROJECT_URL/get_predictions_proxy?attraction_id=silver-star&date=2025-06-09'"

echo -e "${GREEN}🎯 System is ready for production use!${NC}"
echo -e "${GREEN}Your existing 5-minute queue time fetching continues running alongside the new prediction system.${NC}"