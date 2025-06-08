#!/bin/bash

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
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting deployment of Queue Prediction Service${NC}"

# Check if required tools are installed
command -v gcloud >/dev/null 2>&1 || { echo -e "${RED}❌ gcloud CLI is required but not installed. Aborting.${NC}" >&2; exit 1; }

# Set project
echo -e "${YELLOW}📋 Setting up project configuration...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}🔧 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable firestore.googleapis.com

# Create Cloud Storage bucket for models (if it doesn't exist)
echo -e "${YELLOW}📦 Setting up model storage bucket...${NC}"
gsutil mb gs://${MODEL_BUCKET} 2>/dev/null || echo "Bucket already exists"

# Set bucket permissions
gsutil iam ch serviceAccount:$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")-compute@developer.gserviceaccount.com:objectViewer gs://${MODEL_BUCKET}

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
    --timeout 900 \
    --concurrency 100 \
    --max-instances 10 \
    --set-env-vars MODEL_BUCKET=$MODEL_BUCKET,PROJECT_ID=$PROJECT_ID \
    --service-account $(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")-compute@developer.gserviceaccount.com

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Deployment successful!${NC}"
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
    
    echo -e "${GREEN}🌐 Service URL: $SERVICE_URL${NC}"
    echo -e "${GREEN}🏥 Health check: $SERVICE_URL/health${NC}"
    echo -e "${GREEN}📊 Model status: $SERVICE_URL/models/status${NC}"
    
    # Test the health endpoint
    echo -e "${YELLOW}🔍 Testing health endpoint...${NC}"
    curl -s "$SERVICE_URL/health" | jq . || echo "Health check response received"
    
else
    echo -e "${RED}❌ Deployment failed${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 Deployment complete!${NC}"