#!/bin/bash

# Configuration
PROJECT_ID="waitless-aca8b"
SERVICE_NAME="queue-prediction-service"
REGION="europe-west6"
SERVICE_ACCOUNT_NAME="queue-prediction-sa"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔐 Setting up Firestore permissions for Queue Prediction Service${NC}"

# Set project
gcloud config set project $PROJECT_ID

# Create a dedicated service account for the prediction service
echo -e "${YELLOW}👤 Creating service account...${NC}"
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
    --display-name="Queue Prediction Service Account" \
    --description="Service account for queue prediction service with Firestore access"

SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Grant necessary permissions to the service account
echo -e "${YELLOW}🔑 Granting Firestore permissions...${NC}"

# Firestore User role (read/write to Firestore)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/datastore.user"

# Cloud Storage Object Viewer (for model files)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/storage.objectViewer"

# Monitoring Metric Writer (for logging)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/monitoring.metricWriter"

# Logging Write (for logs)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/logging.logWriter"

echo -e "${GREEN}✅ Service account created and permissions granted${NC}"
echo -e "${YELLOW}📋 Service Account: ${SERVICE_ACCOUNT_EMAIL}${NC}"

# Update the Cloud Run service to use the new service account
echo -e "${YELLOW}🚀 Updating Cloud Run service to use new service account...${NC}"

gcloud run services update $SERVICE_NAME \
    --region=$REGION \
    --service-account=$SERVICE_ACCOUNT_EMAIL

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Cloud Run service updated successfully${NC}"
    echo -e "${GREEN}🔐 The service now uses the dedicated service account with proper Firestore permissions${NC}"
    
    # Test the service
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
    echo -e "${YELLOW}🔍 Testing permissions...${NC}"
    echo -e "${GREEN}🌐 Test URL: $SERVICE_URL/debug/firestore?attraction_id=silver-star${NC}"
    
else
    echo -e "${RED}❌ Failed to update Cloud Run service${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 Permissions setup complete!${NC}"
echo -e "${YELLOW}⏱️  Note: It may take a few minutes for permissions to propagate${NC}"