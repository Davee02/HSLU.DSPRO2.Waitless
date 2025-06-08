# Test health check
curl https://queue-prediction-service-holfg7665a-oa.a.run.app/health

# Test prediction (without jq since you don't have it installed)
curl -X POST https://queue-prediction-service-holfg7665a-oa.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"attraction_id": "silver-star", "prediction_hours": 3}'

# Check logs (use 'read' instead of 'tail')
gcloud run services logs read queue-prediction-service --region=europe-west6 --limit=50