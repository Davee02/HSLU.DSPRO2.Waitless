#!/usr/bin/env python3
"""
Script to upload trained models to Google Cloud Storage
"""

import os
import argparse
from pathlib import Path
from google.cloud import storage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_models_to_gcs(local_model_dir: str, bucket_name: str, gcs_prefix: str = "models"):
    """
    Upload all model files from local directory to Google Cloud Storage
    
    Args:
        local_model_dir: Local directory containing model files
        bucket_name: GCS bucket name
        gcs_prefix: Prefix for GCS objects (default: "models")
    """
    
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    local_path = Path(local_model_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Local model directory not found: {local_model_dir}")
    
    # Find all model files
    model_files = []
    
    # Find .pkl files (GradientBoosting models)
    gb_files = list(local_path.glob("*_gb_baseline.pkl"))
    model_files.extend(gb_files)
    
    # Find .pt files (TCN models)
    tcn_files = list(local_path.glob("*_cached_scheduled_sampling_tcn.pt"))
    model_files.extend(tcn_files)
    
    if not model_files:
        logger.warning(f"No model files found in {local_model_dir}")
        return
    
    logger.info(f"Found {len(model_files)} model files to upload")
    
    # Upload each file
    uploaded_count = 0
    total_size = 0
    
    for model_file in model_files:
        try:
            # Calculate relative path and GCS object name
            relative_path = model_file.relative_to(local_path)
            gcs_object_name = f"{gcs_prefix}/{relative_path}"
            
            # Get file size
            file_size = model_file.stat().st_size
            total_size += file_size
            
            # Upload file
            blob = bucket.blob(gcs_object_name)
            
            logger.info(f"Uploading {model_file.name} ({file_size / (1024*1024):.1f} MB) -> gs://{bucket_name}/{gcs_object_name}")
            
            blob.upload_from_filename(str(model_file))
            uploaded_count += 1
            
            logger.info(f"✅ Successfully uploaded {model_file.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to upload {model_file.name}: {e}")
    
    logger.info(f"\n📊 Upload Summary:")
    logger.info(f"   • Files uploaded: {uploaded_count}/{len(model_files)}")
    logger.info(f"   • Total size: {total_size / (1024*1024):.1f} MB")
    logger.info(f"   • Bucket: gs://{bucket_name}/{gcs_prefix}/")

def list_bucket_contents(bucket_name: str, prefix: str = "models"):
    """List contents of the GCS bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    
    logger.info(f"\n📁 Contents of gs://{bucket_name}/{prefix}/:")
    total_size = 0
    count = 0
    
    for blob in blobs:
        size_mb = blob.size / (1024 * 1024) if blob.size else 0
        total_size += blob.size if blob.size else 0
        count += 1
        logger.info(f"   • {blob.name} ({size_mb:.1f} MB)")
    
    logger.info(f"\n📊 Total: {count} files, {total_size / (1024*1024):.1f} MB")

def create_bucket_if_not_exists(bucket_name: str, location: str = "EUROPE-WEST6"):
    """Create GCS bucket if it doesn't exist"""
    client = storage.Client()
    
    try:
        bucket = client.get_bucket(bucket_name)
        logger.info(f"✅ Bucket gs://{bucket_name} already exists in location: {bucket.location}")
        return bucket
    except Exception:
        logger.info(f"Creating bucket gs://{bucket_name} in {location}...")
        bucket = client.create_bucket(bucket_name, location=location)
        logger.info(f"✅ Created bucket gs://{bucket_name} in {location}")
        return bucket

def check_authentication():
    """Check if Google Cloud authentication is properly set up"""
    try:
        from google.auth import default
        credentials, project = default()
        if project:
            logger.info(f"✅ Authentication successful. Project: {project}")
            return True
        else:
            logger.warning("⚠️  Authentication found but no default project set")
            return False
    except Exception as e:
        logger.error(f"❌ Authentication check failed: {e}")
        logger.error("\n🔧 To fix this, run:")
        logger.error("   gcloud auth login")
        logger.error("   gcloud auth application-default login")
        logger.error("   gcloud config set project YOUR_PROJECT_ID")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload ML models to Google Cloud Storage")
    parser.add_argument("--local-dir", required=True, help="Local directory containing model files")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--prefix", default="models", help="GCS prefix for uploaded files")
    parser.add_argument("--create-bucket", action="store_true", help="Create bucket if it doesn't exist")
    parser.add_argument("--list-only", action="store_true", help="Only list bucket contents, don't upload")
    parser.add_argument("--skip-auth-check", action="store_true", help="Skip authentication check")
    
    args = parser.parse_args()
    
    try:
        # Check authentication first
        if not args.skip_auth_check:
            if not check_authentication():
                logger.error("❌ Authentication setup required. Exiting.")
                return 1
        
        # Create bucket if requested
        if args.create_bucket:
            create_bucket_if_not_exists(args.bucket)
        
        if args.list_only:
            list_bucket_contents(args.bucket, args.prefix)
        else:
            # Upload models
            upload_models_to_gcs(args.local_dir, args.bucket, args.prefix)
            
            # List contents after upload
            list_bucket_contents(args.bucket, args.prefix)
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        
        # Provide helpful error messages for common issues
        if "default credentials" in str(e).lower():
            logger.error("\n🔧 Authentication Error - Run these commands:")
            logger.error("   gcloud auth login")
            logger.error("   gcloud auth application-default login")
            logger.error("   gcloud config set project YOUR_PROJECT_ID")
        elif "403" in str(e) or "permission" in str(e).lower():
            logger.error("\n🔧 Permission Error - You may need:")
            logger.error("   1. Storage Admin role on your account")
            logger.error("   2. Billing enabled on your project")
            logger.error("   3. Cloud Storage API enabled")
        elif "404" in str(e) or "not found" in str(e).lower():
            logger.error("\n🔧 Bucket Not Found - Try:")
            logger.error(f"   python upload_models.py --create-bucket --bucket {args.bucket}")
        
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

# Example usage:
# python upload_models.py --local-dir ./models/cached_scheduled_sampling --bucket your-model-bucket --create-bucket