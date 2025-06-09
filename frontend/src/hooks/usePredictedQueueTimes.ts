import { useEffect, useState } from 'react';
import { collection, query, where, orderBy, getDocs, Timestamp, QuerySnapshot, DocumentData, doc } from "firebase/firestore";
import { db } from "../firebase"; // Import your Firestore instance
// Import the mapping data and normalization function from queueTimesService.ts
import { LOCAL_RIDE_MAPPINGS, normalizeRideName } from '../services/queueTimesService';

interface PredictedDataPoint {
  id: string; // Document ID (timestamp string)
  timestamp: Timestamp; // Firestore Timestamp object
  predicted_wait_time: number; // Note: different field name than historical data
  wait_time: number; // Computed/rounded value for chart compatibility
  // Add other fields if your documents have them
}

const usePredictedQueueTimes = (attractionId: string | undefined, selectedDate: Date | null) => {
  const [predictedData, setPredictedData] = useState<PredictedDataPoint[] | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchPredictedData = async () => {
      if (!attractionId || !selectedDate) {
        setPredictedData(null);
        setLoading(false);
        setError(null);
        return;
      }

      // Only allow future dates for predictions
      const today = new Date();
      today.setHours(0, 0, 0, 0); // Reset time to start of day for comparison
      const selectedDateStart = new Date(selectedDate);
      selectedDateStart.setHours(0, 0, 0, 0);

      if (selectedDateStart <= today) {
        setPredictedData([]);
        setLoading(false);
        setError(new Error("Predictions are only available for future dates."));
        return;
      }

      setLoading(true);
      setError(null);

      try {
        // Find the corresponding Firestore document ID (attraction name) using the local ID
        const matchedMapping = LOCAL_RIDE_MAPPINGS.find(mapping => mapping.id === attractionId);

        if (!matchedMapping) {
          console.error("No Firestore mapping found for local attraction ID:", attractionId);
          setPredictedData([]); // Set to empty array to indicate no data
          setLoading(false);
          setError(new Error("Attraction mapping not found."));
          return;
        }

        // The Firestore document ID is the 'name' from the mapping
        const firestoreAttractionId = matchedMapping.name;

        // Format the selected date to match the date part of the document ID format (YYYYMMDD)
        const year = selectedDate.getFullYear();
        const month = (selectedDate.getMonth() + 1).toString().padStart(2, '0'); // Months are 0-indexed
        const day = selectedDate.getDate().toString().padStart(2, '0');

        const dateString = `${year}${month}${day}`; // The date part of the document ID

        console.log("Querying for predicted data for attraction (Firestore ID):", firestoreAttractionId, "on date (document IDs starting with):", dateString);

        // Get a reference to the predictedQueueTimes subcollection using the Firestore attraction ID
        const predictedQueueTimesCollectionRef = collection(doc(db, "attractions", firestoreAttractionId), "predictedQueueTimes");
        
        console.log("Collection reference path:", predictedQueueTimesCollectionRef.path);

        // Create a query to get documents for the selected day by filtering on the document ID
        const q = query(
          predictedQueueTimesCollectionRef,
          orderBy('__name__'), // Order by document ID (which is the timestamp string)
          where('__name__', ">=", dateString), // Filter for document IDs starting with the date string
          where('__name__', "<", dateString + 'z') // Filter for document IDs up to the next day
        );

        const querySnapshot: QuerySnapshot = await getDocs(q);

        const data: PredictedDataPoint[] = [];
        querySnapshot.forEach((doc) => {
          // Ensure the document data matches the interface
          const docData = doc.data() as { timestamp: Timestamp, predicted_wait_time: number };
          data.push({ 
            id: doc.id, 
            timestamp: docData.timestamp,
            predicted_wait_time: docData.predicted_wait_time,
            wait_time: Math.round(docData.predicted_wait_time) // Round to whole number for display
          });
        });

        console.log("Fetched predicted data count:", data.length); // Log the number of documents fetched

        setPredictedData(data);
        setLoading(false);

      } catch (err: any) {
        console.error("Error fetching predicted data:", err);
        // Create a new Error object to ensure consistent error handling
        setError(new Error(err.message || "An unexpected error occurred fetching predicted data."));
        setLoading(false);
      }
    };

    fetchPredictedData();

  }, [attractionId, selectedDate]); // Re-run effect when attractionId or selectedDate changes

  return { predictedData, loading, error };
};

export { usePredictedQueueTimes };