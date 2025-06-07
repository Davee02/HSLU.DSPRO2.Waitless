import { useEffect, useState } from 'react';
import { collection, query, where, orderBy, getDocs, Timestamp, QuerySnapshot, DocumentData, doc } from "firebase/firestore";
import { db } from "../firebase"; // Import your Firestore instance
// Import the mapping data and normalization function from queueTimesService.ts
import { LOCAL_RIDE_MAPPINGS, normalizeRideName } from '../services/queueTimesService';


interface HistoricalDataPoint {
  id: string; // Document ID (timestamp string)
  timestamp: Timestamp; // Firestore Timestamp object
  wait_time: number;
  // Add other fields if your documents have them
}

const useHistoricalQueueTimes = (attractionId: string | undefined, selectedDate: Date | null) => {
  const [historicalData, setHistoricalData] = useState<HistoricalDataPoint[] | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchHistoricalData = async () => {
      if (!attractionId || !selectedDate) {
        setHistoricalData(null);
        setLoading(false);
        setError(null);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        // Find the corresponding Firestore document ID (attraction name) using the local ID
        const matchedMapping = LOCAL_RIDE_MAPPINGS.find(mapping => mapping.id === attractionId);

        if (!matchedMapping) {
          console.error("No Firestore mapping found for local attraction ID:", attractionId);
          setHistoricalData([]); // Set to empty array to indicate no data
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

        console.log("Querying for attraction (Firestore ID):", firestoreAttractionId, "on date (document IDs starting with):", dateString); // Add this logging


        // Get a reference to the queueTimes subcollection using the Firestore attraction ID
        const queueTimesCollectionRef = collection(doc(db, "attractions", firestoreAttractionId), "queueTimes");


        // Create a query to get documents for the selected day by filtering on the document ID
        const q = query(
          queueTimesCollectionRef,
          orderBy('__name__'), // Order by document ID (which is the timestamp string)
          where('__name__', ">=", dateString), // Filter for document IDs starting with the date string
          where('__name__', "<", dateString + 'z') // Filter for document IDs up to the next day
        );


        const querySnapshot: QuerySnapshot = await getDocs(q);

        const data: HistoricalDataPoint[] = [];
        querySnapshot.forEach((doc) => {
           // Ensure the document data matches the interface
          const docData = doc.data() as { timestamp: Timestamp, wait_time: number };
          data.push({ id: doc.id, ...docData });
        });

        console.log("Fetched historical data count:", data.length); // Log the number of documents fetched


        setHistoricalData(data);
        setLoading(false);

      } catch (err: any) {
        console.error("Error fetching historical data:", err);
        // Create a new Error object to ensure consistent error handling
        setError(new Error(err.message || "An unexpected error occurred fetching historical data."));
        setLoading(false);
      }
    };

    fetchHistoricalData();


  }, [attractionId, selectedDate]); // Re-run effect when attractionId or selectedDate changes

  return { historicalData, loading, error };
};

export { useHistoricalQueueTimes };
