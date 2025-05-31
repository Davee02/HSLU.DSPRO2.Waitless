// useQueueTimes.ts

import { useState, useEffect } from 'react';
import { fetchQueueTimes, RideWaitTime } from '../services/queueTimesService';

export interface QueueTimesState {
  queueTimes: Map<string, RideWaitTime> | null;
  loading: boolean;
  error: string | null;
}

export const useQueueTimes = () => {
  const [queueTimes, setQueueTimes] = useState<Map<string, RideWaitTime> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const getQueueTimes = async () => {
      try {
        setLoading(true);
        const data = await fetchQueueTimes();
        setQueueTimes(data);
        setError(null);
      } catch (err) {
        setError('Failed to fetch queue times data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    getQueueTimes();
    // Refresh queue times data every 2 minutes
    const interval = setInterval(getQueueTimes, 2 * 60 * 1000);

    return () => clearInterval(interval);
  }, []);

  // Helper function to get wait time for a specific ride
  const getWaitTimeForRide = (rideId: string): RideWaitTime | null => {
    if (!queueTimes) return null;
    return queueTimes.get(rideId) || null;
  };

  return { 
    queueTimes, 
    loading, 
    error,
    getWaitTimeForRide
  };
};