import { useState, useEffect } from 'react';
import { Weather } from '../types';
import { fetchWeather } from '../services/weatherService';

export const useWeather = () => {
  const [weather, setWeather] = useState<Weather | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const getWeather = async () => {
      try {
        setLoading(true);
        const data = await fetchWeather();
        setWeather(data);
        setError(null);
      } catch (err) {
        setError('Failed to fetch weather data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    getWeather();
    // Refresh weather data every 5 minutes
    const interval = setInterval(getWeather, 5 * 60 * 1000);

    return () => clearInterval(interval);
  }, []);

  return { weather, loading, error };
}; 